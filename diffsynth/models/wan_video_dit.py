import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter
from ..core.gradient import gradient_checkpoint_forward
from .wantodance import WanToDanceRotaryEmbedding, WanToDanceMusicEncoderLayer

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def gather_tokens(x: torch.Tensor, indices: torch.Tensor):
    # x: [B, S, D]
    # indices: [B, N]
    gather_indices = indices.unsqueeze(-1).expand(
        -1, -1, x.shape[-1]
    )
    return torch.gather(x, dim=1, index=gather_indices)


def scatter_tokens_(
    destination: torch.Tensor,
    indices: torch.Tensor,
    source: torch.Tensor,
):
    # destination: [B, S, D]
    # source: [B, N, D]
    scatter_indices = indices.unsqueeze(-1).expand(
        -1, -1, destination.shape[-1]
    )
    destination.scatter_(1, scatter_indices, source)


def update_noise_cache(
    previous: Optional[torch.Tensor],
    selected_patches: torch.Tensor,
    noise_tokens: torch.Tensor,
) -> torch.Tensor:
    """Update a full token-noise cache only at the active token positions.

    On the first RAS step there is no cache to preserve, so the complete
    prediction initializes it. Dense steps select every token and therefore
    replace the cache in full. Sparse steps scatter only the current active
    predictions, retaining the earlier values for inactive tokens.
    """
    noise_tokens = noise_tokens.detach()
    if previous is None or selected_patches.shape[1] == noise_tokens.shape[1]:
        return noise_tokens
    updated = previous.clone()
    scatter_tokens_(updated, selected_patches, gather_tokens(noise_tokens, selected_patches))
    return updated


def cache_ready(cache, *keys):
    return (
        cache is not None
        and all(cache.get(key) is not None for key in keys)
    )


def validate_flow_magnitudes(flow: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Validate/prepare the static flow-magnitude ranking tensor.

    Flow-guided RAS ranks tokens by the optical-flow magnitude pooled from a
    dense reference video. Expected shape is [B, S] with B == x.shape[0] and
    S == x.shape[1] (the token count AFTER patchify). The tensor is moved onto
    x's device and returned; violations raise ``ValueError``.

    Args:
        flow: Optional motion-magnitude tensor [B, S] (floats, non-negative).
        x: The patched DiT input [B, S, D] the flow will rank against.

    Returns:
        ``flow`` relocated to x.device (shape validated).
    """
    if flow.ndim != 2:
        raise ValueError(
            f"flow_magnitudes must be [B, S], got shape {tuple(flow.shape)}"
        )
    if flow.shape[0] != x.shape[0] or flow.shape[1] != x.shape[1]:
        raise ValueError(
            f"flow_magnitudes shape {tuple(flow.shape)} does not match "
            f"[B={x.shape[0]}, S={x.shape[1]}] (token count after patchify)"
        )
    if flow.device != x.device:
        flow = flow.to(device=x.device)
    if not torch.isfinite(flow).all():
        raise ValueError("flow_magnitudes must be finite")
    if (flow < 0).any():
        raise ValueError("flow_magnitudes must be non-negative")
    return flow


def selection_mask_to_grid(mask: torch.Tensor, grid_size: tuple) -> torch.Tensor:
    """Convert a [B, f*h*w] bool selection mask to a [B, 1, f, h, w] float grid.

    Useful for upsampling and overlaying as a heatmap on generated frames to
    visualize which spatial regions RAS selected at each timestep.

    Args:
        mask: Boolean tensor of shape [B, f*h*w], True where a patch was selected.
        grid_size: Tuple (f, h, w) — the spatial grid dimensions (frames, height, width in patches).

    Returns:
        Float tensor of shape [B, 1, f, h, w] with 1.0 at selected positions, 0.0 elsewhere.
    """
    f, h, w = grid_size
    return mask.float().reshape(-1, 1, f, h, w)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads, selected_patches=None):
    """Apply rotary position embeddings using real-valued bf16 math.

    Decomposes the complex rotation (a+ib)(c+is) = (ac-bs) + i(as+bc)
    to operate entirely in the input dtype, avoiding float64 temporaries.

    Args:
        x: [B, S, dim] in bf16 or fp16.
        freqs: complex tensor [S, 1, D/2] where D = head_dim.
               cos/sin are extracted via .real and .imag.
        num_heads: number of attention heads.
        selected_patches: optional [B, S_active] indices into freqs.
    Returns:
        [B, S, dim] rotated tensor in the input dtype.
    """
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    dtype = x.dtype

    if selected_patches is not None:
        freqs = freqs[selected_patches]

    # Extract cos/sin from complex frequencies in the input dtype.
    # freqs is complex64 (complex32 on NPU); .real/.imag are float32 views.
    cos = freqs.real.to(dtype)  # [S, 1, D/2] or [B, S_active, 1, D/2]
    sin = freqs.imag.to(dtype)

    # Split interleaved (real, imag) pairs along the head dim.
    # unflatten(-1, (-1, 2)): [..., D] -> [..., D/2, 2]
    #   dim 0 of the pair = even-indexed (real part)
    #   dim 1 of the pair = odd-indexed  (imag part)
    x_paired = x.unflatten(-1, (-1, 2))
    x_real = x_paired[..., 0]  # [B, S, N, D/2]
    x_imag = x_paired[..., 1]  # [B, S, N, D/2]

    # Real-valued complex rotation:
    #   (x_real + i*x_imag) * (cos + i*sin)
    # = (x_real*cos - x_imag*sin) + i*(x_real*sin + x_imag*cos)
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos

    # Re-interleave: stack into [B, S, N, D/2, 2] then flatten.
    x_out = torch.stack([out_real, out_imag], dim=-1)
    return x_out.flatten(start_dim=2)


def set_to_torch_norm(models):
    for model in models:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.use_torch_norm = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_torch_norm = False
        self.normalized_shape = (dim,)

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        if self.use_torch_norm:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:
            # Compute RMS in the input dtype to avoid allocating a full
            # float32 copy of x (saves ~50% peak memory per call).
            # PyTorch reductions accumulate in float32 internally without
            # materializing a full intermediate tensor.
            return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, kv_cache=None, selected_patches=None):
        # The size of selected_patches should be [B, N_active]
        if selected_patches is not None and cache_ready(kv_cache, "k", "v"):
            k = kv_cache["k"]
            v = kv_cache["v"]

            q = self.norm_q(self.q(x))
            q = rope_apply(q, freqs, self.num_heads, selected_patches)

            k_active = self.norm_k(self.k(x))
            v_active = self.v(x)
            k_active = rope_apply(k_active, freqs, self.num_heads, selected_patches)

            scatter_tokens_(k, selected_patches, k_active)
            scatter_tokens_(v, selected_patches, v_active)

        else: 
            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(x))
            v = self.v(x)
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)

        if kv_cache is not None:
            kv_cache["k"] = k
            kv_cache["v"] = v
        
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor, ctx_kv_cache=None, selected_patches=None):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y

        if (selected_patches is not None and cache_ready(ctx_kv_cache, "k", "v")):  
            q = self.norm_q(self.q(x))
            k = ctx_kv_cache["k"]
            v = ctx_kv_cache["v"]
            x = self.attn(q, k, v)
            if self.has_image_input:
                assert ctx_kv_cache.get("k_img", 0) != 0, "ValueError: KV Cache K for input image must be computed when having image input!"
                assert ctx_kv_cache.get("v_img", 0) != 0, "ValueError: KV Cache V for input image must be computed when having image input!" 
                k_img = ctx_kv_cache["k_img"] 
                v_img = ctx_kv_cache["v_img"]
                y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
                x = x + y
                
            return self.o(x)

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y

        if ctx_kv_cache is not None:
            ctx_kv_cache["k"] = k
            ctx_kv_cache["v"] = v
            if self.has_image_input:
                ctx_kv_cache["k_img"] = k_img
                ctx_kv_cache["v_img"] = v_img

        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()


    def forward(self, x, context, t_mod, freqs, kv_cache=None, ctx_kv_cache=None, selected_patches=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, kv_cache, selected_patches))
        x = x + self.cross_attn(self.norm3(x), context, ctx_kv_cache, selected_patches)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


def wantodance_torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)
    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = wantodance_torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


class WanToDanceInjector(nn.Module):
    def __init__(self, all_modules, all_modules_names, dim=2048, num_heads=32, inject_layer=[0, 27]):
        super().__init__()
        self.injected_block_id = {}
        injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, DiTBlock):
                for inject_id in inject_layer:
                    if f'root.transformer_blocks.{inject_id}' == mod_name:
                        self.injected_block_id[inject_id] = injector_id
                        injector_id += 1

        self.injector = nn.ModuleList(
            [
                CrossAttention(
                    dim=dim,
                    num_heads=num_heads,
                )
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_feat = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6,)
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_vec = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6,)
                for _ in range(injector_id)
            ]
        )


class WanModel(torch.nn.Module):

    _repeated_blocks = ["DiTBlock"]

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads

        if wantodance_enable_dynamicfps or wantodance_enable_unimodel:
            end = int(22350 / 8 + 0.5) # 149f * 30fps * 5s = 22350
            self.freqs = precompute_freqs_cis_3d(head_dim, end=end)
        else:
            self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

        # RAS: token indices selected by the most recent forward() call. Lets the
        # negative CFG branch reuse the positive branch's selection instead of
        # re-deriving it from a different noise prediction.
        self._last_selected_patches = None

        # RAS debug: stores (timestep, mask [B, S], grid_size (f, h, w)) per RAS step
        self.debug_masks = []

        self.prepare_wantodance(in_dim, dim, num_heads, has_image_pos_emb, out_dim, patch_size, eps,
                                wantodance_enable_music_inject, wantodance_music_inject_layers, wantodance_enable_refimage, wantodance_enable_refface,
                                wantodance_enable_global, wantodance_enable_dynamicfps, wantodance_enable_unimodel)

    def prepare_wantodance(
        self,
        in_dim, dim, num_heads, has_image_pos_emb, out_dim, patch_size, eps,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
    ):
        if wantodance_enable_music_inject:
            all_modules, all_modules_names = wantodance_torch_dfs(self.blocks, parent_name="root.transformer_blocks")
            self.music_injector = WanToDanceInjector(all_modules, all_modules_names, dim=dim, num_heads=num_heads, inject_layer=wantodance_music_inject_layers)
        if wantodance_enable_refimage:
            self.img_emb_refimage = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if wantodance_enable_refface:
            self.img_emb_refface = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if wantodance_enable_global or wantodance_enable_dynamicfps or wantodance_enable_unimodel:
            music_feature_dim = 35
            ff_size = 1024
            dropout = 0.1
            latent_dim = 256
            nhead = 4
            activation = F.gelu
            rotary = WanToDanceRotaryEmbedding(dim=latent_dim)
            self.music_projection = nn.Linear(music_feature_dim, latent_dim)
            self.music_encoder = nn.Sequential()
            for _ in range(2):
                self.music_encoder.append(
                    WanToDanceMusicEncoderLayer(
                        d_model=latent_dim,
                        nhead=nhead,
                        dim_feedforward=ff_size,
                        dropout=dropout,
                        activation=activation,
                        batch_first=True,
                        rotary=rotary,
                        device='cuda',
                    )
                )
        if wantodance_enable_unimodel:
            self.patch_embedding_global = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        if wantodance_enable_unimodel:
            self.head_global = Head(dim, out_dim, patch_size, eps)
        self.wantodance_enable_music_inject = wantodance_enable_music_inject
        self.wantodance_enable_refimage = wantodance_enable_refimage
        self.wantodance_enable_refface = wantodance_enable_refface
        self.wantodance_enable_global = wantodance_enable_global
        self.wantodance_enable_dynamicfps = wantodance_enable_dynamicfps
        self.wantodance_enable_unimodel = wantodance_enable_unimodel

    def wantodance_after_transformer_block(self, block_idx, hidden_states):
        if self.wantodance_enable_music_inject:
            if block_idx in self.music_injector.injected_block_id.keys():
                audio_attn_id = self.music_injector.injected_block_id[block_idx]
                audio_emb = self.merged_audio_emb  # b f n c
                num_frames = audio_emb.shape[1]
                input_hidden_states = hidden_states.clone()  # b (f h w) c
                input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)
                attn_hidden_states = self.music_injector.injector_pre_norm_feat[audio_attn_id](input_hidden_states)
                audio_emb = rearrange(audio_emb, "b t c -> (b t) 1 c", t=num_frames)
                attn_audio_emb = audio_emb
                residual_out = self.music_injector.injector[audio_attn_id](attn_hidden_states, attn_audio_emb)
                residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
                hidden_states = hidden_states + residual_out
        return hidden_states

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None, enable_wantodance_global=False):
        if enable_wantodance_global:
            x = self.patch_embedding_global(x)
        else:
            x = self.patch_embedding(x)
        f, h, w = x.shape[2:]
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        x = rearrange(
            x,
            "b d f h w -> b (f h w) d",
        )
        return x, (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def select_region(self,
                x: torch.Tensor,
                ratio: float,
                timestep: torch.Tensor,
                context: torch.Tensor,
                skip_list: torch.Tensor,
                skip_k: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                prev_noise_tokens: Optional[torch.Tensor] = None,
                flow_magnitudes: Optional[torch.Tensor] = None,
                use_heuristics: bool = True,
                k_starvation: float = 0.5,
                **kwargs,
                ):

        # Return a selected_patches object \in [B, N_active]
        #
        # Region selection follows Liu et al. CVPR 2026:
        #   importance = 1 / (std(prev_noise) + eps)
        #   score = importance * exp(k * drop_count)
        # Higher score = more likely to be selected (top-k).
        #
        # - prev_noise_tokens: token-level guided noise from the PREVIOUS step.
        #   Low within-token variance → semantically meaningful region → select.
        # - flow_magnitudes: OPTIONAL static per-token motion magnitude pooled
        #   from the RAFT flow of a dense reference video (flow-guided mode).
        #   When supplied it REPLACES the previous-noise metric:
        #   importance = max(flow_magnitude, 1e-6). The floor keeps the
        #   multiplicative starvation term exp(k*drop_count) effective for
        #   zero-motion patches, so static regions can still be serviced after
        #   enough skipped steps.
        # - drop_count (skip_k): per-token counter of how many steps it was skipped.
        #   exp(k * drop_count) boosts long-skipped tokens to prevent starvation.
        # - Falls back to L2 norm of current latents on the first step (no prev noise).

        N = x.shape[1]
        if use_heuristics:
            if flow_magnitudes is not None:
                # Flow-guided ranking: higher motion magnitude = higher priority.
                importance = torch.clamp(flow_magnitudes, min=1e-6)
            elif prev_noise_tokens is not None:
                # Paper metric: std of predicted noise across the token dimension.
                # Lower std → model is more confident → region is semantically meaningful.
                std_noise = prev_noise_tokens.std(dim=-1)  # [B, S]
                importance = 1.0 / (std_noise + 1e-6)
            else:
                # First step: no previous noise available, use L2 norm of latents.
                importance = (x ** 2).sum(dim=-1)

            # Starvation prevention: boost tokens that have been skipped many times.
            # skip_k stores the per-token drop count (incremented each step,
            # reset for selected tokens).
            score = importance * torch.exp(k_starvation * skip_k)
            selected_patches = score.topk(min(N, math.ceil(N * ratio)), dim=-1).indices
            return selected_patches
        else:
            raise NotImplementedError

    def update_skip_record(self,
            skip_list: torch.Tensor,
            skip_k: torch.Tensor,
            selected_patches: torch.Tensor,
        ):
        """Update per-token drop counters for starvation prevention.

        True starvation accounting: only tokens that were NOT serviced this step
        accumulate a drop count. Selected tokens reset to 0; unselected tokens
        increment by 1. Call this ONCE per step from the positive CFG branch only;
        the negative branch reuses the exact positive selection and must never
        call this again (it would double-count every drop).
        """
        B = skip_k.shape[0]
        # Tick every token up first, then reset exactly the serviced tokens.
        skip_k.add_(1)
        skip_k[torch.arange(0, B, device=skip_k.device), selected_patches] = 0

    def get_selection_masks(self):
        """Return list of (timestep, binary_mask [B, S], grid_size (f, h, w)) for each RAS step.

        Call after inference to retrieve per-step region selection masks for visualization.
        """
        return self.debug_masks

    def clear_selection_masks(self):
        """Clear stored selection masks to free memory."""
        self.debug_masks = []

    def get_last_selected_patches(self):
        """Return the token indices selected by the most recent forward() call.

        Lets the negative CFG branch reuse the positive branch's selection so
        both branches process the same tokens (see RAS-Wan2.1-T2V-1.3B.py).
        """
        return self._last_selected_patches

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                skip_list: torch.Tensor = None,
                skip_k: torch.Tensor = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                kv_cache: list = None,
                ctx_kv_cache: list = None,
                selected_patches: torch.Tensor = None,
                ratio: float = 0.25,
                dumb_update: str = "Previous",
                enable_debug_masks: bool = False,
                prev_noise_tokens: Optional[torch.Tensor] = None,
                flow_magnitudes: Optional[torch.Tensor] = None,
                dumb_noise_tokens: Optional[torch.Tensor] = None,
                starvation_scale: float = 0.5,
                return_noise_tokens: bool = False,
                **kwargs,
                ):
        # RAS dumb-fill source: the previous prediction for THIS branch's
        # condition (posi branch carries prev-posi, nega carries prev-nega).
        # Falls back to the selection source so single-cache callers (no-CFG
        # paths) keep working unchanged.
        if dumb_noise_tokens is None:
            dumb_noise_tokens = prev_noise_tokens
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        # RAS: region selection with KV-cache support
        if kv_cache is not None:
            # Determine whether this is a dense step (all tokens selected)
            if selected_patches is not None:
                region_selected = selected_patches
                is_full = region_selected.shape[1] == x.shape[1]
            else:
                if flow_magnitudes is not None:
                    flow_magnitudes = validate_flow_magnitudes(flow_magnitudes, x)
                region_selected = self.select_region(
                    x, ratio, timestep, context, skip_list, skip_k, clip_feature, y,
                    prev_noise_tokens=prev_noise_tokens,
                    flow_magnitudes=flow_magnitudes,
                    k_starvation=starvation_scale,
                    **kwargs,
                )
                self.update_skip_record(skip_list, skip_k, region_selected)
                is_full = False

            # Remember the selection so the negative CFG branch can process the
            # same tokens as the positive branch (see RAS-Wan2.1-T2V-1.3B.py).
            self._last_selected_patches = region_selected

            # Safety: validate that selected indices are within the token sequence bounds.
            # A common mistake is computing S from VAE latent dims instead of patched dims.
            max_idx = region_selected.max().item()
            num_tokens = x.shape[1]
            if max_idx >= num_tokens:
                raise IndexError(
                    f"RAS selected_patches index {max_idx} is out of bounds for "
                    f"token dimension {num_tokens}. This usually means S (total tokens) "
                    f"was computed from VAE latent spatial dims "
                    f"({x.shape[1] * self.patch_size[1] * self.patch_size[2]} elements) "
                    f"instead of patched dims ({num_tokens} elements). "
                    f"Fix: S = f * (h // patch_size[1]) * (w // patch_size[2])"
                )

            if is_full:
                # Dense fast-path: all tokens selected, skip clone and gather.
                x_active = x
            else:
                x_active = gather_tokens(x, region_selected)
                # Retain a copy of the DiT input only for the fallback dumb update
                # (head on raw input) used when no prior prediction exists yet.
                if dumb_noise_tokens is None:
                    x_dumb = x.clone()

            # Store selection mask for visualization if debug mode is enabled
            if enable_debug_masks:
                mask = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=torch.bool)
                mask.scatter_(1, region_selected, True)
                t_val = int(timestep.item()) if isinstance(timestep, torch.Tensor) else timestep
                self.debug_masks.append((t_val, mask, (f, h, w)))
        else:
            x_active = x
            region_selected = None

        # DiT block processing
        for i, block in enumerate(self.blocks):
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs
                )
            else:
                blk_kv_cache = kv_cache[i] if kv_cache is not None else None
                blk_ctx_kv = ctx_kv_cache[i] if ctx_kv_cache is not None else None
                x_active = block(x_active, context, t_mod, freqs, blk_kv_cache, blk_ctx_kv, region_selected)

        # Head projection and output
        if self.training:
            noise_tokens = self.head(x, t)
            x = self.unpatchify(noise_tokens, (f, h, w))
        elif kv_cache is not None:
            # RAS eval: active tokens get full DiT + head; inactive tokens get a dumb update.
            x_active = self.head(x_active, t)
            if is_full:
                # Dense step: all tokens processed through DiT, no dumb update needed
                # A dense step refreshes every token, so no token should carry a
                # starvation penalty into the next sparse selection.
                if skip_k is not None:
                    skip_k.zero_()
                noise_tokens = x_active
                x = self.unpatchify(noise_tokens, (f, h, w))
            else:
                # Dumb update for inactive tokens; `dumb_update` selects the strategy.
                # "Previous" carries forward this branch's own last prediction from
                # `dumb_noise_tokens` (its condition-specific cache); "Zero" predicts
                # no noise for inactive tokens. Raw DiT input is not fed to head() because
                # head() is trained on the output of the DiT block stack, so applying it to
                # unprocessed tokens yields garbage predictions.
                if dumb_noise_tokens is not None:
                    if dumb_update == "Previous":
                        x_dumb = dumb_noise_tokens.clone()                  # carry forward
                    elif dumb_update == "Zero":
                        x_dumb = torch.zeros_like(dumb_noise_tokens)        # zero prediction
                    else:
                        raise ValueError(f"Unknown dumb_update: {dumb_update!r}")
                else:
                    # Fallback (no prior prediction yet): head on raw DiT input.
                    x_dumb = self.head(x_dumb, t)
                scatter_tokens_(x_dumb, region_selected, x_active)          # fresh active into x_dumb
                noise_tokens = x_dumb
                x = self.unpatchify(noise_tokens, (f, h, w))
        else:
            # Full eval (no RAS): all tokens processed through DiT
            noise_tokens = self.head(x_active, t)
            x = self.unpatchify(noise_tokens, (f, h, w))
        return (x, noise_tokens) if return_noise_tokens else x
