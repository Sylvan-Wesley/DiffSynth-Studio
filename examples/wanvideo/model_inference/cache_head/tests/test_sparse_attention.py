"""CPU-runnable tests for the pluggable sparse self-attention module.

Run from the repo root or the cache_head dir:
    pytest examples/wanvideo/model_inference/cache_head/tests/test_sparse_attention.py
"""

import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from diffsynth.models.wan_video_dit import AttentionModule, WanModel

from sparse_attention import (
    MAX_DENSE_MASK_TOKENS,
    SPARSE_PATTERNS,
    SparseAttentionModule,
    SparsePatternSpec,
    install_sparse_attention,
    materialize_mask,
    set_token_grid,
)


GRID = (3, 4, 4)          # 48 tokens: small enough for an explicit [S, S] reference
NUM_TOKENS = GRID[0] * GRID[1] * GRID[2]
NUM_HEADS = 4
DIM = 32


def qkv(batch: int = 1, seed: int = 0):
    torch.manual_seed(seed)
    return (
        torch.randn(batch, NUM_TOKENS, DIM),
        torch.randn(batch, NUM_TOKENS, DIM),
        torch.randn(batch, NUM_TOKENS, DIM),
    )


def reference_attention(q, k, v, num_heads, mask=None):
    """Explicit softmax attention, independent of the module under test."""
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads).double()
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads).double()
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads).double()
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    out = torch.softmax(scores, dim=-1) @ v
    return rearrange(out, "b n s d -> b s (n d)", n=num_heads)


# ═══════════════════════════════════════════════════════════════
# Pattern specification
# ═══════════════════════════════════════════════════════════════

def test_dense_is_the_default_pattern():
    spec = SparsePatternSpec()
    assert spec.name == "dense"
    assert spec.is_dense


def test_unknown_pattern_is_rejected():
    with pytest.raises(ValueError, match="unknown sparse pattern"):
        SparsePatternSpec(name="definitely_not_a_pattern")


@pytest.mark.parametrize("field", ["spatial_radius", "temporal_radius"])
def test_negative_radius_is_rejected(field):
    with pytest.raises(ValueError, match=f"{field} must be >= 0"):
        SparsePatternSpec(name="spatiotemporal_window", **{field: -1})


def test_window_size_counts_the_chebyshev_box():
    spec = SparsePatternSpec("spatiotemporal_window", spatial_radius=2, temporal_radius=1)
    assert spec.window_size() == 5 * 5 * 3


def test_registry_exposes_the_shipped_patterns():
    assert set(SPARSE_PATTERNS) == {"dense", "spatiotemporal_window"}


# ═══════════════════════════════════════════════════════════════
# Mask geometry
# ═══════════════════════════════════════════════════════════════

def test_dense_mask_admits_everything():
    mask = materialize_mask(SparsePatternSpec("dense"), GRID)
    assert mask.shape == (NUM_TOKENS, NUM_TOKENS)
    assert bool(mask.all())


def test_window_mask_matches_the_declared_window_for_an_interior_query():
    spec = SparsePatternSpec("spatiotemporal_window", spatial_radius=1, temporal_radius=1)
    mask = materialize_mask(spec, GRID)
    # frame 1, row 1, col 1 -> the box fits entirely inside a (3,4,4) grid
    interior = 1 * (GRID[1] * GRID[2]) + 1 * GRID[2] + 1
    assert int(mask[interior].sum()) == spec.window_size()


def test_window_mask_clips_at_the_grid_corner():
    spec = SparsePatternSpec("spatiotemporal_window", spatial_radius=1, temporal_radius=1)
    mask = materialize_mask(spec, GRID)
    # frame 0, row 0, col 0 -> one octant of the box survives
    assert int(mask[0].sum()) == 2 * 2 * 2


def test_window_mask_is_symmetric_and_self_inclusive():
    spec = SparsePatternSpec("spatiotemporal_window", spatial_radius=1, temporal_radius=2)
    mask = materialize_mask(spec, GRID)
    assert torch.equal(mask, mask.t())
    assert bool(mask.diagonal().all())


def test_window_mask_decodes_wan_token_order():
    """Neighbours along the column axis are adjacent indices; neighbours along the
    frame axis are height*width apart.  A wrong decode would silently produce a
    plausible-looking but geometrically meaningless mask."""
    spec = SparsePatternSpec("spatiotemporal_window", spatial_radius=1, temporal_radius=0)
    mask = materialize_mask(spec, GRID)
    plane = GRID[1] * GRID[2]
    query = 1 * plane + 1 * GRID[2] + 1          # (f=1, r=1, c=1)
    assert bool(mask[query, query + 1])          # same row, next column
    assert bool(mask[query, query + GRID[2]])    # next row, same column
    assert not bool(mask[query, query + plane])  # next frame -> temporal_radius=0 blocks it


def test_materializing_a_production_sized_mask_is_refused():
    huge = (21, 30, 52)  # Wan's real grid: S = 32760, so [S, S] would be 1.07 GiB
    assert huge[0] * huge[1] * huge[2] > MAX_DENSE_MASK_TOKENS
    with pytest.raises(ValueError, match="refusing to materialize"):
        materialize_mask(SparsePatternSpec("spatiotemporal_window"), huge)


# ═══════════════════════════════════════════════════════════════
# The attention module
# ═══════════════════════════════════════════════════════════════

def test_dense_module_reproduces_the_teacher_kernel_exactly():
    q, k, v = qkv()
    teacher = AttentionModule(NUM_HEADS)
    sparse = SparseAttentionModule(NUM_HEADS, SparsePatternSpec("dense"))
    sparse.set_grid(GRID)
    assert torch.equal(sparse(q, k, v), teacher(q, k, v))


def test_dense_module_needs_no_grid():
    """The dense path short-circuits to the teacher kernel, so it must not
    require the out-of-band grid stamp."""
    q, k, v = qkv()
    sparse = SparseAttentionModule(NUM_HEADS, SparsePatternSpec("dense"))
    assert sparse(q, k, v).shape == q.shape


def test_sparse_module_matches_an_explicit_masked_softmax():
    spec = SparsePatternSpec("spatiotemporal_window", spatial_radius=1, temporal_radius=1)
    q, k, v = qkv()
    module = SparseAttentionModule(NUM_HEADS, spec)
    module.set_grid(GRID)
    expected = reference_attention(q, k, v, NUM_HEADS, materialize_mask(spec, GRID))
    torch.testing.assert_close(module(q, k, v).double(), expected, rtol=1e-6, atol=1e-6)


def test_sparse_output_differs_from_dense():
    q, k, v = qkv()
    spec = SparsePatternSpec("spatiotemporal_window", spatial_radius=1, temporal_radius=0)
    sparse = SparseAttentionModule(NUM_HEADS, spec)
    sparse.set_grid(GRID)
    dense = SparseAttentionModule(NUM_HEADS, SparsePatternSpec("dense"))
    dense.set_grid(GRID)
    assert not torch.allclose(sparse(q, k, v), dense(q, k, v), atol=1e-4)


def test_sparse_module_requires_a_grid():
    q, k, v = qkv()
    module = SparseAttentionModule(NUM_HEADS, SparsePatternSpec("spatiotemporal_window"))
    with pytest.raises(RuntimeError, match="grid is unset"):
        module(q, k, v)


def test_sparse_module_rejects_a_grid_that_does_not_match_the_tokens():
    q, k, v = qkv()
    module = SparseAttentionModule(NUM_HEADS, SparsePatternSpec("spatiotemporal_window"))
    module.set_grid((2, 4, 4))  # 32 tokens, but 48 arrive
    with pytest.raises(ValueError, match="implies 32 tokens"):
        module(q, k, v)


def test_sparse_module_rejects_the_ras_token_gather_shape():
    q, k, v = qkv()
    module = SparseAttentionModule(NUM_HEADS, SparsePatternSpec("spatiotemporal_window"))
    module.set_grid(GRID)
    with pytest.raises(ValueError, match="matching query/key lengths"):
        module(q[:, : NUM_TOKENS // 2], k, v)


def test_mask_is_built_once_and_reused():
    q, k, v = qkv()
    module = SparseAttentionModule(NUM_HEADS, SparsePatternSpec("spatiotemporal_window"))
    module.set_grid(GRID)
    module(q, k, v)
    module(q, k, v)
    assert len(module._mask_cache) == 1


def test_module_batches_independently():
    spec = SparsePatternSpec("spatiotemporal_window", spatial_radius=1, temporal_radius=1)
    q, k, v = qkv(batch=3)
    module = SparseAttentionModule(NUM_HEADS, spec)
    module.set_grid(GRID)
    batched = module(q, k, v)
    for i in range(3):
        single = module(q[i : i + 1], k[i : i + 1], v[i : i + 1])
        torch.testing.assert_close(batched[i : i + 1], single)


# ═══════════════════════════════════════════════════════════════
# Installation into a DiT
# ═══════════════════════════════════════════════════════════════

def tiny_dit(num_layers: int = 3) -> WanModel:
    torch.manual_seed(0)
    return WanModel(
        dim=DIM, in_dim=16, ffn_dim=64, out_dim=16, text_dim=48, freq_dim=16, eps=1e-6,
        patch_size=(1, 2, 2), num_heads=NUM_HEADS, num_layers=num_layers, has_image_input=False,
    ).eval()


def test_install_replaces_every_self_attention():
    dit = tiny_dit()
    converted = install_sparse_attention(dit, SparsePatternSpec("spatiotemporal_window"))
    assert converted == len(dit.blocks)
    assert all(isinstance(b.self_attn.attn, SparseAttentionModule) for b in dit.blocks)


def test_install_leaves_cross_attention_alone():
    """Cross-attention maps video tokens to text tokens, where the spatial grid
    has no meaning; a sparse mask there would be nonsense."""
    dit = tiny_dit()
    install_sparse_attention(dit, SparsePatternSpec("spatiotemporal_window"))
    assert all(isinstance(b.cross_attn.attn, AttentionModule) for b in dit.blocks)


def test_install_does_not_perturb_the_state_dict():
    """AttentionModule owns no parameters, which is what makes teacher weight
    inheritance free."""
    dit = tiny_dit()
    before = {k: v.clone() for k, v in dit.state_dict().items()}
    install_sparse_attention(dit, SparsePatternSpec("spatiotemporal_window"))
    after = dit.state_dict()
    assert set(before) == set(after)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_set_token_grid_reaches_every_block():
    dit = tiny_dit()
    install_sparse_attention(dit, SparsePatternSpec("spatiotemporal_window"))
    set_token_grid(dit, GRID)
    assert all(b.self_attn.attn.grid == GRID for b in dit.blocks)
    set_token_grid(dit, None)
    assert all(b.self_attn.attn.grid is None for b in dit.blocks)
