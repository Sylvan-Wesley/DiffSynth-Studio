"""Capture and inspect pairwise self-attention in the 15-step Wan teacher.

The capture is intentionally side-channel only: Wan's configured attention
backend still produces the model output.  For the positive CFG branch, this
script recomputes the requested query rows in float32, averages probabilities
across heads, and stores them as losslessly-compressed 8-bit PNG matrices.

Examples:
    python visualize_wan_teacher_attention.py capture
    python visualize_wan_teacher_attention.py serve --output-dir wan_teacher_attention
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import math
from pathlib import Path
import socketserver
from typing import Callable

import numpy as np
from PIL import Image
import torch


DEFAULT_PROMPT = (
    "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，"
    "神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，"
    "偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和"
    "四周草地的生机。中景侧面移动视角。"
)
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
LOG10_FLOOR = -8.0
LOG10_CEILING = 0.0


def frame_token_indices(grid: tuple[int, int, int], frame: int, device=None) -> torch.Tensor:
    """Return flattened token indices for one frame in Wan's f-major ordering."""
    frames, height, width = grid
    if not 0 <= frame < frames:
        raise ValueError(f"latent frame {frame} is outside [0, {frames - 1}]")
    start = frame * height * width
    return torch.arange(start, start + height * width, device=device, dtype=torch.long)


def token_index_to_coordinate(index: int, grid: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert a flattened Wan token index to (frame, row, column)."""
    frames, height, width = grid
    count = frames * height * width
    if not 0 <= index < count:
        raise ValueError(f"token index {index} is outside [0, {count - 1}]")
    frame, spatial = divmod(index, height * width)
    row, column = divmod(spatial, width)
    return frame, row, column


def head_averaged_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    query_indices: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Compute mean_h softmax(Q_h K_h^T / sqrt(d)) for selected query rows.

    The result is float32 on the input device with shape [Q, K].  Heads are
    processed serially to avoid materializing [H, Q, K] for the real Wan grid.
    """
    if q.ndim != 3 or k.ndim != 3 or q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError("attention capture currently requires q/k shaped [1, tokens, dim]")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] % num_heads:
        raise ValueError("q/k dimensions must match and be divisible by num_heads")

    query_indices = query_indices.to(q.device)
    head_dim = q.shape[-1] // num_heads
    q_heads = q.index_select(1, query_indices).reshape(1, -1, num_heads, head_dim)
    k_heads = k.reshape(1, -1, num_heads, head_dim)
    result = torch.zeros(
        (query_indices.numel(), k.shape[1]), device=q.device, dtype=torch.float32
    )
    scale = 1.0 / math.sqrt(head_dim)
    for head in range(num_heads):
        q_head = q_heads[0, :, head].float()
        k_head = k_heads[0, :, head].float()
        probabilities = torch.softmax((q_head @ k_head.transpose(0, 1)) * scale, dim=-1)
        result.add_(probabilities)
    return result.div_(num_heads)


def encode_log_probabilities(
    probabilities: torch.Tensor,
    log10_floor: float = LOG10_FLOOR,
    log10_ceiling: float = LOG10_CEILING,
) -> torch.Tensor:
    """Quantize probabilities to a shared uint8 log10 scale."""
    if log10_floor >= log10_ceiling:
        raise ValueError("log10_floor must be smaller than log10_ceiling")
    values = probabilities.float().clamp_min(10.0**log10_floor).log10()
    values = values.clamp(log10_floor, log10_ceiling)
    values = (values - log10_floor) / (log10_ceiling - log10_floor)
    return values.mul(255.0).round().to(torch.uint8)


def decode_log_probabilities(
    encoded: torch.Tensor,
    log10_floor: float = LOG10_FLOOR,
    log10_ceiling: float = LOG10_CEILING,
) -> torch.Tensor:
    """Approximately invert :func:`encode_log_probabilities`."""
    exponent = encoded.float().div(255.0)
    exponent = exponent * (log10_ceiling - log10_floor) + log10_floor
    return torch.pow(10.0, exponent)


class WanSelfAttentionCapture:
    """Runtime wrapper around each Wan block's self-attention kernel input."""

    def __init__(
        self,
        dit,
        output_dir: Path,
        grid: tuple[int, int, int],
        query_frame: int,
    ):
        self.dit = dit
        self.output_dir = Path(output_dir)
        self.grid = tuple(int(value) for value in grid)
        self.query_frame = query_frame
        self.query_indices = frame_token_indices(self.grid, query_frame)
        self.enabled = False
        self.step_index: int | None = None
        self.timestep: float | None = None
        self.captured_layers: set[int] = set()
        self.step_records: list[dict] = []
        self._original_forwards: list[tuple[object, Callable]] = []

    def install(self) -> None:
        if self._original_forwards:
            raise RuntimeError("attention capture is already installed")
        for layer_index, block in enumerate(self.dit.blocks):
            module = block.self_attn.attn
            original = module.forward

            def wrapped(q, k, v, *, _original=original, _layer=layer_index):
                if self.enabled:
                    self._capture_layer(_layer, q, k)
                return _original(q, k, v)

            self._original_forwards.append((module, original))
            module.forward = wrapped

    def restore(self) -> None:
        for module, original in self._original_forwards:
            module.forward = original
        self._original_forwards.clear()

    def start_step(self, step_index: int, timestep: float) -> None:
        if self.enabled:
            raise RuntimeError("the previous capture step is still active")
        self.step_index = int(step_index)
        self.timestep = float(timestep)
        self.captured_layers = set()
        self.enabled = True

    def finish_step(self) -> None:
        self.enabled = False
        expected = set(range(len(self.dit.blocks)))
        if self.captured_layers != expected:
            missing = sorted(expected - self.captured_layers)
            raise RuntimeError(f"capture step {self.step_index} missed layers {missing}")
        self.step_records.append(
            {
                "index": self.step_index,
                "timestep": self.timestep,
                "layers": [
                    f"maps/step_{self.step_index:03d}/layer_{layer:02d}.png"
                    for layer in range(len(self.dit.blocks))
                ],
            }
        )

    def _capture_layer(self, layer_index: int, q: torch.Tensor, k: torch.Tensor) -> None:
        if self.step_index is None or self.timestep is None:
            raise RuntimeError("start_step must be called before the teacher forward")
        if layer_index in self.captured_layers:
            raise RuntimeError(f"layer {layer_index} was captured twice in one teacher forward")
        expected_tokens = math.prod(self.grid)
        if q.shape[1] != expected_tokens or k.shape[1] != expected_tokens:
            raise ValueError(
                f"expected {expected_tokens} dense self-attention tokens for grid {self.grid}, "
                f"got q={q.shape[1]} and k={k.shape[1]}"
            )

        probabilities = head_averaged_attention(
            q, k, self.query_indices, self.dit.blocks[layer_index].self_attn.num_heads
        )
        encoded = encode_log_probabilities(probabilities).cpu().numpy()
        layer_dir = self.output_dir / "maps" / f"step_{self.step_index:03d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        path = layer_dir / f"layer_{layer_index:02d}.png"
        Image.fromarray(encoded, mode="L").save(path, format="PNG", compress_level=1)
        self.captured_layers.add(layer_index)


VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wan Teacher Pairwise Self-Attention</title>
<style>
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
body { margin: 0; background: #f6f8fb; color: #172033; }
header { padding: 18px 24px 12px; background: white; border-bottom: 1px solid #dce2ec; }
h1 { margin: 0 0 5px; font-size: 21px; font-weight: 650; }
#subtitle { color: #596579; font-size: 13px; }
.controls { display: flex; flex-wrap: wrap; gap: 14px; padding: 14px 24px; background: white; }
label { display: grid; gap: 5px; color: #596579; font-size: 12px; font-weight: 600; }
select, input { min-width: 130px; padding: 7px 9px; border: 1px solid #bec8d7; border-radius: 6px; background: white; }
button { align-self: end; padding: 8px 12px; border: 1px solid #9eabbd; border-radius: 6px; background: #f7f9fc; color: #27344a; cursor: pointer; }
.main { padding: 16px 24px 24px; }
.legend { display: flex; align-items: center; gap: 9px; margin-bottom: 10px; font: 12px ui-monospace, monospace; }
.legend-bar { width: 250px; height: 12px; background: linear-gradient(90deg,#081d58,#225ea8,#41b6c4,#c7e9b4,#ffffd9); border: 1px solid #aeb8c6; }
#viewport { overflow: auto; max-height: calc(100vh - 245px); min-height: 430px; background: #fff; border: 1px solid #cfd7e3; }
#heatmap { display: block; image-rendering: pixelated; }
#status { margin-top: 9px; min-height: 20px; font: 12px ui-monospace, monospace; color: #364155; }
.note { margin-top: 8px; color: #687489; font-size: 12px; }
</style>
</head>
<body>
<header><h1>Wan teacher pairwise self-attention</h1><div id="subtitle">Loading metadata…</div></header>
<div class="controls">
  <label>Denoising step<select id="step"></select></label>
  <label>Transformer layer<select id="layer"></select></label>
  <label>Key-frame view<select id="keyFrame"></select></label>
  <label>Zoom<input id="zoom" type="range" min="0.15" max="4" value="1" step="0.05"></label>
  <button id="reset" type="button">Reset view</button>
</div>
<main class="main">
  <div class="legend"><span>≤10⁻⁸</span><div class="legend-bar"></div><span>1</span><span>(log₁₀ attention probability)</span></div>
  <div id="viewport"><canvas id="heatmap"></canvas></div>
  <div id="status">Move over the heatmap to inspect a token pair.</div>
  <div class="note">Rows are query tokens from the selected query frame. Columns are key tokens, ordered frame → row → column. Probabilities are head-averaged after softmax.</div>
</main>
<script>
const stepSelect = document.getElementById('step');
const layerSelect = document.getElementById('layer');
const frameSelect = document.getElementById('keyFrame');
const zoomInput = document.getElementById('zoom');
const canvas = document.getElementById('heatmap');
const viewport = document.getElementById('viewport');
const status = document.getElementById('status');
const context = canvas.getContext('2d', {alpha: false});
let metadata, encoded, sourceWidth, sourceHeight, keyStart = 0, displayedWidth = 0;

const paletteStops = [
  [0.00,[8,29,88]], [0.25,[34,94,168]], [0.50,[65,182,196]],
  [0.75,[199,233,180]], [1.00,[255,255,217]]
];
function palette(value) {
  const t = value / 255;
  let i = 0;
  while (i + 1 < paletteStops.length && t > paletteStops[i + 1][0]) i++;
  const a = paletteStops[i], b = paletteStops[Math.min(i + 1, paletteStops.length - 1)];
  const u = a[0] === b[0] ? 0 : (t - a[0]) / (b[0] - a[0]);
  return a[1].map((x,j) => Math.round(x + u * (b[1][j] - x)));
}
function coordinate(index) {
  const [,h,w] = metadata.grid;
  const frameSize = h*w;
  const frame = Math.floor(index/frameSize), rem = index%frameSize;
  return [frame, Math.floor(rem/w), rem%w];
}
function probability(byte) {
  if (byte === 0) return '≤1.000e-8';
  return Math.pow(10, -8 + (byte/255)*8).toExponential(4);
}
function render() {
  if (!encoded) return;
  const frame = frameSelect.value;
  const frameSize = metadata.grid[1]*metadata.grid[2];
  keyStart = frame === 'all' ? 0 : Number(frame)*frameSize;
  displayedWidth = frame === 'all' ? sourceWidth : frameSize;
  canvas.width = displayedWidth;
  canvas.height = sourceHeight;
  const image = context.createImageData(displayedWidth, sourceHeight);
  for (let row=0; row<sourceHeight; row++) {
    const sourceOffset = row*sourceWidth + keyStart;
    const targetOffset = row*displayedWidth;
    for (let col=0; col<displayedWidth; col++) {
      const byte = encoded[sourceOffset+col], rgb = palette(byte), p = (targetOffset+col)*4;
      image.data[p]=rgb[0]; image.data[p+1]=rgb[1]; image.data[p+2]=rgb[2]; image.data[p+3]=255;
    }
  }
  context.putImageData(image,0,0);
  const zoom = Number(zoomInput.value);
  canvas.style.width = `${displayedWidth*zoom}px`;
  canvas.style.height = `${sourceHeight*zoom}px`;
}
async function loadMatrix() {
  status.textContent = 'Loading matrix…';
  const step = Number(stepSelect.value), layer = Number(layerSelect.value);
  const image = new Image();
  image.src = metadata.steps[step].layers[layer] + `?v=${Date.now()}`;
  await image.decode();
  const raw = document.createElement('canvas'); raw.width=image.naturalWidth; raw.height=image.naturalHeight;
  const rawContext = raw.getContext('2d'); rawContext.drawImage(image,0,0);
  const rgba = rawContext.getImageData(0,0,raw.width,raw.height).data;
  sourceWidth=raw.width; sourceHeight=raw.height; encoded=new Uint8Array(sourceWidth*sourceHeight);
  for (let i=0; i<encoded.length; i++) encoded[i]=rgba[i*4];
  document.getElementById('subtitle').textContent =
    `${metadata.model_id} · step ${step} · timestep ${metadata.steps[step].timestep.toFixed(3)} · layer ${layer}`;
  status.textContent = 'Move over the heatmap to inspect a token pair.';
  render();
}
canvas.addEventListener('mousemove', event => {
  if (!encoded) return;
  const rect=canvas.getBoundingClientRect();
  const col=Math.min(displayedWidth-1,Math.max(0,Math.floor((event.clientX-rect.left)*displayedWidth/rect.width)));
  const row=Math.min(sourceHeight-1,Math.max(0,Math.floor((event.clientY-rect.top)*sourceHeight/rect.height)));
  const keyIndex=keyStart+col, queryIndex=metadata.query_token_start+row;
  const q=coordinate(queryIndex), k=coordinate(keyIndex), byte=encoded[row*sourceWidth+keyIndex];
  status.textContent=`query (${q.join(', ')}) → key (${k.join(', ')}) · attention ${probability(byte)} · encoded ${byte}/255`;
});
zoomInput.addEventListener('input', render);
frameSelect.addEventListener('change', render);
stepSelect.addEventListener('change', loadMatrix);
layerSelect.addEventListener('change', loadMatrix);
document.getElementById('reset').addEventListener('click', () => {
  frameSelect.value=String(metadata.query_frame); zoomInput.value='1';
  viewport.scrollLeft=0; viewport.scrollTop=0; render();
});

fetch('metadata.json').then(response => response.json()).then(data => {
  metadata=data;
  data.steps.forEach(step => stepSelect.add(new Option(`Step ${step.index} · t=${step.timestep.toFixed(1)}`,step.index)));
  for (let layer=0; layer<data.num_layers; layer++) layerSelect.add(new Option(`Layer ${layer}`,layer));
  frameSelect.add(new Option('All key frames','all'));
  for (let frame=0; frame<data.grid[0]; frame++) frameSelect.add(new Option(`Key frame ${frame}`,frame));
  frameSelect.value=String(data.query_frame);
  loadMatrix();
}).catch(error => { status.textContent=`Could not load viewer data: ${error}`; });
</script>
</body>
</html>
"""


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = list(path.iterdir())
    if existing and not overwrite:
        raise FileExistsError(
            f"output directory {path} is not empty; choose another directory or pass --overwrite"
        )


def write_viewer(output_dir: Path, metadata: dict) -> None:
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "viewer.html").write_text(VIEWER_HTML, encoding="utf-8")


def run_capture(args) -> None:
    from tqdm import tqdm

    from diffsynth.models.wan_video_dit import set_to_torch_norm
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    from diffsynth.utils.data import save_video

    output_dir = Path(args.output_dir).resolve()
    prepare_output_directory(output_dir, args.overwrite)

    print("Loading Wan teacher...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(
            model_id=args.model_id, origin_file_pattern="google/umt5-xxl/"
        ),
    )
    dit, vae, scheduler = pipe.dit, pipe.vae, pipe.scheduler
    set_to_torch_norm([dit])
    dit.eval()

    @torch.no_grad()
    def encode_prompt(text: str) -> torch.Tensor:
        ids, mask = pipe.tokenizer(text, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(pipe.device), mask.to(pipe.device)
        lengths = mask.gt(0).sum(dim=1).long()
        embeddings = pipe.text_encoder(ids, mask)
        for batch_index, length in enumerate(lengths):
            embeddings[batch_index, length:] = 0
        return embeddings

    print("Encoding prompts...")
    context_positive = encode_prompt(args.prompt)
    context_negative = encode_prompt(args.negative_prompt) if args.cfg_scale != 1.0 else None

    latent_frames = (args.num_frames - 1) // 4 + 1
    latent_height = args.height // vae.upsampling_factor
    latent_width = args.width // vae.upsampling_factor
    latent_shape = (1, vae.model.z_dim, latent_frames, latent_height, latent_width)
    latents = pipe.generate_noise(latent_shape, seed=args.seed, rand_device="cpu")
    latents = latents.to(dtype=pipe.torch_dtype, device=pipe.device)
    patch_t, patch_h, patch_w = dit.patch_size
    grid = (latent_frames // patch_t, latent_height // patch_h, latent_width // patch_w)
    frame_token_indices(grid, args.latent_frame)

    scheduler.set_timesteps(
        args.num_inference_steps, denoising_strength=1.0, shift=args.sigma_shift
    )
    capture = WanSelfAttentionCapture(dit, output_dir, grid, args.latent_frame)
    capture.install()

    for model in (pipe.text_encoder, pipe.vae, pipe.image_encoder, pipe.motion_controller):
        if model is not None:
            model.to("cpu")
    torch.cuda.empty_cache()

    print(
        f"Capturing {args.num_inference_steps} steps × {len(dit.blocks)} layers; "
        f"matrix shape={grid[1] * grid[2]}×{math.prod(grid)}"
    )
    try:
        with torch.inference_mode():
            for step_index, scheduler_timestep in enumerate(tqdm(scheduler.timesteps)):
                timestep = scheduler_timestep.unsqueeze(0).to(
                    dtype=pipe.torch_dtype, device=pipe.device
                )
                capture.start_step(step_index, scheduler_timestep.item())
                noise_positive = dit(x=latents, timestep=timestep, context=context_positive)
                capture.finish_step()

                if args.cfg_scale != 1.0:
                    noise_negative = dit(x=latents, timestep=timestep, context=context_negative)
                    noise = noise_negative + args.cfg_scale * (noise_positive - noise_negative)
                else:
                    noise = noise_positive
                latents = scheduler.step(noise, scheduler_timestep, latents)
    finally:
        capture.enabled = False
        capture.restore()

    metadata = {
        "format_version": 1,
        "title": "Wan teacher pairwise self-attention",
        "model_id": args.model_id,
        "num_inference_steps": args.num_inference_steps,
        "num_layers": len(dit.blocks),
        "num_heads": dit.blocks[0].self_attn.num_heads,
        "grid": list(grid),
        "query_frame": args.latent_frame,
        "query_token_start": args.latent_frame * grid[1] * grid[2],
        "matrix_shape": [grid[1] * grid[2], math.prod(grid)],
        "flattening": "frame-major, then row-major, then column-major",
        "value": "mean across heads of softmax(QK^T/sqrt(head_dim))",
        "branch": "positive CFG branch",
        "encoding": {
            "dtype": "uint8",
            "transform": "round(255 * (clip(log10(p), -8, 0) + 8) / 8)",
            "log10_floor": LOG10_FLOOR,
            "log10_ceiling": LOG10_CEILING,
            "clipped_label": "<=1e-8",
        },
        "generation": {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "cfg_scale": args.cfg_scale,
            "sigma_shift": args.sigma_shift,
        },
        "steps": capture.step_records,
    }
    write_viewer(output_dir, metadata)

    if not args.skip_video:
        print("Decoding teacher video...")
        dit.to("cpu")
        torch.cuda.empty_cache()
        vae.to(pipe.device)
        video = vae.batched_tiled_decode(
            latents,
            device=pipe.device,
            tile_size=(30, 52),
            tile_stride=(15, 26),
            tile_batch_size=2,
        )
        video = pipe.vae_output_to_video(video)
        save_video(video, str(output_dir / "teacher.mp4"), fps=15, quality=5)

    print(f"Capture complete: {output_dir}")
    print(
        "View with: python "
        f"{Path(__file__).name} serve --output-dir {json.dumps(str(output_dir))}"
    )


def run_server(args) -> None:
    output_dir = Path(args.output_dir).resolve()
    if not (output_dir / "viewer.html").is_file():
        raise FileNotFoundError(f"{output_dir} does not contain viewer.html")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(output_dir))
    with socketserver.ThreadingTCPServer((args.host, args.port), handler) as server:
        url = f"http://{args.host}:{args.port}/viewer.html"
        print(f"Serving Wan attention viewer at {url} (Ctrl-C to stop)")
        server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="run the teacher and capture attention")
    capture.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    capture.add_argument("--prompt", default=DEFAULT_PROMPT)
    capture.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    capture.add_argument("--num-inference-steps", type=int, default=15)
    capture.add_argument("--latent-frame", type=int, default=10)
    capture.add_argument("--height", type=int, default=480)
    capture.add_argument("--width", type=int, default=832)
    capture.add_argument("--num-frames", type=int, default=81)
    capture.add_argument("--cfg-scale", type=float, default=5.0)
    capture.add_argument("--sigma-shift", type=float, default=5.0)
    capture.add_argument("--seed", type=int, default=0)
    capture.add_argument("--device", default="cuda")
    capture.add_argument("--output-dir", default="wan_teacher_attention")
    capture.add_argument("--overwrite", action="store_true")
    capture.add_argument("--skip-video", action="store_true")
    capture.set_defaults(func=run_capture)

    serve = subparsers.add_parser("serve", help="serve an existing capture locally")
    serve.add_argument("--output-dir", default="wan_teacher_attention")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=run_server)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
