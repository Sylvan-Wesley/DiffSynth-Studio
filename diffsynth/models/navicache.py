"""NaviCache: test-time self-calibration caching for video diffusion.

Port of NaviCache (ICML 2026, "Test-Time Self-Calibration Caching for Video
Generation") for DiffSynth-Studio's ``WanModel``.

NaviCache is a *whole-forward* residual cache. Instead of caching attention KV
states or selecting a subset of tokens (as the RAS machinery in
``wan_video_dit.py`` does), it caches the entire DiT output residual
``residual = output - input`` per denoising step and, on a low-drift step,
skips the whole transformer forward and reconstructs the result as
``input + cached_residual``.

The skip/compute decision is gated by a scalar Kalman filter that tracks the
ratio ``Δoutput / Δinput`` between consecutive steps. This is calibrated
online (no offline profiling, no fixed threshold on the residual norm), so it
adapts to the current sample's trajectory.

Classifier-free guidance calls the DiT twice per step (conditional, then
unconditional). NaviCache decides on the **conditional (positive) branch only**
and reuses that decision for the paired unconditional (negative) branch, exactly
like the RAS scripts restrict region selection to the positive branch.

Reference implementation:
``../NaviCache/NaviCache4Wan2.1/navicache_generate.py`` (``navicache_forward``).
"""

from typing import Optional

import torch


def _mean_abs(t: torch.Tensor) -> float:
    """Mean absolute value over all elements of a tensor (scalar, as a float)."""
    return float(t.flatten().abs().mean().item())


class NaviCache:
    """Stateful wrapper that caches the DiT output residual and skips whole
    forwards when the Kalman-predicted output drift stays under ``thresh``.

    Parameters mirror the NaviCache reference:
    - ``thresh``: accumulated predicted-error threshold (hit/miss boundary).
      Wan2.1 fast/mid/slow ≈ 0.07/0.05/0.04.
    - ``align_steps``: initial exact-compute denoising steps (calibration warm-up).
    - ``num_inference_steps``: total denoising steps (for counter wrap/cutoff).
    - ``cfg``: whether the sampler runs CFG (two forwards per step). When False,
      every forward is treated as the conditional branch.
    - ``process_noise`` / ``measurement_noise``: Kalman Q and R (default 0.05/0.05).
    """

    def __init__(
        self,
        model,
        thresh: float = 0.05,
        align_steps: int = 10,
        num_inference_steps: int = 15,
        cfg: bool = True,
        process_noise: float = 0.05,
        measurement_noise: float = 0.05,
    ):
        self.model = model
        self.thresh = thresh
        self.align_steps = align_steps
        self.cfg = cfg
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Forwards (not steps) per generation: CFG calls the DiT twice per step.
        self.num_forwards = num_inference_steps * (2 if cfg else 1)
        self.align_forwards = align_steps * (2 if cfg else 1)
        # Force the final pair (CFG) / final forward (no CFG) to compute exactly,
        # so the cache is refreshed before the output is committed.
        self.cutoff_forwards = self.num_forwards - (2 if cfg else 1)

        self.reset()

    def reset(self):
        """Re-initialize all per-generation state (safe to reuse one wrapper)."""
        self.forward_count = 0

        # Kalman filter state (scalar floats).
        self.state_ratio: Optional[float] = None
        self.uncertainty: Optional[float] = None
        self.prediction_ratio: Optional[float] = None
        self.accumulated_error = 0.0
        self.should_compute = True

        # Cached tensors (created lazily on the compute path).
        self.previous_raw_input: Optional[torch.Tensor] = None   # cond input, current step
        self.prior_raw_input: Optional[torch.Tensor] = None      # cond input, previous step
        self.previous_output: Optional[torch.Tensor] = None      # cond output, previous compute
        self.cond_residual: Optional[torch.Tensor] = None        # output - input (cond)
        self.uncond_residual: Optional[torch.Tensor] = None      # output - input (uncond)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run one NaviCache-gated DiT forward.

        Returns the unpatchified latent prediction ``[B, C_out, F, H, W]``, the
        same as ``self.model(x, ...)`` with ``return_noise_tokens=False``. On a
        skipped step the transformer is not run at all; the cached residual from
        the previous exact pass is reused instead.
        """
        raw_input = x.clone()

        # CFG runs conditional then unconditional forwards as pairs. The decision
        # (and the Kalman calibration) is made on the CONDITIONAL branch only;
        # the unconditional branch reuses ``self.should_compute`` from the pair.
        is_cond = (self.forward_count % 2 == 0) if self.cfg else True

        if is_cond:
            # Initial alignment steps and the final pair are always computed
            # exactly, so the cache is calibrated before reuse and refreshed near
            # the output.
            if (
                self.forward_count < self.align_forwards
                or self.forward_count >= self.cutoff_forwards
            ):
                self.should_compute = True
                self.accumulated_error = 0.0
            elif (
                self.previous_raw_input is not None
                and self.previous_output is not None
                and self.prediction_ratio is not None
            ):
                # Predict the output drift from the observed latent drift and
                # accumulate it. Skip only while it stays under the threshold.
                raw_input_change = _mean_abs(raw_input - self.previous_raw_input)
                output_norm = _mean_abs(self.previous_output)
                pred_change = self.prediction_ratio * (raw_input_change / output_norm)
                self.accumulated_error += pred_change
                self.should_compute = self.accumulated_error >= self.thresh
                if self.should_compute:
                    self.accumulated_error = 0.0
            else:
                # Not yet calibrated (e.g. no previous exact pass): compute.
                self.should_compute = True

            # Record the current input for the next step's drift estimate.
            self.previous_raw_input = raw_input.clone()

        # ---- Skip: reuse the cached residual for this CFG branch. ----
        if not self.should_compute:
            residual = self.cond_residual if is_cond else self.uncond_residual
            if residual is not None:
                self._advance()
                return raw_input + residual

        # ---- Compute: run the real transformer forward. ----
        output = self.model(
            x,
            timestep,
            context,
            clip_feature=clip_feature,
            y=y,
            **kwargs,
        )

        if is_cond:
            # Exact conditional passes provide the measurements for the online
            # self-calibration (Kalman) update.
            if self.previous_output is not None:
                output_change = _mean_abs(output - self.previous_output)
                if self.prior_raw_input is not None:
                    input_change = _mean_abs(
                        self.previous_raw_input - self.prior_raw_input
                    )
                    z = output_change / (input_change + 1e-8)
                    is_warmup = self.forward_count < self.align_forwards
                    if self.state_ratio is None or is_warmup:
                        self.state_ratio = z
                        self.uncertainty = 1.0
                    else:
                        self.uncertainty = self.uncertainty + self.process_noise
                        fusion = self.uncertainty / (
                            self.uncertainty + self.measurement_noise + 1e-8
                        )
                        self.state_ratio = self.state_ratio + fusion * (z - self.state_ratio)
                        self.uncertainty = (1.0 - fusion) * self.uncertainty
                    self.prediction_ratio = self.state_ratio

            # Shift the two-step input delay line and refresh the output/residual.
            self.prior_raw_input = self.previous_raw_input
            self.previous_output = output.clone()
            self.cond_residual = output - raw_input
        else:
            self.uncond_residual = output - raw_input

        self._advance()
        return output

    def _advance(self):
        self.forward_count += 1
        if self.forward_count >= self.num_forwards:
            self.forward_count = 0
