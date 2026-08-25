"""Velocity/residual analysis using a warped Wan flow-matching schedule.

This is intentionally separate from ``analyze_velocity_residual.py`` so the
baseline schedule and its artifacts remain available for comparison.  It uses
the same model, prompt, seed, CFG, and analysis metrics, but replaces Wan's
usual ``shift=5`` sigma schedule with the ``t_compute`` schedule below.

Usage:
    python analyze_velocity_residual_warped.py
"""

import torch

out_npz = "velocity_analysis_warped.npz"
out_png = "velocity_analysis_warped.png"
out_video = "velocity_analysis_warped.mp4"

warp_s = 1e-4
warp_rho = 2.25


def t_compute(t, step_num, s=warp_s, rho=warp_rho):
    """Return the increasing warped time coordinate for a step index tensor.

    ``t`` is expected to contain indices in ``[0, step_num)``.  The small
    ``s`` term is retained from the supplied method to avoid a NaN at zero.
    """
    if step_num <= 0:
        raise ValueError(f"step_num must be positive, got {step_num}.")
    t = torch.as_tensor(t)
    # At early steps the two quantities in the outer subtraction are almost
    # equal.  Evaluate the supplied expression in float64 so those steps keep
    # distinct times, then return the standard scheduler precision.
    result_dtype = torch.float64 if t.dtype == torch.float64 else torch.float32
    t = t.to(torch.float64)
    warped_time = torch.sqrt(
        1 + s - torch.sqrt(
            (1 - (t / (step_num + 1)) ** (2 * rho) - s) / (1 - 2 * s)
        )
    )
    return warped_time.to(dtype=result_dtype)


def make_warped_schedule(step_num, s=warp_s, rho=warp_rho):
    """Build descending Wan sigmas and model timesteps for ``step_num`` steps.

    The supplied warp increases from data time toward noise time.  Wan
    denoising starts from noise, so its integration schedule is the complement
    ``sigma = 1 - t_compute(step_index, step_num)``.  Unlike the baseline, this
    intentionally does not apply Wan's additional rational ``shift=5`` warp.
    """
    if step_num <= 0:
        raise ValueError(f"step_num must be positive, got {step_num}.")
    step_indices = torch.arange(step_num, dtype=torch.float32)
    warped_times = t_compute(step_indices, step_num, s=s, rho=rho)
    sigmas = 1 - warped_times
    timesteps = sigmas * 1000
    return warped_times, sigmas, timesteps


def main():
    # Keep the model setup and analysis calculations identical to the baseline,
    # but load them only for an actual inference run.  This keeps the schedule
    # helpers above CPU-testable without DiffSynth's optional model dependencies.
    import numpy as np
    from tqdm import tqdm

    import analyze_velocity_residual as base

    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.models.wan_video_dit import set_to_torch_norm
    from diffsynth.utils.data import save_video

    model_id = base.model_id
    num_inference_steps = base.num_inference_steps
    cfg_scale = base.cfg_scale
    seed = base.seed
    num_frames = base.num_frames
    height = base.height
    width = base.width
    prompt = base.prompt
    negative_prompt = base.negative_prompt
    plot = base.plot
    sigma_marker = base.sigma_marker

    device = base._detect_device()
    print(f"Device: {device}")

    print("Loading model...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id=model_id, origin_file_pattern="google/umt5-xxl/"),
    )

    dit = pipe.dit
    scheduler = pipe.scheduler
    set_to_torch_norm([dit])
    print(f"  DiT blocks: {len(dit.blocks)}, dim: {dit.dim}, patch: {dit.patch_size}")

    print("Encoding prompts...")
    ctx_posi = base.encode_prompt(pipe, prompt)
    ctx_nega = base.encode_prompt(pipe, negative_prompt)

    print("Initializing latents...")
    z_dim = pipe.vae.model.z_dim
    latent_frames = (num_frames - 1) // 4 + 1
    latent_h = height // pipe.vae.upsampling_factor
    latent_w = width // pipe.vae.upsampling_factor
    shape = (1, z_dim, latent_frames, latent_h, latent_w)
    latents = pipe.generate_noise(shape, seed=seed, rand_device="cpu")
    latents = latents.to(dtype=pipe.torch_dtype, device=device)
    initial_latents = latents.detach().float().cpu().squeeze(0).numpy()

    # Override the scheduler state as a coherent pair: ``step`` uses sigmas for
    # Euler integration and the DiT receives the corresponding 0..1000 times.
    warped_times, sigmas, timesteps = make_warped_schedule(num_inference_steps)
    scheduler.sigmas = sigmas
    scheduler.timesteps = timesteps
    print(
        f"  Warped timesteps: {len(timesteps)} steps, "
        f"sigma range [{sigmas[0]:.3f}, {sigmas[-1]:.3f}], "
        f"s={warp_s:g}, rho={warp_rho:g}"
    )

    dit.eval()
    velocities = []

    print(f"\nDenoising ({num_inference_steps} warped steps, CFG {cfg_scale})...")
    with torch.inference_mode():
        for progress_id, timestep in enumerate(tqdm(scheduler.timesteps)):
            t = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=device)

            noise_posi = dit(x=latents, timestep=t, context=ctx_posi)
            noise_nega = dit(x=latents, timestep=t, context=ctx_nega)
            noise_pred = noise_nega + cfg_scale * (noise_posi - noise_nega)

            velocities.append(noise_pred.detach().float().cpu().squeeze(0).numpy())
            latents = scheduler.step(noise_pred, scheduler.timesteps[progress_id], latents)

    final_latents = latents.detach().float().cpu().squeeze(0).numpy()
    edm_curvature, cos_instant_straight = base.edm_straightness_metrics(
        velocities, initial_latents, final_latents, sigma_start=float(sigmas[0]),
    )

    n_pairs = len(velocities) - 1
    F = velocities[0].shape[1]
    C = velocities[0].shape[0]

    pair_sigmas = sigmas[1:].cpu().numpy()
    pair_timesteps = timesteps[1:].cpu().numpy()
    cos_pre_res = np.zeros(n_pairs)
    cos_pre_cur = np.zeros(n_pairs)
    pear_pre_res = np.zeros(n_pairs)
    rel_norm = np.zeros(n_pairs)
    f_par_1d = np.zeros(n_pairs)
    f_par_grow = np.zeros(n_pairs)
    ranks = np.zeros((n_pairs, F, 3), dtype=np.int64)
    stable = np.zeros((n_pairs, F, 3), dtype=np.float64)
    svals = np.zeros((n_pairs, F, 3, C), dtype=np.float64)

    span = base.GrowingSpan()
    span.add(velocities[0])
    for i in range(n_pairs):
        v_pre = velocities[i]
        v_cur = velocities[i + 1]
        residual = v_cur - v_pre

        cos_pre_res[i] = base.cosine(v_pre, residual)
        cos_pre_cur[i] = base.cosine(v_pre, v_cur)
        pear_pre_res[i] = base.pearson(v_pre, residual)
        rel_norm[i] = base.relative_norm(residual, v_pre)
        f_par_1d[i] = base.project_fraction_1d(residual, v_pre)
        f_par_grow[i] = span.project_fraction(residual)

        ranks[i, :, 0], stable[i, :, 0], svals[i, :, 0, :] = base.per_frame_rank(v_pre)
        ranks[i, :, 1], stable[i, :, 1], svals[i, :, 1, :] = base.per_frame_rank(residual)
        ranks[i, :, 2], stable[i, :, 2], svals[i, :, 2, :] = base.per_frame_rank(v_cur)
        span.add(v_cur)

    header = (
        f"{'step':>4} {'sigma':>7} {'t':>5} "
        f"{'cos(pre,r)':>10} {'pearson':>8} {'||r||/||pre||':>12} "
        f"{'rk_pre':>6} {'rk_res':>6} {'rk_cur':>6} "
        f"{'f_par_1d':>9} {'f_par_grow':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for i in range(n_pairs):
        print(
            f"{i:>4} {pair_sigmas[i]:>7.3f} {int(pair_timesteps[i]):>5} "
            f"{cos_pre_res[i]:>10.4f} {pear_pre_res[i]:>8.4f} {rel_norm[i]:>12.4f} "
            f"{ranks[i, :, 0].mean():>6.1f} {ranks[i, :, 1].mean():>6.1f} "
            f"{ranks[i, :, 2].mean():>6.1f} {f_par_1d[i]:>9.4f} "
            f"{f_par_grow[i]:>10.4f}"
        )

    print("\nEDM-style instantaneous velocity vs. overall straight-line displacement")
    print(f"{'step':>4} {'sigma':>7} {'t':>5} {'curvature':>12} {'cos(v,chord)':>13}")
    print("-" * 49)
    for i in range(len(velocities)):
        print(
            f"{i:>4} {sigmas[i]:>7.3f} {int(timesteps[i]):>5} "
            f"{edm_curvature[i]:>12.6g} {cos_instant_straight[i]:>13.4f}"
        )

    np.savez(
        out_npz,
        warped_times=warped_times.cpu().numpy(),
        sigmas=sigmas.cpu().numpy(),
        timesteps=timesteps.cpu().numpy(),
        pair_sigmas=pair_sigmas,
        pair_timesteps=pair_timesteps,
        cos_pre_res=cos_pre_res,
        cos_pre_cur=cos_pre_cur,
        pear_pre_res=pear_pre_res,
        rel_norm=rel_norm,
        f_par_1d=f_par_1d,
        f_par_grow=f_par_grow,
        edm_curvature=edm_curvature,
        cos_instant_straight=cos_instant_straight,
        ranks=ranks,
        stable_ranks=stable,
        singular_values=svals,
    )
    print(f"\nSaved analysis to: {out_npz}")

    if plot:
        base._make_plot(
            pair_sigmas, cos_pre_res, pear_pre_res, rel_norm,
            f_par_1d, f_par_grow, ranks, sigmas.cpu().numpy(), edm_curvature,
            cos_instant_straight, out_png, sigma_marker,
        )
        print(f"Saved plot to: {out_png}")

    # Decode the final warped-schedule latents with the normal Wan VAE and
    # persist the same generated sample that produced this analysis.
    print("Decoding generated video...")
    dit.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    video = pipe.vae.decode(
        latents,
        device=device,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    )
    video = pipe.vae_output_to_video(video)
    save_video(video, out_video, fps=15, quality=5)
    print(f"Saved generated video to: {out_video}")


if __name__ == "__main__":
    main()
