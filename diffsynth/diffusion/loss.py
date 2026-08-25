from .base_pipeline import BasePipeline
import torch


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    if "lora" in inputs:
        # Image-to-LoRA models need to load lora here.
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"]) * inputs.get("noise_scale", 1.0)
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def FlowMatchSFTAudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # audio
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler.add_noise(inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = pipe.scheduler.training_target(inputs["audio_input_latents"], audio_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    if inputs.get("audio_input_latents") is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio
    return loss


def FlowMatchSFTMiniMaxH3AudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep_video = pipe.scheduler.timesteps[timestep_id].to(dtype=torch.float32, device=pipe.device)
    timestep_audio = pipe.scheduler_audio.timesteps[timestep_id].to(dtype=torch.float32, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep_video)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep_video)

    if "audio_input_latents" in inputs:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler_audio.add_noise(inputs["audio_input_latents"], audio_noise, timestep_audio)
        training_target_audio = pipe.scheduler_audio.training_target(inputs["audio_input_latents"], audio_noise, timestep_audio)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(
        **models, **inputs,
        timestep_video=timestep_video, timestep_audio=timestep_audio,
    )

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep_video)
    if "audio_input_latents" in inputs:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler_audio.training_weight(timestep_audio)
        loss = loss + loss_audio
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


def StartDMDLoss(pipe: BasePipeline, **inputs):
    # Current implementation assume tau = 0
    if "input_latents" not in inputs or inputs["input_latents"] is None:
        raise ValueError("StartDMDLoss requires SFT-prepared `input_latents`.")

    step_num = inputs["step_num"]

    tau = 0.0 if inputs.get("tau", None) is None else float(inputs["tau"])
    if tau != 0.0:
        raise ValueError("StartDMDLoss currently supports only tau=0.0.")

    dmd_update = inputs.get("dmd_update", None)
    if dmd_update is None:
        if inputs.get("fake_loss", 0):
            dmd_update = "fake_score"
        elif inputs.get("generator_loss", 0):
            dmd_update = "generator"
    if dmd_update not in ("fake_score", "generator"):
        raise ValueError("StartDMDLoss requires dmd_update='fake_score' or dmd_update='generator'.")

    # Scheduler sigmas are the noise levels: sigma=1 is pure noise, sigma=0 is clean.
    # They are ordered from high noise to low noise along the denoising trajectory.
    sigmas = pipe.scheduler.sigmas
    max_id_int = int(sigmas.shape[0]) - 1
    tau_sigma_cpu = sigmas.new_tensor(0.0)
    # Sample s from the full interval [1, 0].
    s_id_int = int(torch.randint(0, sigmas.shape[0], (1,)).item())

    tau_sigma = tau_sigma_cpu.to(dtype=pipe.torch_dtype, device=pipe.device)
    s_sigma = sigmas[s_id_int].to(dtype=pipe.torch_dtype, device=pipe.device)
    s_timestep = pipe.scheduler.timesteps[s_id_int:s_id_int + 1].to(dtype=pipe.torch_dtype, device=pipe.device)


    generator_inputs = inputs.copy()
    noise_scale = inputs.get("noise_scale", 1.0)
    eps_endpoint = torch.randn_like(inputs["input_latents"]) * noise_scale

    def consistency_sample():
        generator_noise = torch.randn_like(inputs["input_latents"]) * noise_scale
        generator_inputs["latents"] = generator_noise
        ids = torch.linspace(0, max_id_int, step_num).int()
        id_num = ids.shape[0] - 1
        for (i, t_id) in enumerate(ids):
            if i == id_num:
                cur_sigma = 0
            else:
                cur_sigma = sigmas[ids[i + 1]]

            assert cur_sigma <= noise_scale, "ValueError: Sigmas should be smaller than noise_scale"

            generator_noise = torch.randn_like(inputs["input_latents"])

            current_timestep = pipe.scheduler.timesteps[t_id:t_id+1].to(dtype=pipe.torch_dtype, device=pipe.device)
            x_g = pipe.model_fn(
                **pipe.get_iteration_models("generator"),
                **generator_inputs,
                timestep=current_timestep,
                progress_id=t_id,
            )
            generator_inputs["latents"] = x_g * (noise_scale - cur_sigma) + cur_sigma * generator_noise
        x_g = generator_inputs["latents"]
        return x_g

    pipe.load_role_models_to_device("generator")
    if dmd_update == "fake_score":
        with torch.no_grad():
            x_g = consistency_sample()
    else:
        x_g = consistency_sample()
    if dmd_update == "fake_score":
        x_g = x_g.detach()

    def noise_to_high_level(x_tau, noise):
        denom = torch.clamp(1.0 - tau_sigma, min=1e-6)
        x0_impl = (x_tau - tau_sigma * noise) / denom
        v_impl = noise - x0_impl
        x_s = x_tau + (s_sigma - tau_sigma) * v_impl
        return x_s, v_impl

    if dmd_update == "fake_score":
        fake_inputs = inputs.copy()
        x_s, v_impl = noise_to_high_level(x_g.detach(), eps_endpoint)
        fake_inputs["latents"] = x_s.detach()
        pipe.load_role_models_to_device("fake_score")
        v_pred = pipe.model_fn(
            **pipe.get_iteration_models("fake_score"),
            **fake_inputs,
            timestep=s_timestep,
            progress_id=s_id_int,
        )
        return torch.nn.functional.mse_loss(v_pred.float(), v_impl.detach().float())

    with torch.no_grad():
        x_s, _ = noise_to_high_level(x_g.detach(), eps_endpoint)
        fake_inputs = inputs.copy()
        real_inputs = inputs.copy()
        fake_inputs["latents"] = x_s
        real_inputs["latents"] = x_s
        pipe.load_role_models_to_device("real_score")
        v_real = pipe.model_fn(
            **pipe.get_iteration_models("real_score"),
            **real_inputs,
            timestep=s_timestep,
            progress_id=s_id_int,
        )
        pipe.load_role_models_to_device("fake_score")
        v_fake = pipe.model_fn(
            **pipe.get_iteration_models("fake_score"),
            **fake_inputs,
            timestep=s_timestep,
            progress_id=s_id_int,
        )
        ds = s_sigma - tau_sigma
        pred_real = x_s - ds * v_real
        pred_fake = x_s - ds * v_fake
        p_real = x_g.detach() - pred_real
        p_fake = x_g.detach() - pred_fake
        reduce_dims = tuple(range(1, p_real.ndim))
        weight = torch.abs(p_real).mean(dim=reduce_dims, keepdim=True).detach()
        grad = (p_real - p_fake) / (weight + 1e-8)
        grad = torch.nan_to_num(grad)
        target = (x_g - grad).detach()
    return 0.5 * torch.nn.functional.mse_loss(x_g.float(), target.float())


def StartDMDLossDDIM(pipe: BasePipeline, inputs_shared=None, inputs_posi=None, inputs_nega=None, **inputs):
    if inputs_shared is None:
        model_inputs_shared = dict(inputs)
        loss_inputs = model_inputs_shared
    else:
        model_inputs_shared = dict(inputs_shared)
        loss_inputs = dict(model_inputs_shared)
        loss_inputs.update(inputs)
    inputs_posi = {} if inputs_posi is None else dict(inputs_posi)
    inputs_nega = {} if inputs_nega is None else dict(inputs_nega)

    # Note, "" instead of zero embedding is passed in as the negative branch since SDXL is directly trained on it.

    if "input_latents" not in loss_inputs or loss_inputs["input_latents"] is None:
        raise ValueError("StartDMDLossDDIM requires SFT-prepared `input_latents`.")

    if not hasattr(pipe.scheduler, "alphas_cumprod"):
        raise ValueError("StartDMDLossDDIM requires a DDIM-style scheduler with `alphas_cumprod`.")

    step_num = int(loss_inputs["step_num"])
    if step_num <= 0:
        raise ValueError("StartDMDLossDDIM requires `step_num` to be positive.")

    tau = 0.0 if loss_inputs.get("tau", None) is None else float(loss_inputs["tau"])
    if tau != 0.0:
        raise ValueError("StartDMDLossDDIM currently supports only tau=0.0.")

    dmd_update = loss_inputs.get("dmd_update", None)
    if dmd_update is None:
        if loss_inputs.get("fake_loss", 0):
            dmd_update = "fake_score"
        elif loss_inputs.get("generator_loss", 0):
            dmd_update = "generator"
    if dmd_update not in ("fake_score", "generator"):
        raise ValueError("StartDMDLossDDIM requires dmd_update='fake_score' or dmd_update='generator'.")

    timesteps = pipe.scheduler.timesteps
    max_id_int = int(timesteps.shape[0]) - 1
    num_train_timesteps = len(pipe.scheduler.alphas_cumprod)
    min_step_percent = float(loss_inputs.get("dmd_min_step_percent", 0.02))
    max_step_percent = float(loss_inputs.get("dmd_max_step_percent", 0.98))
    if not 0.0 <= min_step_percent < max_step_percent <= 1.0:
        raise ValueError("DMD timestep bounds require 0 <= dmd_min_step_percent < dmd_max_step_percent <= 1.")
    min_timestep = max(0, min(int(min_step_percent * num_train_timesteps), num_train_timesteps - 1))
    max_timestep = max(0, min(int(max_step_percent * num_train_timesteps), num_train_timesteps - 1))
    if min_timestep >= max_timestep:
        raise ValueError("DMD timestep bounds collapsed to an empty range.")
    s_timestep_int = int(torch.randint(min_timestep, max_timestep + 1, (1,), device=pipe.device).item())
    s_id_int = s_timestep_int
    s_timestep = torch.tensor([s_timestep_int], dtype=torch.float32, device=pipe.device)
    noise_scale = loss_inputs.get("noise_scale", 1.0)

    cfg_scale = loss_inputs.get("cfg_scale", 1.0)

    def alpha_sigma(timestep):
        timestep_int = int(round(timestep.flatten()[0].detach().float().cpu().item()))
        timestep_int = max(0, min(timestep_int, len(pipe.scheduler.alphas_cumprod) - 1))
        alpha_prod = torch.tensor(
            pipe.scheduler.alphas_cumprod[timestep_int],
            dtype=torch.float32,
            device=pipe.device,
        )
        alpha = torch.sqrt(torch.clamp(alpha_prod, min=1e-12))
        sigma = torch.sqrt(torch.clamp(1.0 - alpha_prod, min=0.0))
        return alpha, sigma

    def predict_model_output(role, role_inputs, timestep, progress_id, cfg_scale):
        role_attrs = getattr(pipe, "dmd_model_role_attrs", {})
        lora_module = getattr(pipe, role_attrs[role], None) if role in role_attrs else None
        return pipe.cfg_guided_model_fn(
            pipe.model_fn,
            cfg_scale,
            role_inputs,
            inputs_posi,
            inputs_nega,
            lora_module=lora_module,
            **pipe.get_iteration_models(role),
            timestep=timestep,
            progress_id=progress_id,
        )

    def model_output_to_eps_x0_score(model_output, latents, timestep):
        alpha, sigma = alpha_sigma(timestep)
        alpha_safe = torch.clamp(alpha, min=1e-6)
        sigma_safe = torch.clamp(sigma, min=1e-6)
        prediction_type = getattr(pipe.scheduler, "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            eps_pred = model_output
            x0_pred = (latents - sigma * eps_pred) / alpha_safe
        elif prediction_type == "v_prediction":
            x0_pred = alpha * latents - sigma * model_output
            eps_pred = sigma * latents + alpha * model_output
        else:
            raise NotImplementedError(f"{prediction_type} is not implemented for StartDMDLossDDIM.")
        score = -eps_pred / sigma_safe
        return eps_pred, x0_pred, score

    def predict_eps_x0_score(role, role_inputs, timestep, progress_id, cfg_scale):
        model_output = predict_model_output(role, role_inputs, timestep, progress_id, cfg_scale)
        return model_output_to_eps_x0_score(model_output, role_inputs["latents"], timestep)

    def training_target(clean_latents, noise, alpha, sigma):
        prediction_type = getattr(pipe.scheduler, "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            return noise
        elif prediction_type == "v_prediction":
            return alpha * noise - sigma * clean_latents
        else:
            raise NotImplementedError(f"{prediction_type} is not implemented for StartDMDLossDDIM.")

    def consistency_sample():
        generator_inputs = model_inputs_shared.copy()
        generator_inputs["latents"] = torch.randn_like(loss_inputs["input_latents"]) * noise_scale
        ids = torch.linspace(0, max_id_int, step_num + 1).int()
        id_num = ids.shape[0] - 2
        stop_turn = torch.randint(0, id_num + 1, (1,)).item()
        for i, t_id in enumerate(ids):
            t_id_int = int(t_id.item())
            current_timestep = timesteps[t_id_int:t_id_int + 1].to(dtype=torch.float32, device=pipe.device)
            _, x_g, _ = predict_eps_x0_score("generator", generator_inputs, current_timestep, t_id_int, 1.0)
            if i == stop_turn:
                generator_inputs["latents"] = x_g
                return x_g
            else:
                next_id_int = int(ids[i + 1].item())
                next_timestep = timesteps[next_id_int:next_id_int + 1].to(dtype=torch.float32, device=pipe.device)
                next_alpha, next_sigma = alpha_sigma(next_timestep)
                generator_noise = torch.randn_like(loss_inputs["input_latents"]) * noise_scale
                generator_inputs["latents"] = next_alpha * x_g + next_sigma * generator_noise
                generator_inputs["latents"] = generator_inputs["latents"].detach()
        return generator_inputs["latents"]

    pipe.load_role_models_to_device("generator")
    if dmd_update == "fake_score":
        with torch.no_grad():
            x_g = consistency_sample()
        x_g = x_g.detach()
    else:
        x_g = consistency_sample()

    s_alpha, s_sigma = alpha_sigma(s_timestep)
    eps_endpoint = torch.randn_like(loss_inputs["input_latents"]) * noise_scale

    if dmd_update == "fake_score":
        fake_inputs = model_inputs_shared.copy()
        fake_inputs["latents"] = (s_alpha * x_g.detach() + s_sigma * eps_endpoint).detach()
        pipe.load_role_models_to_device("fake_score")
        model_output = predict_model_output("fake_score", fake_inputs, s_timestep, s_id_int, 1.0)
        target = training_target(x_g.detach(), eps_endpoint, s_alpha, s_sigma)
        return torch.nn.functional.mse_loss(model_output.float(), target.detach().float())

    with torch.no_grad():
        x_s = s_alpha * x_g.detach() + s_sigma * eps_endpoint
        fake_inputs = model_inputs_shared.copy()
        real_inputs = model_inputs_shared.copy()
        fake_inputs["latents"] = x_s
        real_inputs["latents"] = x_s
        pipe.load_role_models_to_device("real_score")
        _, x_real, real_score = predict_eps_x0_score("real_score", real_inputs, s_timestep, s_id_int, cfg_scale)
        pipe.load_role_models_to_device("fake_score")
        _, x_fake, fake_score = predict_eps_x0_score("fake_score", fake_inputs, s_timestep, s_id_int, 1.0)
        reduce_dims = tuple(range(1, x_real.ndim))
        weight = torch.abs(x_real - x_g).mean(dim=reduce_dims, keepdim=True).detach()
        grad = (x_fake - x_real) / (weight + 1e-8)
        grad = torch.nan_to_num(grad)
        target = (x_g - grad).detach()
    return 0.5 * torch.nn.functional.mse_loss(x_g.float(), target.float())


# def ConsistencyLoss(pipe: BasePipeline, **inputs):

#     assert type(getattr(inputs, "c_skip", 0)) != int, "c_skip should be initialized when training consistency model"
#     assert type(getattr(inputs, "c_out", 0)) != int, "c_out should be initialized when training consistency model"

#     c_skip = inputs.get("c_skip")
#     c_out = inputs.get("c_out")

#     if "lora" in inputs:
#         # Image-to-LoRA models need to load lora here.
#         pipe.clear_lora(verbose=0)
#         pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

#     max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
#     min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

#     timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
#     timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

#     noise = torch.randn_like(inputs["input_latents"]) * inputs.get("noise_scale", 1.0)
#     inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
#     x_pre = inputs["latents"]
#     teacher_model = {"teacher" : getattr(pipe, "teacher")}
#     teacher_pred = pipe.model_fn(**teacher_model, **inputs, timestep=timestep)
#     x_post = pipe.step(pipe.scheduler, progress_id=timestep_id, noise_pred=teacher_pred, **inputs)

#     x_pre = x_pre.detach()
#     x_post = x_post.detach()

#     if "first_frame_latents" in inputs:
#         raise NotImplementedError("Have not implemented first frame editing module")

#     student_model = {"student" : getattr(pipe, "student")}
#     student_pred = c_out(timestep_id) * pipe.model_fn(**student_model, **inputs, timestep=timestep) + c_skip(timestep_id) * x_pre
#     student_pred_post = c_out(timestep_id) * pipe.model_fn(**student_model, **inputs, timestep=timestep) + c_skip(timestep_id) * x_post

#     # if "first_frame_latents" in inputs:
#     #     noise_pred = noise_pred[:, :, 1:]
#     #     training_target = training_target[:, :, 1:]

#     loss = torch.nn.functional.mse_loss(student_pred, student_pred_post)
#     loss = loss * pipe.scheduler.training_weight(timestep)
#     return loss



class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            denom = sigma_ - sigma
            denom = torch.sign(denom) * torch.clamp(denom.abs(), min=1e-6)
            target = (latents_ - inputs_shared["latents"]) / denom
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
