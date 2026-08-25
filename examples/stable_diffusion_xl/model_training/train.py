import torch, os, argparse, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline, ModelConfig, SDXLUnit_PromptEmbedder, SDXLUnit_AddTimeIdsComputer
from diffsynth.diffusion import *
from diffsynth.utils.lora.sdxl import SdxlLoRAConverter
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class StableDiffusionXLTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, tokenizer_2_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_alpha=None, lora_checkpoint=None,
        dmd_generator_lora_checkpoint=None,
        dmd_fake_score_lora_checkpoint=None,
        dmd_real_score_lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        quant_options=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        prompt_only_height=1024, prompt_only_width=1024,
        dmd_cfg_scale=1.0,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, quant_options=quant_options, device=device)
        tokenizer_config = self.parse_path_or_model_id(tokenizer_path, ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer/"))
        tokenizer_2_config = self.parse_path_or_model_id(tokenizer_2_path, ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer_2/"))
        self.pipe = StableDiffusionXLPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, tokenizer_2_config=tokenizer_2_config)
        self.prompt_only_height = prompt_only_height or 1024
        self.prompt_only_width = prompt_only_width or 1024
        self.dmd_cfg_scale = dmd_cfg_scale
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)

        # Training mode
        if task in ("dmd", "dmd:train"):
            self.setup_dmd_model_roles(
                self.pipe,
                lora_target_modules=lora_target_modules,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                generator_lora_checkpoint=dmd_generator_lora_checkpoint,
                fake_score_lora_checkpoint=dmd_fake_score_lora_checkpoint,
                real_score_lora_checkpoint=dmd_real_score_lora_checkpoint,
                base_model_name=lora_base_model,
            )
        else:
            self.switch_pipe_to_training_mode(
                self.pipe, trainable_models,
                lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
                preset_lora_path, preset_lora_model,
                lora_alpha=lora_alpha,
                task=task,
            )
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            # DMD intentionally reuses the SFT dataset/cache schema.
            # Prefer running sft:data_process first; this alias is equivalent.
            "dmd:data_process": lambda pipe, *args: args,
            "dmd:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega, dmd_update, tau, dmd_step: StartDMDLossDDIM(
                pipe, inputs_shared, inputs_posi, inputs_nega, dmd_update=dmd_update, tau=tau, step_num=dmd_step
            ),
            "meanflow:data_process": lambda pipe, *args: args,
        }
        
    def get_pipeline_inputs(self, data):
        if "prompt" not in data:
            raise ValueError("SDXL training data must include a `prompt` field.")
        image = data.get("image", None)
        has_image = image is not None and not (isinstance(image, list) and all(item is None for item in image))
        if not has_image and self.task not in ("dmd", "dmd:train"):
            raise ValueError("Prompt-only SDXL training is only supported for `dmd` and `dmd:train`.")

        if not has_image:
            return self.get_prompt_only_dmd_inputs(data)

        if isinstance(image, list):
            if len(image) != 1:
                raise ValueError("Image-backed SDXL DMD batches are not supported. Use --dmd_batch_size 1 or a prompt-only dataset.")
            image = image[0]

        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"prompt": data.get("negative_prompt", "")}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": image,
            "height": image.size[1],
            "width": image.size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": self.dmd_cfg_scale,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def get_prompt_only_dmd_inputs(self, data):
        prompt = data["prompt"]
        if not isinstance(prompt, (str, list, tuple)):
            raise ValueError("Prompt-only SDXL DMD data must provide `prompt` as a string or a list of strings.")
        if isinstance(prompt, tuple):
            prompt = list(prompt)
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        height, width = self.pipe.check_resize_height_width(self.prompt_only_height, self.prompt_only_width, verbose=0)

        prompt_embedder = SDXLUnit_PromptEmbedder()
        add_time_ids_computer = SDXLUnit_AddTimeIdsComputer()
        inputs_shared = {
            "input_latents": torch.zeros(
                (batch_size, self.pipe.unet.in_channels, height // 8, width // 8),
                dtype=self.pipe.torch_dtype,
                device=self.pipe.device,
            ),
            "height": height,
            "width": width,
            "cfg_scale": self.dmd_cfg_scale,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "_skip_pipeline_units": True,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)

        negative_prompt = data.get("negative_prompt", "")
        if isinstance(prompt, list) and isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        elif isinstance(prompt, list) and isinstance(negative_prompt, tuple):
            negative_prompt = list(negative_prompt)
        if isinstance(negative_prompt, list) and len(negative_prompt) == 1 and batch_size > 1:
            negative_prompt = negative_prompt * batch_size
        if isinstance(negative_prompt, list) and len(negative_prompt) != batch_size:
            raise ValueError("Prompt-only SDXL DMD `negative_prompt` must be a string or match the prompt batch size.")

        with torch.no_grad():
            inputs_posi = prompt_embedder.process(self.pipe, prompt)
            inputs_nega = prompt_embedder.process(self.pipe, negative_prompt) if inputs_shared["cfg_scale"] != 1 else {}
            inputs_shared.update(add_time_ids_computer.process(self.pipe, height, width))
        if inputs_shared["add_time_ids"].shape[0] == 1 and batch_size > 1:
            inputs_shared["add_time_ids"] = inputs_shared["add_time_ids"].repeat_interleave(batch_size, dim=0)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None, generator_loss=0, fake_loss=0, dmd_update=None, tau=None, dmd_step=4):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        skip_pipeline_units = inputs[0].pop("_skip_pipeline_units", False)
        if not skip_pipeline_units:
            for unit in self.pipe.units:
                inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        if self.task in ("dmd", "dmd:train"):
            if dmd_update is None:
                dmd_update = "fake_score" if fake_loss else "generator" if generator_loss else None
            self.set_dmd_update_mode(dmd_update)
            loss = self.task_to_loss[self.task](self.pipe, *inputs, dmd_update, tau, dmd_step)
        else:
            loss = self.task_to_loss[self.task](self.pipe, *inputs)

        return loss


def parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--tokenizer_2_path", type=str, default=None, help="Path to tokenizer 2.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format.")
    return parser


if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=32,
            width_division_factor=32,
        )
    )
    model = StableDiffusionXLTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        tokenizer_2_path=args.tokenizer_2_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_checkpoint=args.lora_checkpoint,
        dmd_generator_lora_checkpoint=args.dmd_generator_lora_checkpoint,
        dmd_fake_score_lora_checkpoint=args.dmd_fake_score_lora_checkpoint,
        dmd_real_score_lora_checkpoint=args.dmd_real_score_lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        quant_options=args.quant_options,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        prompt_only_height=args.height or 1024,
        prompt_only_width=args.width or 1024,
        dmd_cfg_scale=args.dmd_cfg_scale,
        task=args.task,
        device="cpu" if args.enable_model_cpu_offload else accelerator.device,
    )
    logger_cls = DMDModelLogger if args.task in ("dmd", "dmd:train") else ModelLogger
    model_logger = logger_cls(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        state_dict_converter=SdxlLoRAConverter.align_to_opensource_format if args.align_to_opensource_format else lambda x:x,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        enable_csv_log=args.enable_csv_log,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        # DMD data processing is identical to SFT data processing.
        "dmd:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
        "dmd:train": launch_dmd_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
