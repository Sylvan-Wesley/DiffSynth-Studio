import torch
from tqdm import tqdm
from typing import Union

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion.base_pipeline import BasePipeline


class StableDiffusion35Pipeline(BasePipeline):
    """Stable Diffusion 3.5 inference pipeline.

    This first SD 3.5 integration delegates model execution to Hugging Face
    Diffusers' StableDiffusion3Pipeline. SD 3.5 uses an MMDiT transformer and
    flow matching scheduler, so it is intentionally separate from the legacy
    SD1.5/SDXL UNet pipelines in this repository.
    """

    def __init__(self, pipe, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
        )
        self.pipe = pipe

    @staticmethod
    def _load_diffusers_pipeline():
        try:
            from diffusers import StableDiffusion3Pipeline
        except ImportError as error:
            raise ImportError(
                "StableDiffusion35Pipeline requires diffusers. "
                "Install DiffSynth-Studio again or run `pip install diffusers accelerate`."
            ) from error
        return StableDiffusion3Pipeline

    @staticmethod
    def _resolve_pretrained_path(
        model_id: str,
        model_config: ModelConfig = None,
        local_model_path: str = None,
        download_source: str = "huggingface",
    ):
        if model_config is not None:
            model_config.download_if_necessary()
            return model_config.path, {}

        load_kwargs = {}
        if local_model_path is not None:
            load_kwargs["cache_dir"] = local_model_path

        if download_source.lower() == "modelscope":
            model_config = ModelConfig(
                model_id=model_id,
                origin_file_pattern="",
                local_model_path=local_model_path,
                download_source="modelscope",
            )
            model_config.download_if_necessary()
            return model_config.path, {}
        elif download_source.lower() == "huggingface":
            return model_id, load_kwargs
        else:
            raise ValueError("`download_source` should be `huggingface` or `modelscope`.")

    @staticmethod
    def from_pretrained(
        model_id: str = "stabilityai/stable-diffusion-3.5-large",
        model_config: ModelConfig = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        local_model_path: str = None,
        download_source: str = "huggingface",
        enable_model_cpu_offload: bool = False,
        enable_sequential_cpu_offload: bool = False,
        use_t5_encoder: bool = True,
        **kwargs,
    ):
        """Load an SD 3.5 model.

        Args:
            model_id: Hugging Face or ModelScope model ID.
            model_config: Optional DiffSynth ModelConfig. When supplied, this
                config controls download/local path resolution.
            torch_dtype: Storage and compute dtype passed to Diffusers.
            device: Runtime device for non-offloaded inference.
            local_model_path: Optional cache directory for downloaded models.
            download_source: `huggingface` for Diffusers Hub loading or
                `modelscope` for a pre-download through ModelConfig.
            enable_model_cpu_offload: Delegate to Diffusers model CPU offload.
            enable_sequential_cpu_offload: Delegate to Diffusers sequential CPU
                offload. This saves more VRAM and is slower.
            use_t5_encoder: Set to False to skip the third T5 text encoder for
                lower VRAM usage at reduced prompt fidelity.
            **kwargs: Extra arguments forwarded to
                StableDiffusion3Pipeline.from_pretrained.
        """
        diffusers_pipeline_cls = StableDiffusion35Pipeline._load_diffusers_pipeline()
        pretrained_path, load_kwargs = StableDiffusion35Pipeline._resolve_pretrained_path(
            model_id=model_id,
            model_config=model_config,
            local_model_path=local_model_path,
            download_source=download_source,
        )

        load_kwargs.update(kwargs)
        load_kwargs["torch_dtype"] = torch_dtype
        if not use_t5_encoder:
            load_kwargs.setdefault("text_encoder_3", None)
            load_kwargs.setdefault("tokenizer_3", None)

        pipe = diffusers_pipeline_cls.from_pretrained(pretrained_path, **load_kwargs)
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        elif enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to(device)

        return StableDiffusion35Pipeline(pipe, device=device, torch_dtype=torch_dtype)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if hasattr(self, "pipe") and self.pipe is not None:
            self.pipe = self.pipe.to(self.device, dtype=self.torch_dtype)
        return self

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, list[str]],
        negative_prompt: Union[str, list[str]] = "",
        prompt_2: Union[str, list[str]] = None,
        prompt_3: Union[str, list[str]] = None,
        negative_prompt_2: Union[str, list[str]] = None,
        negative_prompt_3: Union[str, list[str]] = None,
        cfg_scale: float = 3.5,
        height: int = 1024,
        width: int = 1024,
        seed: int = None,
        rand_device: str = "cpu",
        num_inference_steps: int = 28,
        t5_sequence_length: int = 512,
        progress_bar_cmd=tqdm,
        **kwargs,
    ):
        height, width = self.check_resize_height_width(height, width)
        generator = None
        if seed is not None:
            generator = torch.Generator(rand_device).manual_seed(seed)

        if progress_bar_cmd is None:
            self.pipe.set_progress_bar_config(disable=True)
        else:
            self.pipe.set_progress_bar_config(disable=False)

        output = self.pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            guidance_scale=cfg_scale,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            max_sequence_length=t5_sequence_length,
            **kwargs,
        )
        images = output.images
        if isinstance(images, list) and len(images) == 1:
            return images[0]
        return images
