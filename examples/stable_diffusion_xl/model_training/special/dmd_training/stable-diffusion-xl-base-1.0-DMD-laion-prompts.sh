#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../../.."

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-./models}"

SDXL_DMD2_LORA_TARGET_MODULES="to_q,to_k,to_v,to_out.0,proj_in,proj_out,ff.net.0.proj,ff.net.2,conv1,conv2,conv_shortcut,downsamplers.0.conv,upsamplers.0.conv,time_emb_proj"

accelerate launch examples/stable_diffusion_xl/model_training/train.py \
  --dataset_base_path "data/laion_got" \
  --dataset_metadata_path "data/laion_got/prompts.jsonl" \
  --data_file_keys "" \
  --height 1024 \
  --width 1024 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "stabilityai/stable-diffusion-xl-base-1.0:text_encoder/model.safetensors,stabilityai/stable-diffusion-xl-base-1.0:text_encoder_2/model.safetensors,stabilityai/stable-diffusion-xl-base-1.0:unet/diffusion_pytorch_model.safetensors" \
  --learning_rate 5e-5 \
  --num_epochs 5 \
  --save_steps 10000 \
  --remove_prefix_in_ckpt "pipe.unet." \
  --output_path "./models/train/stable-diffusion-xl-base-1.0_dmd_laion_prompts" \
  --lora_base_model "unet" \
  --lora_target_modules "${SDXL_DMD2_LORA_TARGET_MODULES}" \
  --lora_rank 64 \
  --lora_alpha 8 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --dataset_num_workers 4 \
  --dmd_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --dmd_cfg_scale 8.0 \
  --dmd_tau 0.0 \
  --dmd_step 4 \
  --dmd_dfake_gen_ratio 5 \
  --task "dmd:train"
  # --enable_wandb_log
