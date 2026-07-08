#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../../.."

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-./models}"

accelerate launch examples/stable_diffusion_xl/model_training/train.py \
  --dataset_base_path "data/laion_got" \
  --dataset_metadata_path "data/laion_got/prompts.jsonl" \
  --data_file_keys "" \
  --height 768 \
  --width 768 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "stabilityai/stable-diffusion-xl-base-1.0:text_encoder/model.safetensors,stabilityai/stable-diffusion-xl-base-1.0:text_encoder_2/model.safetensors,stabilityai/stable-diffusion-xl-base-1.0:unet/diffusion_pytorch_model.safetensors" \
  --learning_rate 5e-7 \
  --num_epochs 1 \
  --save_steps 1000 \
  --remove_prefix_in_ckpt "pipe.unet." \
  --output_path "./models/train/stable-diffusion-xl-base-1.0_dmd_laion_prompts" \
  --lora_base_model "unet" \
  --lora_target_modules "" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --dataset_num_workers 4 \
  --dmd_batch_size 2 \
  --dmd_tau 0.0 \
  --dmd_step 4 \
  --dmd_dfake_gen_ratio 5 \
  --task "dmd:train" \
  --enable_wandb_log
