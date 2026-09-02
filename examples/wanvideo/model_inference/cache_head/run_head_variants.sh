#!/usr/bin/env bash
# Runs the supervised cache-head training arm once per head variant, sequentially
# (each run claims the whole 8-GPU node via --nproc_per_node=8).
set -euo pipefail

HEAD_VARIANTS=(latent_fusion latent_residual latent_residual_deep)

for variant in "${HEAD_VARIANTS[@]}"; do
    echo "=== Starting head-variant: ${variant} ==="
    torchrun --standalone --nproc_per_node=8 \
        examples/wanvideo/model_inference/cache_head/cache_head_model_training.py \
        --arm supervised \
        --head-variant "${variant}" \
        --lr 5e-4 \
        --captions examples/wanvideo/model_inference/cache_head/mixkit_captions.jsonl \
        --epochs 40 \
        --batch-size 4 \
        --micro-batch 4 \
        --val-subset 128 \
        --val-batches 1 \
        --heatmap-every 0 \
        --precision bf16 \
        --save-dir "runs/supervised/dual-dense-tf-${variant}" \
        --trajectory-dir /data2/weixinyuan \
        --optimizer-steps-per-iteration 10 \
        --log-interval 1 \
        --full-steps 1,2,3,4,5,6,7 \
        --seed 0 \
        --wandb-project cache-head-supervised \
        --wandb-run-name "supervised-tf-${variant}"
    echo "=== Finished head-variant: ${variant} ==="
done
