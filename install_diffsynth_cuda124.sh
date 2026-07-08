#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-diffsynth-cu124}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if command -v conda >/dev/null 2>&1; then
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
  fi

  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
else
  echo "conda was not found. Create and activate a Python ${PYTHON_VERSION} environment first." >&2
  exit 1
fi

python -m pip install -U pip setuptools wheel

python -m pip uninstall -y \
  torch torchvision torchaudio triton \
  cuda-bindings cuda-pathfinder cuda-toolkit \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc \
  nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft \
  nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse \
  nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink \
  nvidia-nvshmem-cu13 nvidia-nvtx || true

python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0+cu124" \
  "torchvision==0.21.0+cu124" \
  "torchaudio==2.6.0+cu124"

python -m pip install \
  "diffsynth==2.0.12" \
  "diffusers==0.32.2" \
  "transformers==4.49.0" \
  "accelerate==1.4.0" \
  "peft==0.14.0" \
  "datasets==3.3.2" \
  "huggingface_hub==0.29.3" \
  "tokenizers==0.21.0" \
  "safetensors==0.5.3" \
  "numpy==1.26.4" \
  "pandas==2.2.3" \
  "pyarrow==19.0.1" \
  "pillow==11.1.0" \
  "imageio==2.37.0" \
  "imageio-ffmpeg==0.6.0" \
  "einops==0.8.1" \
  "sentencepiece==0.2.0" \
  "protobuf==4.25.6" \
  "ftfy==6.3.1" \
  "modelscope>=1.23,<2"

python -m pip check

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if torch.version.cuda != "12.4":
    raise SystemExit(f"Expected CUDA 12.4 PyTorch wheel, got {torch.version.cuda}")
PY

