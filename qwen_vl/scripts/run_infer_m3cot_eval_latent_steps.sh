#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_ROOT="/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/output/inference/m3cot"
CONFIG_FILE="args/qwen_m3cot.yaml"

for latent_steps in {1,2,4,5}; do
  run_tag="original_ivtlr_latent_steps_${latent_steps}"
  echo "Running inference with latent_steps=$latent_steps"
  python "infer_m3cot.py" \
    --config "$CONFIG_FILE" \
    --checkpoint "/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/output/qwen_IVTLR_m3cot/epoch_16_full_model_fp32.pth" \
    --run_tag "$run_tag" \
    --output_root "$OUTPUT_ROOT" \
    --latent_steps "$latent_steps"
done
