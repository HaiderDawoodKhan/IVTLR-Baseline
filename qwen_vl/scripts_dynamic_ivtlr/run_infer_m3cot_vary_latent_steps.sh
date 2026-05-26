#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_ROOT="outputs_dynamic_ivtlr"
CHECKPOINT_DIR="$OUTPUT_ROOT/qwen_IVTLR_m3cot_no_hidden_distill_8_steps"
CONFIG_FILE="args/qwen_m3cot_no_hidden_distill_8_steps.yaml"

find_latest_checkpoint() {
  local dir="$1"
  local latest

  latest=$(find "$dir" -maxdepth 1 -type f -name "epoch_*_full_model_fp32.pth" | sort -V | tail -n 1)
  if [[ -z "$latest" ]]; then
    return 1
  fi
  echo "$latest"
}

checkpoint=$(find_latest_checkpoint "$CHECKPOINT_DIR") || {
  echo "No M3CoT checkpoint found in $CHECKPOINT_DIR" >&2
  exit 1
}

echo "Using M3CoT checkpoint: $checkpoint"

for latent_steps in {0..10}; do
  run_tag="latent_steps_${latent_steps}"
  echo "Running inference with latent_steps=$latent_steps"
  python "infer_m3cot.py" \
    --config "$CONFIG_FILE" \
    --checkpoint "$checkpoint" \
    --run_tag "$run_tag" \
    --output_root "$OUTPUT_ROOT" \
    --latent_steps "$latent_steps"
done
