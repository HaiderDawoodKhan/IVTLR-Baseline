#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

OUTPUT_ROOT="outuputs_dynamic_ivtlr"

find_latest_checkpoint() {
  local dir="$1"
  local latest

  latest=$(find "$dir" -maxdepth 1 -type f -name "epoch_*_full_model_fp32.pth" | sort -V | tail -n 1)
  if [[ -z "$latest" ]]; then
    return 1
  fi
  echo "$latest"
}

eval_m3cot() {
  local checkpoint_dir="$1"
  local config_file="$2"
  local run_tag="$3"
  local checkpoint

  checkpoint=$(find_latest_checkpoint "$checkpoint_dir") || {
    echo "No M3CoT checkpoint found in $checkpoint_dir" >&2
    exit 1
  }

  echo "Using M3CoT checkpoint: $checkpoint"
  python "infer_m3cot.py" --config "$config_file" --checkpoint "$checkpoint" --run_tag "$run_tag" --output_root "$OUTPUT_ROOT"
}

eval_sqa() {
  local checkpoint_dir="$1"
  local config_file="$2"
  local run_tag="$3"
  local checkpoint

  checkpoint=$(find_latest_checkpoint "$checkpoint_dir") || {
    echo "No ScienceQA checkpoint found in $checkpoint_dir" >&2
    exit 1
  }

  echo "Using ScienceQA checkpoint: $checkpoint"
  python "infer_sqa.py" --config "$config_file" --checkpoint "$checkpoint" --run_tag "$run_tag" --output_root "$OUTPUT_ROOT"
}

eval_m3cot "$OUTPUT_ROOT/qwen_IVTLR_m3cot" "args/qwen_m3cot.yaml" "latest_dynamic_hidden_distill"
eval_sqa "$OUTPUT_ROOT/qwen_IVTLR_sqa" "args/qwen_sqa.yaml" "latest_dynamic_hidden_distill"
eval_m3cot "$OUTPUT_ROOT/qwen_IVTLR_m3cot_no_hidden_distill" "args/qwen_m3cot_no_hidden_distill.yaml" "latest_dynamic_no_hidden_distill"
eval_sqa "$OUTPUT_ROOT/qwen_IVTLR_sqa_no_hidden_distill" "args/qwen_sqa_no_hidden_distill.yaml" "latest_dynamic_no_hidden_distill"
