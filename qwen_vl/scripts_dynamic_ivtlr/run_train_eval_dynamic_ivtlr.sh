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

run_deepspeed() {
  local script="$1"
  local config_file="$2"
  local log_file="$3"

  deepspeed --master_port "$MASTER_PORT" "$script" "$config_file" --deepspeed --deepspeed_config ds_config.json 2>&1 | tee "$log_file"
}

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

# Finish all M3CoT variants before starting ScienceQA.
# run_deepspeed "qwenvl_run_m3cot.py" "args/qwen_m3cot_no_hidden_distill.yaml" "qwenvl_m3cot_dynamic_no_hidden_distill.log"
eval_m3cot "$OUTPUT_ROOT/qwen_IVTLR_m3cot_no_hidden_distill" "args/qwen_m3cot_no_hidden_distill.yaml" "latest_dynamic_no_hidden_distill"

# run_deepspeed "qwenvl_run_m3cot.py" "args/qwen_m3cot_no_hidden_distill_8_steps.yaml" "qwenvl_m3cot_dynamic_no_hidden_distill.log"
# eval_m3cot "$OUTPUT_ROOT/qwen_IVTLR_m3cot_no_hidden_distill_8_steps" "args/qwen_m3cot_no_hidden_distill_8_steps.yaml" "latest_dynamic_no_hidden_distill_8_steps"

# run_deepspeed "qwenvl_run_m3cot.py" "args/qwen_m3cot.yaml" "qwenvl_m3cot_dynamic_hidden_distill.log"
# eval_m3cot "$OUTPUT_ROOT/qwen_IVTLR_m3cot" "args/qwen_m3cot.yaml" "latest_dynamic_hidden_distill"

# run_deepspeed "qwenvl_run_sqa.py" "args/qwen_sqa.yaml" "qwenvl_scienceqa_dynamic_hidden_distill.log"
# eval_sqa "$OUTPUT_ROOT/qwen_IVTLR_sqa" "args/qwen_sqa.yaml" "latest_dynamic_hidden_distill"

# run_deepspeed "qwenvl_run_sqa.py" "args/qwen_sqa_no_hidden_distill.yaml" "qwenvl_scienceqa_dynamic_no_hidden_distill.log"
# eval_sqa "$OUTPUT_ROOT/qwen_IVTLR_sqa_no_hidden_distill" "args/qwen_sqa_no_hidden_distill.yaml" "latest_dynamic_no_hidden_distill"
