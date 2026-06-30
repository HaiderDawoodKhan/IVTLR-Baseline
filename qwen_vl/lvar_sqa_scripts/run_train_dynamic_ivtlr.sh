#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_ROOT="lvar_sqa_phase_1"
GENERATED_CONFIG_DIR="args/generated_curriculum"
BASE_SQA_8_STEP_CONFIG="args/qwen_sqa_no_hidden_distill_8_steps.yaml"
M3COT_RHO_SCHEDULE='[0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]'
FIXED_MASK_SCHEDULE='[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8]'

run_deepspeed() {
  local script="$1"
  local config_file="$2"
  local log_file="$3"

  deepspeed --master_port "$MASTER_PORT" "$script" "$config_file" --deepspeed --deepspeed_config ds_config.json 2>&1 | tee "$log_file"
}

create_curriculum_config() {
  local base_config="$1"
  local output_config="$2"
  local run_name="$3"
  local prefix_span="$4"
  local fixed_mask="$5"

  mkdir -p "$GENERATED_CONFIG_DIR"
  python - "$base_config" "$output_config" "$run_name" "$prefix_span" "$fixed_mask" "$M3COT_RHO_SCHEDULE" "$FIXED_MASK_SCHEDULE" <<'PY'
import ast
import sys
import yaml

base_config, output_config, run_name, prefix_span, fixed_mask, rho_schedule, fixed_mask_schedule = sys.argv[1:]

with open(base_config) as f:
    config = yaml.safe_load(f)

config.update(
    {
        "name": run_name,
        "prefix_span": prefix_span == "true",
        "fixed_mask_step_curriculum": fixed_mask == "true",
        "fixed_mask_max_steps": 8,
        "fixed_mask_schedule": ast.literal_eval(fixed_mask_schedule),
        "rho_schedule": ast.literal_eval(rho_schedule),
        "num_epochs": 20,
    }
)

with open(output_config, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY
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

create_curriculum_config \
  "$BASE_SQA_8_STEP_CONFIG" \
  "$GENERATED_CONFIG_DIR/qwen_sqa_no_hidden_distill_8_steps_prefix_span.yaml" \
  "qwen_IVTLR_sqa_no_hidden_distill_8_steps_prefix_span" \
  "true" \
  "false"
run_deepspeed \
  "qwenvl_run_sqa.py" \
  "$GENERATED_CONFIG_DIR/qwen_sqa_no_hidden_distill_8_steps_prefix_span.yaml" \
  "qwenvl_sqa_dynamic_no_hidden_distill_8_steps_prefix_span.log"
# eval_sqa \
#   "$OUTPUT_ROOT/qwen_IVTLR_sqa_no_hidden_distill_8_steps_prefix_span" \
#   "$GENERATED_CONFIG_DIR/qwen_sqa_no_hidden_distill_8_steps_prefix_span.yaml" \
#   "latest_dynamic_no_hidden_distill_8_steps_prefix_span"
