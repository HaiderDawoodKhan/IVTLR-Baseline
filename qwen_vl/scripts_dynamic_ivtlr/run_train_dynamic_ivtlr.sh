#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export PYTHONUNBUFFERED=1

run_deepspeed() {
  local script="$1"
  local config_file="$2"
  local log_file="$3"

  deepspeed --master_port "$MASTER_PORT" "$script" "$config_file" --deepspeed --deepspeed_config ds_config.json 2>&1 | tee "$log_file"
}

run_deepspeed "qwenvl_run_m3cot.py" "args/qwen_m3cot.yaml" "qwenvl_m3cot_dynamic_hidden_distill.log"
run_deepspeed "qwenvl_run_m3cot.py" "args/qwen_m3cot_no_hidden_distill.yaml" "qwenvl_m3cot_dynamic_no_hidden_distill.log"
run_deepspeed "qwenvl_run_sqa.py" "args/qwen_sqa.yaml" "qwenvl_scienceqa_dynamic_hidden_distill.log"
run_deepspeed "qwenvl_run_sqa.py" "args/qwen_sqa_no_hidden_distill.yaml" "qwenvl_scienceqa_dynamic_no_hidden_distill.log"
