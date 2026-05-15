#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=29501
export PYTHONUNBUFFERED=1

run_deepspeed() {
  local script="$1"
  local config_file="$2"
  local log_file="$3"
  deepspeed --master_port "$MASTER_PORT" "$script" "$config_file" --deepspeed --deepspeed_config ds_config.json 2>&1 | tee "$log_file"
}

run_deepspeed qwenvl_run.py args/qwen_m3cot.yaml qwenvl_m3cot.log
run_deepspeed qwenvl_run_sqa.py args/qwen_sqa.yaml qwenvl_scienceqa.log