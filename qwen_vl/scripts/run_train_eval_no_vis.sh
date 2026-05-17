#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=29501
export PYTHONUNBUFFERED=1

# IVTLR evaluate
python infer_m3cot.py --checkpoint output/qwen_IVTLR_m3cot/epoch_16_full_model_fp32.pth --run_tag no_visual --disable_visual_insert
python infer_sqa.py --config args/qwen_sqa.yaml --checkpoint output/qwen_IVTLR_sqa/epoch_16_full_model_fp32.pth --run_tag no_visual --disable_visual_insert

run_deepspeed() {
  local script="$1"
  local config_file="$2"
  local log_file="$3"
  deepspeed --master_port "$MASTER_PORT" "$script" "$config_file" --deepspeed --deepspeed_config ds_config.json 2>&1 | tee "$log_file"
}

# M3CoT
run_deepspeed qwenvl_run_m3cot.py args/qwen_m3cot_no_vis.yaml qwenvl_m3cot_no_vis.log --disable_visual_insert
python infer_m3cot.py --config args/qwen_m3cot_no_vis.yaml --checkpoint output/qwen_IVTLR_m3cot_no_vis/epoch_16_full_model_fp32.pth --run_tag "epoch_16" --disable_visual_insert --disabled_model

# ScienceQA
run_deepspeed qwenvl_run_sqa.py args/qwen_sqa_no_vis.yaml qwenvl_scienceqa_no_vis.log --disable_visual_insert
python infer_sqa.py --config args/qwen_sqa_no_vis.yaml --checkpoint output/qwen_IVTLR_sqa_no_vis/epoch_16_full_model_fp32.pth --run_tag "epoch_16" --disable_visual_insert --disabled_model