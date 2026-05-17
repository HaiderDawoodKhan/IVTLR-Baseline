#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=29501
export PYTHONUNBUFFERED=1

# IVTLR evaluate
python infer_m3cot.py --checkpoint output/qwen_IVTLR_m3cot/epoch_16_full_model_fp32.pth --run_tag no_reasoning --no-reasoning
python infer_sqa.py --config args/qwen_sqa.yaml --checkpoint output/qwen_IVTLR_sqa/epoch_16_full_model_fp32.pth --run_tag no_reasoning --no-reasoning
