#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# M3CoT evaluation
python infer_m3cot.py --checkpoint output/qwen_IVTLR_m3cot/epoch_4_full_model_fp32.pth --run_tag epoch_4
python infer_m3cot.py --checkpoint output/qwen_IVTLR_m3cot/epoch_8_full_model_fp32.pth --run_tag epoch_8
python infer_m3cot.py --checkpoint output/qwen_IVTLR_m3cot/epoch_12_full_model_fp32.pth --run_tag epoch_12

# ScienceQA evaluation
python infer_sqa.py --config args/qwen_sqa.yaml --checkpoint output/qwen_IVTLR_sqa/epoch_4_full_model_fp32.pth --run_tag epoch_4
python infer_sqa.py --config args/qwen_sqa.yaml --checkpoint output/qwen_IVTLR_sqa/epoch_8_full_model_fp32.pth --run_tag epoch_8
python infer_sqa.py --config args/qwen_sqa.yaml --checkpoint output/qwen_IVTLR_sqa/epoch_12_full_model_fp32.pth --run_tag epoch_12
