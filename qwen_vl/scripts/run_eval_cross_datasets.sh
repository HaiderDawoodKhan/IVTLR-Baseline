#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# M3CoT ckpt on SQA evaluation
python infer_m3cot.py --checkpoint output/qwen_IVTLR_sqa/epoch_16_full_model_fp32.pth --run_tag sqa_ckpt

# ScienceQA on M3CoT ckpt evaluation
python infer_sqa.py --config args/qwen_sqa.yaml --checkpoint output/qwen_IVTLR_m3cot/epoch_16_full_model_fp32.pth --run_tag m3cot_ckpt
