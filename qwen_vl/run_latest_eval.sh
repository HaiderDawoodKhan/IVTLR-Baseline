#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M3COT_DIR="$ROOT/output/qwen_IVTLR_m3cot"
SQA_DIR="$ROOT/output/qwen_IVTLR_sqa"

find_latest_checkpoint() {
  local dir="$1"
  local pattern="$2"
  local latest

  latest=$(find "$dir" -maxdepth 1 -type f -name "$pattern" | sort -V | tail -n 1)
  if [[ -z "$latest" ]]; then
    return 1
  fi
  echo "$latest"
}

m3cot_ckpt=$(find_latest_checkpoint "$M3COT_DIR" "epoch_*_full_model_fp32.pth") || {
  echo "No M3CoT checkpoint found in $M3COT_DIR" >&2
  exit 1
}

sqa_ckpt=$(find_latest_checkpoint "$SQA_DIR" "epoch_*_full_model_fp32.pth") || {
  echo "No SQA checkpoint found in $SQA_DIR" >&2
  exit 1
}

echo "Using M3CoT checkpoint: $m3cot_ckpt"
python "$ROOT/infer.py" --checkpoint "$m3cot_ckpt"

echo "Using SQA checkpoint: $sqa_ckpt"
python "$ROOT/infer_sqa.py" --config args/qwen_sqa.yaml --checkpoint "$sqa_ckpt"
