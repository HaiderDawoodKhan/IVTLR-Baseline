#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_ROOT="outputs_lvar"
CONFIG_FILE="args/qwen_m3cot_no_hidden_distill_8_steps.yaml"


for latent_steps in {9,10}; do
  run_tag="latent_steps_${latent_steps}"
  echo "Running inference with latent_steps=$latent_steps"
  python "infer_m3cot.py" \
    --config "$CONFIG_FILE" \
    --checkpoint "/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/outputs_dynamic_ivtlr/qwen_IVTLR_m3cot_no_hidden_distill_8_steps_prefix_span/epoch_20_full_model_fp32.pth" \
    --run_tag "$run_tag" \
    --output_root "$OUTPUT_ROOT" \
    --latent_steps "$latent_steps" \
    --use_validation_set
done

# run_tag="dynamic_latent_steps"
# python "infer_m3cot.py" \
#   --config "$CONFIG_FILE" \
#   --checkpoint "/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/outputs_dynamic_ivtlr/qwen_IVTLR_m3cot_no_hidden_distill_8_steps_prefix_span/epoch_20_full_model_fp32.pth" \
#   --run_tag "$run_tag" \
#   --output_root "$OUTPUT_ROOT" \
#   --dynamic_latent_steps \
#   --use_validation_set

for latent_steps in {0..10}; do
  run_tag="latent_steps_${latent_steps}"
  echo "Running inference with latent_steps=$latent_steps"
  python "infer_sqa.py" \
    --config "args/qwen_sqa_no_hidden_distill_8_steps.yaml" \
    --checkpoint "/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/lvar_sqa_phase_1/qwen_IVTLR_sqa_no_hidden_distill_8_steps_prefix_span/epoch_20_full_model_fp32.pth" \
    --run_tag "$run_tag" \
    --output_root "outputs_lvar_sqa" \
    --latent_steps "$latent_steps"
done

for latent_steps in {0..10}; do
  run_tag="latent_steps_${latent_steps}"
  echo "Running inference with latent_steps=$latent_steps"
  python "infer_sqa.py" \
    --config "args/qwen_sqa_no_hidden_distill_8_steps.yaml" \
    --checkpoint "/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/lvar_sqa_phase_1/qwen_IVTLR_sqa_no_hidden_distill_8_steps_prefix_span/epoch_20_full_model_fp32.pth" \
    --run_tag "$run_tag" \
    --output_root "outputs_lvar_sqa_validation" \
    --latent_steps "$latent_steps" \
    --use_validation_set
done