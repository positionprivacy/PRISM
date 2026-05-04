#!/usr/bin/env bash
set -euo pipefail

# Example only: replace these paths with your own evaluated scripts and fitted reference parameters.
export PADVC_HF_CACHE="${PADVC_HF_CACHE:-./.cache/hf/hub}"
export PADVC_OCR_BACKEND="${PADVC_OCR_BACKEND:-rapidocr}"
export RENDER_WORKERS="${RENDER_WORKERS:-4}"
export AUDIT_WORKERS="${AUDIT_WORKERS:-4}"
export PADVC_SHARDS="${PADVC_SHARDS:-4}"
export TD_WORKERS="${TD_WORKERS:-8}"

scripts/run_evaluation_pipeline.sh \
  your_model_run/cleaned_scripts \
  results/eval_your_model \
  your_model_run/task_manifest.json \
  data/your_prompts.jsonl \
  results/reference_padvc/padvc_norm_params.json \
  results/reference_td/td_center_params.json \
  your_model_name
