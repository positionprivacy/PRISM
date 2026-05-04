# Scripts Overview

English | [中文](README.zh.md)

This directory contains the public evaluation workflow and a small set of optional data-prep and generation utilities.

## Evaluation

- `render_directory.py`: render a directory of `.py` scene files into `.mp4` outputs.
- `audit_single.py`: run spatial auditing on one Manim script and emit a segment-level report.
- `audit_batch.py`: batch wrapper for `audit_single.py` with resumable JSON summaries.
- `padvc.py`: core PADVC implementation; defaults to `rapidocr` and supports offline embedding models via `PADVC_HF_CACHE`, `PADVC_ZH_MODEL`, and `PADVC_EN_MODEL`.
- `score_padvc.py`: compute per-sample `PADVC_raw`, `PADVC_center`, and `uPADVC`.
- `fit_reference_padvc.py`: fit PADVC center parameters from reference videos.
- `score_td.py`: compute per-sample `TD_raw`, `TD_center`, and `uTD`.
- `fit_reference_td.py`: fit TD center parameters from reference videos.
- `compute_text_expansion.py`: estimate text expansion directly from generated code.
- `run_render_padvc_pipeline.sh`: convenience wrapper for render + PADVC scoring when audit results already exist.
- `run_evaluation_pipeline.sh`: end-to-end wrapper for render, audit, PADVC, TD, and optional text expansion.

## Data Preparation

- `prepare_reference_dataset.py`: build reference-answer JSONL files from task manifests and Markdown sources.
- `curate_dataset.py`: dataset cleanup utilities with two subcommands:
  - `clean-waits`: AST-aware cleanup of `self.wait(...)` calls.
  - `replace-image-rows`: replace image-containing rows with unused no-image candidates.

## Optional Generation

- `generate_code.py`: call the configured LLM and save cleaned Manim code outputs plus metadata.
- `error_taxonomy.py`: classify code-generation failures into the benchmark taxonomy.
- `merge_padvc_shards.py`: merge PADVC shard outputs into one result set.
- `repo_config.py`: repository-local path and environment helpers.
- `check_environment.py`: check Python packages, external commands, and key environment variables.

## Command Examples

### Evaluate an Existing Output Directory

```bash
scripts/run_evaluation_pipeline.sh \
  your_model_run/cleaned_scripts \
  results/eval_your_model \
  your_model_run/task_manifest.json \
  data/your_prompts.jsonl \
  results/reference_padvc/padvc_norm_params.json \
  results/reference_td/td_center_params.json \
  your_model_name
```

### Render and Audit Only

```bash
python scripts/render_directory.py \
  --input-dir your_model_run/cleaned_scripts \
  --output-dir results/eval_your_model/videos \
  --results-json results/eval_your_model/render_results.json \
  --workers 4

python scripts/audit_batch.py \
  --input-dir your_model_run/cleaned_scripts \
  --output-dir results/eval_your_model/audit \
  --workers 4 \
  --no-images
```

### Score PADVC and TD

```bash
python scripts/score_padvc.py \
  --task-manifest your_model_run/task_manifest.json \
  --audit-results results/eval_your_model/audit/results.json \
  --render-results results/eval_your_model/render_results.json \
  --sample-jsonl data/your_prompts.jsonl \
  --norm-params results/reference_padvc/padvc_norm_params.json \
  --output-dir results/eval_your_model/padvc_final \
  --ocr-cache-dir results/eval_your_model/ocr_cache \
  --ocr-backend rapidocr \
  --quiet-padvc

python scripts/score_td.py \
  --input-jsonl results/eval_your_model/padvc_final/padvc_scores.jsonl \
  --params-json results/reference_td/td_center_params.json \
  --output-dir results/eval_your_model/td_final \
  --workers 8
```

### Fit Reference Parameters

```bash
python scripts/fit_reference_padvc.py \
  --dataset-jsonl data/your_reference_dataset.jsonl \
  --video-root results/reference_videos \
  --output-dir results/reference_padvc \
  --ocr-backend rapidocr

python scripts/fit_reference_td.py \
  --dataset-jsonl results/reference_padvc/padvc_reference_raw_scores.jsonl \
  --output-dir results/reference_td
```

### Check Environment

```bash
python scripts/check_environment.py
```

### Optional Generation

```bash
python scripts/generate_code.py \
  --input-jsonl examples/sample_prompts.jsonl \
  --instruction-field instruction \
  --model your-model-name \
  --workers 2 \
  --temperature 0.7 \
  --output-dir results/example_generation
```
