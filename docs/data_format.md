# Data Formats

This document defines the file formats expected by the public pipeline.

## 1. Prompt JSONL

Used by the evaluation pipeline to recover prompt-side information such as instruction text, language, and source ids. It is also accepted by `scripts/generate_code.py` when you want to generate model outputs inside this repository.

Each line is a JSON object with at least:

```json
{"id": "sample_0001", "instruction": "Write a Manim scene for ..."}
```

Optional fields such as `language`, `difficulty`, or `source_id` are preserved when possible.

## 2. Task Manifest JSON

Used by evaluation scripts to map a sample id to its script path, markdown source, or reference code.

Example:

```json
[
  {
    "id": "example_0001",
    "md_path": "examples/markdowns/example_0001.md",
    "source_file_path": "examples/reference_code/example_0001.py"
  }
]
```

For evaluation-only use, the most important field is still `id`; the remaining fields can be as light as your workflow allows, as long as downstream scripts can align sample ids consistently.

## 3. Minimal Inputs for Evaluating Existing Outputs

If you already have a model output directory and only want to evaluate it, prepare:

- a directory of Manim scripts such as `your_model_run/cleaned_scripts/*.py`
- a matching `task_manifest.json`
- a prompt JSONL with the same sample ids
- fitted reference parameters for PADVC and TD

This is sufficient for `scripts/run_evaluation_pipeline.sh`.

## 4. Reference JSONL

Produced by `scripts/prepare_reference_dataset.py`.

Each line contains:

- `id`
- `instruction`
- `output`
- `md_path`
- `source_file_path`

This file is typically used to fit reference-center parameters for PADVC and TD.

## 5. Generation Run Layout

`scripts/generate_code.py` writes a directory with the following structure:

- `cleaned_scripts/`: cleaned Manim code, one `.py` per sample
- `raw_outputs/`: raw model responses
- `metadata/`: per-sample metadata including token usage and latency
- `prompt_snapshots/`: final prompts sent to the model
- `markdown_snapshots/`: optional markdown snapshots when prompts are built from markdown files
- `results.json`: run-level summary
- `task_manifest.json`: manifest for downstream rendering and scoring

## 6. Evaluation Run Layout

`scripts/run_evaluation_pipeline.sh` writes:

- `videos/`: rendered `.mp4` files
- `render_results.json`
- `audit/`
- `padvc_shards/`
- `padvc_final/`
- `td_final/`
- `ocr_cache/`
- `logs/`
