# Manim-Bench Toolkit

English | [中文](README.zh.md)

A lightweight benchmark toolkit for evaluating Manim code outputs. It focuses on rendering, deterministic spatial audit, PADVC/TD scoring, and text-expansion analysis. A small generation wrapper is included, but evaluation is the primary workflow.

Use the toy files in `examples/` for smoke tests, then place your own prompts, manifests, reference code, and evaluation outputs under `data/` and `results/` or configure custom paths with environment variables.

## Repository Layout

- `scripts/`: command-line tools for evaluation, metrics, and optional generation
- `manim_bench/llm_call/`: minimal LLM client wrapper for optional generation
- `docs/`: technical documentation for data formats, metrics, and audit semantics
- `examples/`: small toy inputs and configuration examples
- `data/`: local dataset workspace
- `results/`: local output workspace

## System Requirements

Recommended environment:

- Linux or macOS
- Python 3.10+
- Manim Community Edition 0.19.0
- FFmpeg
- Cairo / Pango / `pkg-config` build libraries
- LaTeX toolchain for `Tex` and `MathTex`
- CJK-capable fonts if you render Chinese text

Ubuntu example:

```bash
sudo apt-get update
sudo apt-get install -y \
  ffmpeg pkg-config libcairo2-dev libpango1.0-dev \
  texlive texlive-latex-extra texlive-fonts-recommended \
  texlive-xetex dvisvgm ghostscript \
  fonts-noto-cjk fontconfig
```

macOS example:

```bash
brew install ffmpeg cairo pango pkg-config mactex-no-gui font-noto-sans-cjk
```

## Python Installation

Create an isolated environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then check the environment:

```bash
python scripts/check_environment.py
```

## OCR and Small Models

PADVC depends on OCR and text-similarity models.

Default OCR backend:

- default: `rapidocr-onnxruntime`
- optional fallback: `paddleocr`

Text-similarity models used by `scripts/padvc.py`:

- Chinese: `shibing624/text2vec-base-chinese`
- English/multilingual: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

The PADVC implementation defaults to offline Hugging Face mode. Prepare model snapshots before running PADVC, then point the scripts to them:

```bash
export PADVC_HF_CACHE=/path/to/huggingface/hub
# or set explicit local snapshot directories
export PADVC_ZH_MODEL=/path/to/text2vec-base-chinese
export PADVC_EN_MODEL=/path/to/paraphrase-multilingual-MiniLM-L12-v2
```

Useful runtime variables:

```bash
export PADVC_OCR_BACKEND=rapidocr
export PADVC_OCR_CACHE_DIR=results/ocr_cache
export PADVC_DEBUG=0
```

## Quickstart: Evaluate Existing Model Outputs

The main workflow assumes you already have a directory of generated Manim scripts, for example:

- `your_model_run/cleaned_scripts/*.py`
- `your_model_run/task_manifest.json`
- a prompt JSONL such as `data/your_prompts.jsonl`
- fitted reference parameters for PADVC and TD

If your outputs were not produced by `scripts/generate_code.py`, prepare the minimal manifest and prompt JSONL formats described in `docs/data_format.md`.

Then run the full evaluation pipeline:

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

For stage-by-stage commands, see `scripts/README.md`.

## Reference Parameters

`PADVC_center` and `TD_center` require reference statistics. Fit them on your own curated reference set:

```bash
python scripts/fit_reference_padvc.py \
  --dataset-jsonl data/your_reference_dataset.jsonl \
  --video-root results/reference_videos \
  --output-dir results/reference_padvc

python scripts/fit_reference_td.py \
  --dataset-jsonl results/reference_padvc/padvc_reference_raw_scores.jsonl \
  --output-dir results/reference_td
```

The example parameter files under `examples/params/` are placeholders for smoke tests only.

## Optional: Generate Model Outputs

If you want to produce model outputs inside this repository, copy the template and fill in your provider settings:

```bash
cp manim_bench/llm_call/config.example.json manim_bench/llm_call/config.json
```

You can also select a different config path with:

```bash
export MANIM_BENCH_LLM_CONFIG=/path/to/config.json
```

Then run the optional generation wrapper:

```bash
python scripts/generate_code.py \
  --input-jsonl examples/sample_prompts.jsonl \
  --instruction-field instruction \
  --model your-model-name \
  --workers 2 \
  --temperature 0.7 \
  --output-dir results/example_generation
```

## Documentation

- `docs/data_format.md`: expected JSON/JSONL layouts
- `docs/spatial_audit.md`: spatial-audit semantics
- `docs/metrics.md`: PADVC, TD, and text-expansion overview
- `docs/code_error_taxonomy.md`: code-failure categories
- `scripts/README.md`: script inventory and command examples
