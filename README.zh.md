# Manim-Bench Toolkit

[English](README.md) | 中文

这是一个用于评测 Manim 代码输出的轻量 benchmark 工具仓库，重点覆盖渲染、确定性空间审计、PADVC/TD 计算和文本扩展度分析。仓库中也提供了一个小型生成封装，但主要工作流是评测而不是生成。

你可以先使用 `examples/` 中的 toy 文件进行 smoke test，再将自己的 prompts、manifests、参考代码和评测输出放到 `data/` 与 `results/`，或通过环境变量指定自定义路径。

## 仓库结构

- `scripts/`：评测、指标计算以及可选生成的命令行工具
- `manim_bench/llm_call/`：用于可选生成的最小化 LLM 调用封装
- `docs/`：数据格式、指标和审计语义的技术文档
- `examples/`：小型示例输入和配置模板
- `data/`：本地数据工作区
- `results/`：本地结果工作区

## 系统依赖

推荐环境：

- Linux 或 macOS
- Python 3.10+
- Manim Community Edition 0.19.0
- FFmpeg
- Cairo / Pango / `pkg-config` 编译依赖
- `Tex` / `MathTex` 所需的 LaTeX 工具链
- 若需要渲染中文，建议安装支持 CJK 的字体

Ubuntu 示例：

```bash
sudo apt-get update
sudo apt-get install -y \
  ffmpeg pkg-config libcairo2-dev libpango1.0-dev \
  texlive texlive-latex-extra texlive-fonts-recommended \
  texlive-xetex dvisvgm ghostscript \
  fonts-noto-cjk fontconfig
```

macOS 示例：

```bash
brew install ffmpeg cairo pango pkg-config mactex-no-gui font-noto-sans-cjk
```

## Python 安装

建议创建独立环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

安装后可以运行环境检查：

```bash
python scripts/check_environment.py
```

## OCR 与小模型

PADVC 依赖 OCR 和文本相似度模型。

默认 OCR 后端：

- 默认：`rapidocr-onnxruntime`
- 可选兜底：`paddleocr`

`scripts/padvc.py` 使用的文本相似度模型：

- 中文：`shibing624/text2vec-base-chinese`
- 英文 / 多语言：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

PADVC 默认以离线 Hugging Face 模式运行。建议先准备好模型快照，再通过环境变量指向本地路径：

```bash
export PADVC_HF_CACHE=/path/to/huggingface/hub
# 或直接指定本地模型目录
export PADVC_ZH_MODEL=/path/to/text2vec-base-chinese
export PADVC_EN_MODEL=/path/to/paraphrase-multilingual-MiniLM-L12-v2
```

常用运行时变量：

```bash
export PADVC_OCR_BACKEND=rapidocr
export PADVC_OCR_CACHE_DIR=results/ocr_cache
export PADVC_DEBUG=0
```

## Quickstart：评测已有模型输出

主工作流假设你已经有一个模型输出目录，例如：

- `your_model_run/cleaned_scripts/*.py`
- `your_model_run/task_manifest.json`
- 一个 prompt JSONL，例如 `data/your_prompts.jsonl`
- 已拟合好的 PADVC 与 TD 参考参数

如果这些输出不是由 `scripts/generate_code.py` 生成的，请先按照 `docs/data_format.md` 中的格式准备最小 manifest 和 prompt JSONL。

然后直接运行完整评测流水线：

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

如果你想分步骤运行，请查看 `scripts/README.md`。

## 参考参数

`PADVC_center` 和 `TD_center` 都需要参考答案分布参数。请基于你自己的 reference set 进行拟合：

```bash
python scripts/fit_reference_padvc.py \
  --dataset-jsonl data/your_reference_dataset.jsonl \
  --video-root results/reference_videos \
  --output-dir results/reference_padvc

python scripts/fit_reference_td.py \
  --dataset-jsonl results/reference_padvc/padvc_reference_raw_scores.jsonl \
  --output-dir results/reference_td
```

`examples/params/` 下的参数文件只是 smoke test 占位示例，不代表真实 benchmark 参数。

## 可选：生成模型输出

如果你希望在仓库中直接生成模型输出，可以先复制模板并填写 provider 配置：

```bash
cp manim_bench/llm_call/config.example.json manim_bench/llm_call/config.json
```

你也可以通过环境变量指定其它配置文件：

```bash
export MANIM_BENCH_LLM_CONFIG=/path/to/config.json
```

然后再运行可选的生成脚本：

```bash
python scripts/generate_code.py \
  --input-jsonl examples/sample_prompts.jsonl \
  --instruction-field instruction \
  --model your-model-name \
  --workers 2 \
  --temperature 0.7 \
  --output-dir results/example_generation
```

## 文档

- `docs/data_format.md`：输入输出 JSON / JSONL 格式说明
- `docs/spatial_audit.md`：空间审计语义说明
- `docs/metrics.md`：PADVC、TD 与文本扩展度说明
- `docs/code_error_taxonomy.md`：代码错误分类说明
- `scripts/README.md`：脚本列表与命令示例
