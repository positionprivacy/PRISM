# 脚本说明

[English](README.md) | 中文

这个目录包含公开版本的评测主流程，以及少量可选的数据准备与生成工具。

## 评测主流程

- `render_directory.py`：批量渲染目录下的 `.py` 场景文件并输出 `.mp4`。
- `audit_single.py`：对单个 Manim 脚本执行空间审计，并输出分段级报告。
- `audit_batch.py`：`audit_single.py` 的批处理封装，支持断点续跑。
- `padvc.py`：PADVC 核心实现；默认使用 `rapidocr`，并支持通过 `PADVC_HF_CACHE`、`PADVC_ZH_MODEL`、`PADVC_EN_MODEL` 指向离线模型。
- `score_padvc.py`：计算逐样本 `PADVC_raw`、`PADVC_center` 和 `uPADVC`。
- `fit_reference_padvc.py`：根据参考视频拟合 PADVC center 参数。
- `score_td.py`：计算逐样本 `TD_raw`、`TD_center` 和 `uTD`。
- `fit_reference_td.py`：根据参考视频拟合 TD center 参数。
- `compute_text_expansion.py`：直接从生成代码估计文本扩展度。
- `run_render_padvc_pipeline.sh`：在已有 audit 结果时，执行 render + PADVC 的便捷脚本。
- `run_evaluation_pipeline.sh`：端到端执行 render、audit、PADVC、TD 和可选的 text expansion。

## 数据准备

- `prepare_reference_dataset.py`：根据 task manifest、Markdown 和参考代码构造 reference-answer JSONL。
- `curate_dataset.py`：数据清洗工具，包含两个子命令：
  - `clean-waits`：基于 AST 感知清洗 `self.wait(...)`
  - `replace-image-rows`：将含图片条目替换为未使用的无图候选条目

## 可选生成

- `generate_code.py`：调用配置好的 LLM，保存清洗后的 Manim 代码和元数据。
- `error_taxonomy.py`：按 benchmark taxonomy 分类代码生成失败类型。
- `merge_padvc_shards.py`：合并多个 PADVC shard 输出。
- `repo_config.py`：仓库级路径与环境变量辅助函数。
- `check_environment.py`：检查 Python 包、外部命令和关键环境变量。

## 命令示例

### 评测已有模型输出目录

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

### 只做渲染与审计

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

### 计算 PADVC 与 TD

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

### 拟合参考参数

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

### 环境检查

```bash
python scripts/check_environment.py
```

### 可选生成

```bash
python scripts/generate_code.py \
  --input-jsonl examples/sample_prompts.jsonl \
  --instruction-field instruction \
  --model your-model-name \
  --workers 2 \
  --temperature 0.7 \
  --output-dir results/example_generation
```
