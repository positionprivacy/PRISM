# PRISM 数据集

[English](README.md) | 中文

这里包含已发布的 PRISM benchmark 数据集。

## 文件

- `prism_dataset.jsonl`：完整 benchmark 数据集，共 10,372 条。
- `prism_dataset.summary.json`：发布数据的清洗与验证摘要。

## 字段

每一行 JSONL 包含四个字段：

- `id`：匿名化后的 PRISM 样本编号。
- `language`：`en` 或 `zh`。
- `instruction`：与语言一致的 benchmark 指令。
- `reference_answer`：参考 Manim 代码答案。

发布数据包含 5,199 条英文样本和 5,173 条中文样本。
