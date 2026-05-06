# PRISM Dataset

English | [中文](README.zh.md)

This directory contains the released PRISM benchmark dataset.

## Files

- `prism_manim_bench.jsonl`: full benchmark dataset with 10,372 examples.
- `prism_manim_bench.summary.json`: validation and cleaning summary for the released dataset.

## Schema

Each JSONL row contains four fields:

- `id`: anonymized PRISM identifier.
- `language`: `en` or `zh`.
- `instruction`: benchmark instruction in the corresponding language.
- `reference_answer`: reference Manim code answer.

The released dataset contains 5,199 English examples and 5,173 Chinese examples.
