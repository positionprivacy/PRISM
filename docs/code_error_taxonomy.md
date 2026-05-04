# Code Error Taxonomy

`scripts/error_taxonomy.py` groups failed generations into the following categories.

| Category | Definition | What it usually means |
| --- | --- | --- |
| API hallucination | The code references a class, constant, method, or interface that does not exist in the target Manim version. | The model is inventing domain knowledge. |
| API misuse | The API exists, but arguments, attributes, or call patterns are invalid. | The model remembers names but not usage constraints. |
| Text rendering error | Rendering fails because of LaTeX, MarkupText, font, encoding, or related text-toolchain issues. | The model is unstable on text-and-render integration. |
| Format pollution | The response mixes code with explanations, markdown fences, or reasoning traces that break execution. | The model fails instruction following before code execution even starts. |
| Syntax error | The Python source is syntactically invalid. | The model fails at basic code generation. |
| Other error | Any remaining runtime failure not covered above. | Catch-all bucket for uncovered cases. |

The taxonomy is intended for benchmark diagnosis, not for perfect root-cause attribution.
