# Metrics

This repository implements three public metric families: PADVC, TD, and text expansion.

## PADVC

Prompt-Aware Dynamic Visual Complexity (PADVC) measures non-text geometric evolution relative to text burden and prompt complexity.

The public scripts expose:

- `PADVC_raw`: raw dynamic intensity
- `PADVC_center`: reference-centered version of `PADVC_raw`
- `uPADVC`: `PADVC_raw` multiplied by binary visual pass

At a high level:

- OCR is used to estimate text masks on sampled frames.
- Frame-difference events are extracted from the rendered video.
- Non-text structural changes are measured with Laplacian energy.
- Prompt complexity is estimated from markdown structure and action words.

Reference fitting is handled by `scripts/fit_reference_padvc.py`, and per-sample scoring is handled by `scripts/score_padvc.py`.

## TD

Temporal Density (TD) measures pixel-level change per unit time.

The public scripts expose:

- `TD_raw`: average thresholded frame-difference rate per second
- `TD_center`: reference-centered version of `TD_raw`
- `uTD`: `TD_center` multiplied by visual structure rate

Reference fitting is handled by `scripts/fit_reference_td.py`, and per-sample scoring is handled by `scripts/score_td.py`.

## Text Expansion

`scripts/compute_text_expansion.py` estimates how much textual information the generated animation introduces relative to the prompt.

The current implementation is code-based rather than OCR-based:

- it parses the generated Python AST
- extracts text-bearing constructors such as `Text`, `MarkupText`, and `Tex`
- estimates rendered text length
- compares the result against prompt text length

This makes the metric fast and reproducible, and avoids adding another video pass.

## Reference Parameters

Both `PADVC_center` and `TD_center` require parameters fitted from your own reference-answer set:

- `mu`
- `sigma`
- `eps`

The repository does not ship official benchmark reference statistics. You are expected to fit them on your own curated reference subset.
