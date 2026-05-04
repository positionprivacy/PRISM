# Spatial Audit

The spatial auditor checks whether a rendered Manim scene remains visually usable at stable checkpoints.

## What It Detects

The public auditor focuses on three error types:

- `overlap`: unintended geometric overlap that harms readability
- `out_of_bounds`: content leaves the visible frame
- `leak`: local structural spillover, such as text escaping a surrounding container

## How It Works

The auditor operates on Manim scene structure rather than on a generic VLM:

1. It intercepts scene construction and stable visual checkpoints.
2. It extracts object-level geometry from the rendered scene.
3. It applies heuristic exemptions for benign overlap patterns.
4. It writes segment-level JSON reports plus an aggregate result.

The goal is not aesthetic scoring. The goal is deterministic structure checking.

## Output Semantics

For each sample, the batch auditor reports:

- whether auditing succeeded
- whether the sample passes the strict binary visual check
- per-segment error counts
- aggregate rates such as visual structure rate

`visual_structure_pass = 1` means all audited segments pass all enabled checks.

## Scope

The auditor is intentionally strict. It is best used as a structural usability filter, not as a full teaching-quality judge.
