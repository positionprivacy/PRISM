import argparse
import contextlib
import io
import json
import math
from pathlib import Path

from padvc import PADVCCalculator

DETAIL_VERSION = 4
REQUIRED_OK_FIELDS = (
    "detail_version",
    "p_exponent",
    "video_fps",
    "video_frame_count",
    "video_duration_sec",
    "text_sample_stride",
    "text_peak_frame_index",
    "sampled_text_energy_series",
    "frame_diff_series",
    "frame_diff_count",
    "event_threshold_mode",
    "event_threshold_abs",
    "event_threshold_ratio",
    "delta_mode",
    "event_energy_details",
)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_existing(path: Path):
    if not path.exists():
        return {}
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                print(
                    f"[warn] skip bad existing line file={path} line={line_no} err={type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            if row.get("id"):
                rows[row["id"]] = row
    return rows


def row_has_full_details(row):
    if not isinstance(row, dict):
        return False
    status = row.get("status")
    if status != "ok":
        return bool(status)
    if int(row.get("detail_version") or 0) < DETAIL_VERSION:
        return False
    return all(field in row for field in REQUIRED_OK_FIELDS)


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_task_map(task_manifest_path: Path):
    rows = load_json(task_manifest_path)
    return {row["id"]: row for row in rows}


def build_audit_map(audit_results_path: Path):
    payload = load_json(audit_results_path)
    return {row["id"]: row for row in payload["details"]}


def build_render_map(render_results_path: Path):
    payload = load_json(render_results_path)
    return {row["id"]: row for row in payload["details"]}


def build_sample_map(sample_jsonl_path: Path | None):
    if sample_jsonl_path is None:
        return {}
    rows = load_jsonl(sample_jsonl_path)
    result = {}
    for row in rows:
        if row.get("id"):
            result[row["id"]] = row
        if row.get("source_id"):
            result[row["source_id"]] = row
    return result


def compute_padvc_center(raw_value: float, mu: float, sigma: float, eps: float):
    value = max(float(raw_value), 0.0)
    log_value = math.log(value + eps)
    z = (log_value - mu) / max(float(sigma), 1e-12)
    return math.exp(-0.5 * z * z)


def aggregate(records):
    ok = [row for row in records if row.get("status") == "ok"]
    render_ok = [row for row in records if row.get("render_success")]
    visual_ok = [row for row in ok if row.get("visual_structure_pass") == 1]
    by_diff = {}
    for diff in ("easy", "medium", "hard"):
        subset = [row for row in ok if row.get("difficulty") == diff]
        by_diff[diff] = {
            "count": len(subset),
            "avg_padvc_raw": mean([row.get("padvc_raw", 0.0) for row in subset]),
            "avg_padvc_center": mean([row.get("padvc_center", 0.0) for row in subset]),
            "avg_u_padvc": mean([row.get("u_padvc", 0.0) for row in subset]),
            "avg_visual_structure_pass": mean([1 if row.get("visual_structure_pass") == 1 else 0 for row in subset]),
            "avg_visual_structure_rate": mean([row.get("visual_structure_rate", 0.0) for row in subset]),
        }
    return {
        "detail_version": DETAIL_VERSION,
        "formula": {
            "padvc_center": "exp(-0.5 * ((log(padvc_raw + eps) - mu) / sigma)^2)",
            "u_padvc": "padvc_raw * visual_structure_pass",
        },
        "total_records": len(records),
        "render_success_count": len(render_ok),
        "scored_count": len(ok),
        "visual_pass_count": len(visual_ok),
        "avg_padvc_raw": mean([row.get("padvc_raw", 0.0) for row in ok]),
        "avg_padvc_center": mean([row.get("padvc_center", 0.0) for row in ok]),
        "avg_u_padvc": mean([row.get("u_padvc", 0.0) for row in ok]),
        "avg_visual_structure_pass_scored": mean([1 if row.get("visual_structure_pass") == 1 else 0 for row in ok]),
        "avg_padvc_raw_visual_pass": mean([row.get("padvc_raw", 0.0) for row in visual_ok]),
        "avg_padvc_center_visual_pass": mean([row.get("padvc_center", 0.0) for row in visual_ok]),
        "avg_u_padvc_visual_pass": mean([row.get("u_padvc", 0.0) for row in visual_ok]),
        "avg_visual_structure_rate_scored": mean([row.get("visual_structure_rate", 0.0) for row in ok]),
        "by_difficulty": by_diff,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--audit-results", required=True)
    parser.add_argument("--render-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-jsonl")
    parser.add_argument("--ocr-cache-dir")
    parser.add_argument("--norm-params", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--p", type=float, default=0.7)
    parser.add_argument("--event-threshold-mode", default="absolute")
    parser.add_argument("--event-threshold-abs", type=float, default=50000.0)
    parser.add_argument("--event-threshold-ratio", type=float, default=0.08)
    parser.add_argument("--delta-mode", choices=["absolute", "positive"], default="positive")
    parser.add_argument("--text-dilate", type=int, default=7)
    parser.add_argument("--ocr-backend", choices=["paddle", "rapidocr"], default="rapidocr")
    parser.add_argument("--rapidocr-package-path")
    parser.add_argument("--rapid-use-cls", action="store_true")
    parser.add_argument("--rapid-no-rec", action="store_true")
    parser.add_argument("--rapid-text-score", type=float, default=0.5)
    parser.add_argument("--rapid-box-thresh", type=float, default=0.5)
    parser.add_argument("--rapid-unclip-ratio", type=float, default=1.6)
    parser.add_argument("--rapid-det-limit-side-len", type=int, default=736)
    parser.add_argument("--sticky-peak-rescue", action="store_true")
    parser.add_argument("--sticky-primary-above-ratio", type=float, default=0.95)
    parser.add_argument("--sticky-secondary-above-ratio", type=float, default=0.98)
    parser.add_argument("--sticky-primary-event-max", type=int, default=2)
    parser.add_argument("--sticky-secondary-event-max", type=int, default=3)
    parser.add_argument("--sticky-peak-smooth-window", type=int, default=3)
    parser.add_argument("--sticky-peak-quantile", type=float, default=0.75)
    parser.add_argument("--sticky-peak-min-rel-height", type=float, default=0.14)
    parser.add_argument("--sticky-peak-merge-gap", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--quiet-padvc", action="store_true")
    args = parser.parse_args()

    task_map = build_task_map(Path(args.task_manifest))
    audit_map = build_audit_map(Path(args.audit_results))
    render_map = build_render_map(Path(args.render_results))
    sample_map = build_sample_map(Path(args.sample_jsonl)) if args.sample_jsonl else {}
    norm_params = load_json(Path(args.norm_params))
    center_mu = float(norm_params["mu"])
    center_sigma = float(norm_params["sigma"])
    center_eps = float(norm_params.get("eps", 1e-8))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = output_dir / "padvc_scores.jsonl"
    summary_json = output_dir / "summary.json"

    existing_all = load_existing(results_jsonl) if args.skip_existing else {}
    existing_complete = {sample_id: row for sample_id, row in existing_all.items() if row_has_full_details(row)}
    existing_incomplete = {sample_id: row for sample_id, row in existing_all.items() if sample_id not in existing_complete}

    padvc_config = {
        "p": args.p,
        "event_threshold_mode": args.event_threshold_mode,
        "event_threshold_abs": args.event_threshold_abs,
        "event_threshold_ratio": args.event_threshold_ratio,
        "delta_mode": args.delta_mode,
        "text_dilate": args.text_dilate,
        "ocr_backend": args.ocr_backend,
        "rapidocr_package_path": args.rapidocr_package_path,
        "rapid_use_cls": args.rapid_use_cls,
        "rapid_use_rec": not args.rapid_no_rec,
        "rapid_text_score": args.rapid_text_score,
        "rapid_box_thresh": args.rapid_box_thresh,
        "rapid_unclip_ratio": args.rapid_unclip_ratio,
        "rapid_det_limit_side_len": args.rapid_det_limit_side_len,
        "sticky_peak_rescue": args.sticky_peak_rescue,
        "sticky_primary_above_ratio": args.sticky_primary_above_ratio,
        "sticky_secondary_above_ratio": args.sticky_secondary_above_ratio,
        "sticky_primary_event_max": args.sticky_primary_event_max,
        "sticky_secondary_event_max": args.sticky_secondary_event_max,
        "sticky_peak_smooth_window": args.sticky_peak_smooth_window,
        "sticky_peak_quantile": args.sticky_peak_quantile,
        "sticky_peak_min_rel_height": args.sticky_peak_min_rel_height,
        "sticky_peak_merge_gap": args.sticky_peak_merge_gap,
    }
    calc = PADVCCalculator(
        device=args.device,
        score_norm_method="none",
        score_output="raw",
        ocr_cache_dir=args.ocr_cache_dir or str(output_dir / "ocr_cache"),
        **padvc_config,
    )

    ordered_ids = sorted(audit_map.keys())
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index out of range")
    ordered_ids = [sample_id for idx, sample_id in enumerate(ordered_ids) if idx % args.num_shards == args.shard_index]
    if args.limit is not None:
        ordered_ids = ordered_ids[: args.limit]

    records = []
    with results_jsonl.open("w", encoding="utf-8") as writer:
        for sample_id in sorted(existing_complete):
            writer.write(json.dumps(existing_complete[sample_id], ensure_ascii=False) + "\n")
        writer.flush()

        if args.skip_existing:
            print(
                f"[resume] keep_complete={len(existing_complete)} recompute_or_fill={len(existing_incomplete)} total_existing={len(existing_all)}",
                flush=True,
            )

        for idx, sample_id in enumerate(ordered_ids, start=1):
            if sample_id in existing_complete:
                records.append(existing_complete[sample_id])
                print(f"[{idx}/{len(ordered_ids)}] skip {sample_id}", flush=True)
                continue

            audit = audit_map[sample_id]
            task = task_map.get(sample_id, {})
            render = render_map.get(sample_id, {})
            sample = sample_map.get(task.get("source_id")) or sample_map.get(sample_id) or {}
            row = {
                "detail_version": DETAIL_VERSION,
                "id": sample_id,
                "source_id": task.get("source_id"),
                "difficulty": sample.get("difficulty"),
                "difficulty_score": sample.get("difficulty_score"),
                "audit_success": audit.get("audit_success"),
                "visual_structure_pass": audit.get("visual_structure_pass"),
                "visual_structure_rate": audit.get("visual_structure_rate"),
                "render_success": render.get("success"),
                "video_path": render.get("video_path"),
            }
            try:
                if audit.get("audit_success") != 1:
                    raise RuntimeError("audit_success != 1")
                if not render.get("success") or not render.get("video_path"):
                    raise RuntimeError("render video missing")
                instruction = task.get("instruction_text")
                if not instruction:
                    raise RuntimeError("instruction_text missing")
                if args.quiet_padvc:
                    with contextlib.redirect_stdout(io.StringIO()):
                        payload = calc.evaluate_single(instruction, render["video_path"], return_details=True)
                else:
                    payload = calc.evaluate_single(instruction, render["video_path"], return_details=True)
                row.update(payload)
                row["padvc_center"] = float(compute_padvc_center(row["padvc_raw"], center_mu, center_sigma, center_eps))
                row["u_padvc"] = float(row["padvc_raw"] * float(int(row.get("visual_structure_pass", 0) or 0)))
                row["padvc_norm"] = row["padvc_center"]
                row["padvc_norm_name"] = "padvc_center"
                row["status"] = "ok"
                print(
                    f"[{idx}/{len(ordered_ids)}] {sample_id} raw={row['padvc_raw']:.6f} center={row['padvc_center']:.6f} u={row['u_padvc']:.6f}",
                    flush=True,
                )
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{idx}/{len(ordered_ids)}] {sample_id} failed: {row['error']}", flush=True)
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            writer.flush()
            records.append(row)

    summary = aggregate(records)
    summary["norm_params"] = str(Path(args.norm_params))
    summary["center_mu"] = center_mu
    summary["center_sigma"] = center_sigma
    summary["center_eps"] = center_eps
    summary["padvc_config"] = padvc_config
    summary["task_manifest"] = args.task_manifest
    summary["audit_results"] = args.audit_results
    summary["render_results"] = args.render_results
    summary["num_shards"] = args.num_shards
    summary["shard_index"] = args.shard_index
    dump_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
