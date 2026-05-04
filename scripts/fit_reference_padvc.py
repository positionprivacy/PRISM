#!/usr/bin/env python3
import argparse
import contextlib
import io
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from padvc import PADVCCalculator


WORKER = None
WORKER_QUIET = True


def init_worker(
    ocr_cache_dir: str | None,
    device: str,
    quiet: bool,
    padvc_config: dict,
):
    global WORKER, WORKER_QUIET
    WORKER_QUIET = quiet
    WORKER = PADVCCalculator(
        device=device,
        ocr_cache_dir=ocr_cache_dir,
        score_norm_method="none",
        score_output="raw",
        **padvc_config,
    )


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_existing(path: Path):
    if not path.exists():
        return {}
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("id"):
                rows[row["id"]] = row
    return rows


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_instruction(row, instruction_fields):
    for field in instruction_fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def resolve_video_path(row, video_root: Path | None):
    if row.get("video_path"):
        path = Path(row["video_path"])
        if path.exists():
            return path
    if video_root is None:
        return None
    candidate = video_root / f"{row['id']}.mp4"
    if candidate.exists():
        return candidate
    return None


def process_one(task):
    global WORKER, WORKER_QUIET
    row = task["row"]
    instruction = task["instruction"]
    video_path = task["video_path"]
    record = {
        "id": row["id"],
        "source_id": row.get("source_id"),
        "language": row.get("language"),
        "video_path": str(video_path) if video_path else None,
    }
    try:
        if instruction is None:
            raise ValueError("instruction missing")
        if video_path is None:
            raise FileNotFoundError("video missing")
        if WORKER_QUIET:
            with contextlib.redirect_stdout(io.StringIO()):
                padvc_raw = WORKER.evaluate_single(instruction, str(video_path), return_details=False)
        else:
            padvc_raw = WORKER.evaluate_single(instruction, str(video_path), return_details=False)
        record["status"] = "ok"
        record["padvc_raw"] = float(padvc_raw)
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def summarize(records, eps: float):
    successful = [row for row in records if row.get("status") == "ok"]
    raws = [float(row["padvc_raw"]) for row in successful]
    logs = [math.log(value + eps) for value in raws]
    mu = statistics.fmean(logs)
    sigma = statistics.pstdev(logs)
    if sigma <= 0:
        sigma = 1e-12
    raw_arr = np.array(raws, dtype=float)
    log_arr = np.array(logs, dtype=float)
    return {
        "total_records": len(records),
        "success_records": len(successful),
        "failed_records": len(records) - len(successful),
        "sample_count": len(raws),
        "eps": float(eps),
        "mu": float(mu),
        "sigma": float(sigma),
        "raw_min": float(raw_arr.min()) if len(raw_arr) else 0.0,
        "raw_p5": float(np.percentile(raw_arr, 5)) if len(raw_arr) else 0.0,
        "raw_p25": float(np.percentile(raw_arr, 25)) if len(raw_arr) else 0.0,
        "raw_median": float(np.percentile(raw_arr, 50)) if len(raw_arr) else 0.0,
        "raw_p75": float(np.percentile(raw_arr, 75)) if len(raw_arr) else 0.0,
        "raw_p95": float(np.percentile(raw_arr, 95)) if len(raw_arr) else 0.0,
        "raw_max": float(raw_arr.max()) if len(raw_arr) else 0.0,
        "log_min": float(log_arr.min()) if len(log_arr) else 0.0,
        "log_p5": float(np.percentile(log_arr, 5)) if len(log_arr) else 0.0,
        "log_p25": float(np.percentile(log_arr, 25)) if len(log_arr) else 0.0,
        "log_median": float(np.percentile(log_arr, 50)) if len(log_arr) else 0.0,
        "log_p75": float(np.percentile(log_arr, 75)) if len(log_arr) else 0.0,
        "log_p95": float(np.percentile(log_arr, 95)) if len(log_arr) else 0.0,
        "log_max": float(log_arr.max()) if len(log_arr) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-root")
    parser.add_argument("--language")
    parser.add_argument("--instruction-fields", default="instruction,user_input,paired_user_input")
    parser.add_argument("--ocr-cache-dir")
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
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--quiet-padvc", action="store_true")
    args = parser.parse_args()

    dataset_jsonl = Path(args.dataset_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_root = Path(args.video_root) if args.video_root else None
    instruction_fields = [item.strip() for item in args.instruction_fields.split(",") if item.strip()]

    raw_jsonl = output_dir / "padvc_reference_raw_scores.jsonl"
    params_json = output_dir / "padvc_norm_params.json"

    rows = load_jsonl(dataset_jsonl)
    if args.language:
        rows = [row for row in rows if row.get("language") == args.language]
    if args.limit is not None:
        rows = rows[: args.limit]

    existing = load_existing(raw_jsonl) if args.skip_existing else {}
    results = {row_id: row for row_id, row in existing.items()}

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

    tasks = []
    for row in rows:
        if row["id"] in results:
            continue
        tasks.append(
            {
                "row": row,
                "instruction": resolve_instruction(row, instruction_fields),
                "video_path": resolve_video_path(row, video_root),
            }
        )

    completed = len(existing)
    print(
        f"[START] total={len(rows)} existing={len(existing)} pending={len(tasks)} "
        f"workers={args.workers}",
        flush=True,
    )

    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=init_worker,
        initargs=(
            args.ocr_cache_dir,
            args.device,
            args.quiet_padvc,
            padvc_config,
        ),
    ) as executor:
        future_map = {executor.submit(process_one, task): task["row"]["id"] for task in tasks}
        for future in as_completed(future_map):
            record = future.result()
            results[record["id"]] = record
            completed += 1
            if record["status"] == "ok":
                print(f"[{completed}/{len(rows)}] {record['id']} padvc_raw={record['padvc_raw']:.6f}", flush=True)
            else:
                print(f"[{completed}/{len(rows)}] {record['id']} failed: {record['error']}", flush=True)

    ordered = [results[row["id"]] for row in rows if row["id"] in results]
    write_jsonl(raw_jsonl, ordered)
    summary = summarize(ordered, args.eps)
    summary["padvc_config"] = padvc_config
    summary["dataset_jsonl"] = str(dataset_jsonl)
    summary["video_root"] = str(video_root) if video_root else None
    summary["language"] = args.language
    summary["instruction_fields"] = instruction_fields
    summary["ocr_cache_dir"] = args.ocr_cache_dir
    dump_json(params_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
