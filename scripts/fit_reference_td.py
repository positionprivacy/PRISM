#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from score_td import compute_td_raw


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


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def process_one(row, pixel_threshold: int, video_root: Path | None):
    sample_id = row["id"]
    video_path = resolve_video_path(row, video_root)
    record = {
        "id": sample_id,
        "source_id": row.get("source_id"),
        "language": row.get("language"),
        "video_path": str(video_path) if video_path else None,
    }
    try:
        if video_path is None:
            raise FileNotFoundError("video missing")
        td_raw = compute_td_raw(str(video_path), pixel_threshold)
        record["status"] = "ok"
        record["td_raw"] = float(td_raw)
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def summarize(rows):
    successful = [row for row in rows if row.get("status") == "ok"]
    raws = [float(row["td_raw"]) for row in successful]
    positives = sorted(value for value in raws if value > 0)
    eps = float(positives[0]) if positives else 1e-8
    logs = [math.log(value + eps) for value in raws]
    mu = statistics.fmean(logs)
    sigma = statistics.pstdev(logs)
    if sigma <= 0:
        sigma = 1e-12

    raw_arr = np.array(raws, dtype=float)
    log_arr = np.array(logs, dtype=float)
    return {
        "total_records": len(rows),
        "success_records": len(successful),
        "failed_records": len(rows) - len(successful),
        "sample_count": len(raws),
        "threshold": None,
        "eps": eps,
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
    parser.add_argument("--pixel-threshold", type=int, default=25)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    dataset_jsonl = Path(args.dataset_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_root = Path(args.video_root) if args.video_root else None

    raw_jsonl = output_dir / "td_reference_raw_scores.jsonl"
    params_json = output_dir / "td_center_params.json"

    rows = load_jsonl(dataset_jsonl)
    if args.language:
        rows = [row for row in rows if row.get("language") == args.language]
    if args.limit is not None:
        rows = rows[: args.limit]

    existing = load_existing(raw_jsonl) if args.skip_existing else {}
    results = {row_id: row for row_id, row in existing.items()}
    pending = [row for row in rows if row["id"] not in results]

    completed = len(existing)
    print(
        f"[START] total={len(rows)} existing={len(existing)} pending={len(pending)} "
        f"workers={args.workers} threshold={args.pixel_threshold}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(process_one, row, args.pixel_threshold, video_root): row["id"]
            for row in pending
        }
        for future in as_completed(future_map):
            record = future.result()
            results[record["id"]] = record
            completed += 1
            if record["status"] == "ok":
                print(f"[{completed}/{len(rows)}] {record['id']} td_raw={record['td_raw']:.6f}", flush=True)
            else:
                print(f"[{completed}/{len(rows)}] {record['id']} failed: {record['error']}", flush=True)

    ordered = [results[row["id"]] for row in rows if row["id"] in results]
    write_jsonl(raw_jsonl, ordered)
    summary = summarize(ordered)
    summary["threshold"] = args.pixel_threshold
    summary["dataset_jsonl"] = str(dataset_jsonl)
    summary["video_root"] = str(video_root) if video_root else None
    summary["language"] = args.language
    write_json(params_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
