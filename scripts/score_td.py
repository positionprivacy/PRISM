import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def compute_td_raw(video_path: str, pixel_threshold: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    diff_duration = (total_frames - 1) / fps if fps > 0 and total_frames > 1 else 0.0
    if diff_duration <= 0:
        cap.release()
        return 0.0

    ok, prev_frame = cap.read()
    if not ok:
        cap.release()
        return 0.0

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    total_variation = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        non_zero_ratio = float(np.count_nonzero(diff > pixel_threshold) / diff.size)
        total_variation += non_zero_ratio
        prev_gray = gray

    cap.release()
    return float(total_variation / diff_duration)


def build_row(base_row, td_raw, mu, sigma, eps):
    td_center_z = (math.log(float(td_raw) + eps) - mu) / sigma
    td_center = math.exp(-0.5 * td_center_z * td_center_z)
    visual_rate = float(base_row.get("visual_structure_rate", 0.0) or 0.0)

    row = {
        "id": base_row.get("id"),
        "source_id": base_row.get("source_id"),
        "difficulty": base_row.get("difficulty"),
        "difficulty_score": base_row.get("difficulty_score"),
        "render_success": bool(base_row.get("render_success")),
        "visual_structure_pass": int(base_row.get("visual_structure_pass", 0) or 0),
        "visual_structure_rate": visual_rate,
        "video_path": base_row.get("video_path"),
        "td_raw": float(td_raw),
        "td_center": float(td_center),
        "td_center_z": float(td_center_z),
        "u_td": float(td_center * visual_rate),
    }

    for key in ("padvc_raw", "padvc_center", "u_padvc"):
        if key in base_row:
            row[key] = base_row[key]
    return row


def summarize_subset(rows):
    return {
        "count": len(rows),
        "mean_td_raw": mean([row.get("td_raw", 0.0) for row in rows]),
        "median_td_raw": median([row.get("td_raw", 0.0) for row in rows]),
        "mean_td_center": mean([row.get("td_center", 0.0) for row in rows]),
        "median_td_center": median([row.get("td_center", 0.0) for row in rows]),
        "mean_u_td": mean([row.get("u_td", 0.0) for row in rows]),
        "median_u_td": median([row.get("u_td", 0.0) for row in rows]),
        "mean_visual_rate": mean([row.get("visual_structure_rate", 0.0) for row in rows]),
    }


def median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def summarize_bins(rows):
    bins = []
    for idx in range(len(BIN_EDGES) - 1):
        lo = BIN_EDGES[idx]
        hi = BIN_EDGES[idx + 1]
        if idx == len(BIN_EDGES) - 2:
            subset = [row for row in rows if lo <= row.get("td_center", 0.0) <= hi]
            label = [lo, hi]
        else:
            subset = [row for row in rows if lo <= row.get("td_center", 0.0) < hi]
            label = [lo, hi]
        bins.append(
            {
                "range": label,
                "count": len(subset),
                "visual_pass_rate": mean([row.get("visual_structure_pass", 0) for row in subset]),
                "mean_td_raw": mean([row.get("td_raw", 0.0) for row in subset]),
                "mean_u_td": mean([row.get("u_td", 0.0) for row in subset]),
            }
        )
    return bins


def make_summary(rows, mu, sigma, eps, pixel_threshold, input_jsonl: str, output_jsonl: str):
    rendered = [row for row in rows if row.get("render_success")]
    visual_pass = [row for row in rows if row.get("visual_structure_pass") == 1]
    visual_fail = [row for row in rows if row.get("visual_structure_pass") != 1]
    by_difficulty = {}
    for diff in ("easy", "medium", "hard"):
        subset = [row for row in rows if row.get("difficulty") == diff]
        by_difficulty[diff] = summarize_subset(subset)

    return {
        "params": {
            "mu": mu,
            "sigma": sigma,
            "eps": eps,
            "threshold": pixel_threshold,
        },
        "source_jsonl": input_jsonl,
        "output_jsonl": output_jsonl,
        "overall": summarize_subset(rows),
        "render_success": summarize_subset(rendered),
        "visual_pass": summarize_subset(visual_pass),
        "visual_fail": summarize_subset(visual_fail),
        "by_difficulty": by_difficulty,
        "td_center_bins": summarize_bins(rows),
    }


def process_one(base_row, pixel_threshold, mu, sigma, eps):
    render_success = bool(base_row.get("render_success"))
    video_path = base_row.get("video_path")
    td_raw = 0.0
    if render_success and video_path and Path(video_path).exists():
        td_raw = compute_td_raw(video_path, pixel_threshold)
    return build_row(base_row, td_raw, mu, sigma, eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--pixel-threshold", type=int, default=25)
    parser.add_argument("--params-json")
    parser.add_argument("--mu", type=float, default=-3.6128059890794395)
    parser.add_argument("--sigma", type=float, default=0.5952279192727674)
    parser.add_argument("--eps", type=float, default=9.807249641469183e-05)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.params_json:
        params = load_json(Path(args.params_json))
        args.mu = float(params["mu"])
        args.sigma = float(params["sigma"])
        args.eps = float(params.get("eps", args.eps))

    input_jsonl = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "td_center_scores.jsonl"
    summary_json = output_dir / "summary.json"

    base_rows = load_jsonl(input_jsonl)
    existing = load_existing(output_jsonl) if args.skip_existing else {}
    results = {row["id"]: row for row in existing.values()}
    pending = [row for row in base_rows if row.get("id") not in results]

    print(
        f"[START] total={len(base_rows)} existing={len(existing)} pending={len(pending)} "
        f"workers={args.workers} threshold={args.pixel_threshold}",
        flush=True,
    )

    completed = len(existing)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(process_one, row, args.pixel_threshold, args.mu, args.sigma, args.eps): row.get("id")
            for row in pending
        }
        for future in as_completed(future_map):
            row = future.result()
            results[row["id"]] = row
            completed += 1
            print(
                f"[{completed}/{len(base_rows)}] {row['id']} "
                f"td_raw={row['td_raw']:.6f} td_center={row['td_center']:.6f} u_td={row['u_td']:.6f}",
                flush=True,
            )
            if completed % max(1, args.save_interval) == 0:
                ordered = [results[row["id"]] for row in base_rows if row.get("id") in results]
                write_jsonl(output_jsonl, ordered)
                dump_json(
                    summary_json,
                    make_summary(
                        ordered,
                        args.mu,
                        args.sigma,
                        args.eps,
                        args.pixel_threshold,
                        str(input_jsonl),
                        str(output_jsonl),
                    ),
                )

    ordered = [results[row["id"]] for row in base_rows if row.get("id") in results]
    write_jsonl(output_jsonl, ordered)
    dump_json(
        summary_json,
        make_summary(
            ordered,
            args.mu,
            args.sigma,
            args.eps,
            args.pixel_threshold,
            str(input_jsonl),
            str(output_jsonl),
        ),
    )
    print(f"[DONE] {summary_json}", flush=True)


if __name__ == "__main__":
    main()
