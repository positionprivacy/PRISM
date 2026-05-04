#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(values):
    return sum(values) / len(values) if values else 0.0


def merge_summaries(summaries):
    if not summaries:
        return {}
    by_diff = {}
    for diff in ("easy", "medium", "hard"):
        diff_rows = [s.get("by_difficulty", {}).get(diff, {}) for s in summaries]
        counts = [row.get("count", 0) for row in diff_rows]
        by_diff[diff] = {
            "count": sum(counts),
            "avg_padvc_raw": mean([row.get("avg_padvc_raw", 0.0) for row in diff_rows if row]),
            "avg_padvc_center": mean([row.get("avg_padvc_center", 0.0) for row in diff_rows if row]),
            "avg_u_padvc": mean([row.get("avg_u_padvc", 0.0) for row in diff_rows if row]),
            "avg_visual_structure_pass": mean([row.get("avg_visual_structure_pass", 0.0) for row in diff_rows if row]),
            "avg_visual_structure_rate": mean([row.get("avg_visual_structure_rate", 0.0) for row in diff_rows if row]),
        }
    return {
        "detail_version": max(int(s.get("detail_version", 0)) for s in summaries),
        "formula": summaries[0].get("formula", {}),
        "total_records": sum(int(s.get("total_records", 0)) for s in summaries),
        "render_success_count": sum(int(s.get("render_success_count", 0)) for s in summaries),
        "scored_count": sum(int(s.get("scored_count", 0)) for s in summaries),
        "visual_pass_count": sum(int(s.get("visual_pass_count", 0)) for s in summaries),
        "avg_padvc_raw": mean([float(s.get("avg_padvc_raw", 0.0)) for s in summaries]),
        "avg_padvc_center": mean([float(s.get("avg_padvc_center", 0.0)) for s in summaries]),
        "avg_u_padvc": mean([float(s.get("avg_u_padvc", 0.0)) for s in summaries]),
        "avg_visual_structure_pass_scored": mean([float(s.get("avg_visual_structure_pass_scored", 0.0)) for s in summaries]),
        "avg_padvc_raw_visual_pass": mean([float(s.get("avg_padvc_raw_visual_pass", 0.0)) for s in summaries]),
        "avg_padvc_center_visual_pass": mean([float(s.get("avg_padvc_center_visual_pass", 0.0)) for s in summaries]),
        "avg_u_padvc_visual_pass": mean([float(s.get("avg_u_padvc_visual_pass", 0.0)) for s in summaries]),
        "avg_visual_structure_rate_scored": mean([float(s.get("avg_visual_structure_rate_scored", 0.0)) for s in summaries]),
        "by_difficulty": by_diff,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []
    for shard_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        jsonl_path = shard_dir / "padvc_scores.jsonl"
        if jsonl_path.exists():
            all_rows.extend(load_jsonl(jsonl_path))
        summary_path = shard_dir / "summary.json"
        if summary_path.exists():
            summaries.append(load_json(summary_path))

    all_rows.sort(key=lambda row: row.get("id", ""))
    write_jsonl(output_dir / "padvc_scores.jsonl", all_rows)
    dump_json(output_dir / "summary.json", merge_summaries(summaries))
    print(output_dir / "padvc_scores.jsonl")
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
