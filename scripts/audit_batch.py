import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from repo_config import get_tmp_subdir


TMP_ROOT = get_tmp_subdir("audit_batch")


def mean(values):
    return sum(values) / len(values) if values else 0.0


def count_segments(report):
    total_segments = 0
    if not isinstance(report, dict):
        return 0
    for scene_report in report.values():
        if isinstance(scene_report, dict):
            total_segments += len(scene_report)
    return total_segments


def summarize_audit_report(report):
    total_segments = 0
    clean_segments = 0
    issue_counts = {
        "out_of_bounds": 0,
        "overlaps": 0,
        "leaks": 0,
    }

    for scene_report in report.values():
        for segment_report in scene_report.values():
            total_segments += 1
            segment_issues = 0
            for key in issue_counts:
                count = len(segment_report.get(key, []))
                issue_counts[key] += count
                segment_issues += count
            if segment_issues == 0:
                clean_segments += 1

    total_issues = sum(issue_counts.values())
    if total_segments == 0:
        return {
            "segment_count": 0,
            "visual_structure_rate": 0.0,
            "visual_structure_pass": 0,
            "visual_issue_total": 0,
            "visual_issue_breakdown": issue_counts,
            "visual_audit_report": report,
        }

    return {
        "segment_count": total_segments,
        "visual_structure_rate": clean_segments / total_segments,
        "visual_structure_pass": 1 if total_issues == 0 else 0,
        "visual_issue_total": total_issues,
        "visual_issue_breakdown": issue_counts,
        "visual_audit_report": report,
    }


def extract_render_error(stderr_text):
    stderr_lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
    if not stderr_lines:
        return "Audit failed with no stderr"
    error_summary = stderr_lines[-1]
    if (error_summary.startswith("/") or ".log" in error_summary) and len(stderr_lines) > 1:
        for line in reversed(stderr_lines):
            if "Error:" in line:
                return line
    return error_summary


def aggregate(details):
    total = len(details)
    succeeded = [item for item in details if item.get("audit_success")]
    clean = [item for item in details if item.get("visual_structure_pass")]
    kept_no_oob = [
        item
        for item in succeeded
        if count_segments(item.get("visual_audit_report")) > 0
        and (item.get("visual_issue_breakdown") or {}).get("out_of_bounds", 0) == 0
    ]
    clean_no_oob = [
        item
        for item in kept_no_oob
        if (item.get("visual_issue_breakdown") or {}).get("out_of_bounds", 0) == 0
        and (item.get("visual_issue_breakdown") or {}).get("overlaps", 0) == 0
        and (item.get("visual_issue_breakdown") or {}).get("leaks", 0) == 0
    ]
    return {
        "total": total,
        "completed": total,
        "audit_success_count": len(succeeded),
        "audit_success_rate": len(succeeded) / total if total else 0.0,
        "visual_pass_count": len(clean),
        "visual_pass_rate": len(clean) / total if total else 0.0,
        "visual_pass_rate_among_audited": len(clean) / len(succeeded) if succeeded else 0.0,
        "visual_pass_count_no_oob": len(clean_no_oob),
        "visual_pass_rate_no_oob": len(clean_no_oob) / len(kept_no_oob) if kept_no_oob else 0.0,
        "kept_count_no_oob": len(kept_no_oob),
        "avg_visual_structure_rate": mean([item.get("visual_structure_rate", 0.0) for item in details]),
        "avg_issue_total": mean([item.get("visual_issue_total", 0) for item in details]),
        "avg_runtime_sec": mean([item.get("runtime_sec", 0.0) for item in details]),
        "issue_counts": {
            "out_of_bounds": sum((item.get("visual_issue_breakdown") or {}).get("out_of_bounds", 0) for item in details),
            "overlaps": sum((item.get("visual_issue_breakdown") or {}).get("overlaps", 0) for item in details),
            "leaks": sum((item.get("visual_issue_breakdown") or {}).get("leaks", 0) for item in details),
        },
        "error_count": sum(1 for item in details if item.get("error_msg")),
    }


def load_existing(result_path):
    if not result_path.exists():
        return {}
    result_json = json.loads(result_path.read_text(encoding="utf-8"))
    details = result_json.get("details", [])
    if not isinstance(details, list):
        return {}
    return {item["id"]: item for item in details if isinstance(item, dict) and item.get("id")}


def write_results(result_path, details, input_dir, workers, timeout_sec):
    sorted_details = sorted(details, key=lambda item: item["id"])
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": str(input_dir),
        "workers": workers,
        "timeout_sec": timeout_sec,
        "summary": aggregate(sorted_details),
        "details": sorted_details,
    }
    tmp_path = result_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(result_path)


def audit_one(auditor_script, py_path, audit_root, timeout_sec, env):
    task_id = py_path.stem
    audit_dir = audit_root / task_id
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_json_path = audit_dir / "audit_report.json"
    audit_log_path = audit_dir / "audit.log"

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(auditor_script),
                str(py_path),
                "--json-out",
                str(audit_json_path),
                "--output-dir",
                str(audit_dir),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
        runtime_sec = time.perf_counter() - start
        audit_log_path.write_text(
            proc.stdout + "\n" + "=" * 20 + " STDERR " + "=" * 20 + "\n" + proc.stderr,
            encoding="utf-8",
        )

        result = {
            "id": task_id,
            "file_path": str(py_path),
            "audit_success": 0,
            "runtime_sec": runtime_sec,
            "segment_count": 0,
            "visual_structure_rate": 0.0,
            "visual_structure_pass": 0,
            "visual_issue_total": 0,
            "visual_issue_breakdown": {"out_of_bounds": 0, "overlaps": 0, "leaks": 0},
            "error_msg": None,
        }

        if audit_json_path.exists():
            report = json.loads(audit_json_path.read_text(encoding="utf-8") or "{}")
            result.update(summarize_audit_report(report))
            result["audit_success"] = 1

        if proc.returncode != 0:
            result["error_msg"] = extract_render_error(proc.stderr)
        elif not audit_json_path.exists():
            result["error_msg"] = "Audit report not generated"

        # Empty reports or zero-segment reports are not valid successful audits.
        if result.get("audit_success"):
            report = result.get("visual_audit_report") or {}
            if count_segments(report) == 0:
                result["audit_success"] = 0
                result["segment_count"] = 0
                result["visual_structure_rate"] = 0.0
                result["visual_structure_pass"] = 0
                result["visual_issue_total"] = 0
                result["visual_issue_breakdown"] = {"out_of_bounds": 0, "overlaps": 0, "leaks": 0}
                result["error_msg"] = result.get("error_msg") or "Audit produced no valid segments"

        return result
    except subprocess.TimeoutExpired:
        runtime_sec = time.perf_counter() - start
        return {
            "id": task_id,
            "file_path": str(py_path),
            "audit_success": 0,
            "runtime_sec": runtime_sec,
            "segment_count": 0,
            "visual_structure_rate": 0.0,
            "visual_structure_pass": 0,
            "visual_issue_total": 0,
            "visual_issue_breakdown": {"out_of_bounds": 0, "overlaps": 0, "leaks": 0},
            "error_msg": f"TimeoutExpired: > {timeout_sec}s",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_root = output_dir / "audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    auditor_script = Path(__file__).with_name("audit_single.py")

    py_files = sorted(input_dir.glob("*.py"))
    existing = load_existing(result_path)
    pending = [path for path in py_files if path.stem not in existing]

    print(
        f"[START] total={len(py_files)} existing={len(existing)} pending={len(pending)} "
        f"workers={args.workers} timeout={args.timeout_sec}",
        flush=True,
    )

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(TMP_ROOT / "mplconfig"))
    env.setdefault("TMPDIR", str(TMP_ROOT / "tmp"))
    env.setdefault("TEMP", env["TMPDIR"])
    env.setdefault("TMP", env["TMPDIR"])
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    if args.no_images:
        env["AUDIT_SAVE_IMAGES"] = "0"

    completed = len(existing)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(audit_one, auditor_script, py_path, audit_root, args.timeout_sec, env): py_path.stem
            for py_path in pending
        }
        for future in as_completed(future_map):
            result = future.result()
            existing[result["id"]] = result
            completed += 1
            print(
                f"[{completed}/{len(py_files)}] {result['id']} "
                f"ok={result.get('audit_success', 0)} "
                f"pass={result.get('visual_structure_pass', 0)} "
                f"rate={result.get('visual_structure_rate', 0.0):.3f} "
                f"issues={result.get('visual_issue_total', 0)}",
                flush=True,
            )
            if completed % max(1, args.save_interval) == 0:
                write_results(result_path, list(existing.values()), input_dir, args.workers, args.timeout_sec)

    write_results(result_path, list(existing.values()), input_dir, args.workers, args.timeout_sec)
    print(f"[DONE] {result_path}", flush=True)


if __name__ == "__main__":
    main()
