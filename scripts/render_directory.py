import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

from repo_config import get_tmp_subdir


def get_scene_class(file_path: Path):
    content = file_path.read_text(encoding="utf-8")
    match = re.search(
        r"class\s+(\w+)\((?:Scene|MovingCameraScene|ThreeDScene|VectorScene|ZoomedScene|LinearTransformationScene)",
        content,
    )
    return match.group(1) if match else None


def find_final_mp4(media_dir: Path, output_name: str):
    exact = sorted(path for path in media_dir.rglob(output_name) if "partial_movie_files" not in path.parts)
    if exact:
        return exact[0]
    matches = [path for path in media_dir.rglob("*.mp4") if "partial_movie_files" not in path.parts]
    return sorted(matches)[0] if matches else None


def render_one(py_file: Path, output_dir: Path, manim_bin: str, timeout_sec: int | None):
    sample_id = py_file.stem
    dest_video = output_dir / f"{sample_id}.mp4"
    if dest_video.exists():
        return {
            "id": sample_id,
            "file_path": str(py_file),
            "success": True,
            "skipped": True,
            "video_path": str(dest_video),
            "runtime_sec": 0.0,
            "error": None,
        }

    scene_class = get_scene_class(py_file)
    if not scene_class:
        return {
            "id": sample_id,
            "file_path": str(py_file),
            "success": False,
            "skipped": False,
            "video_path": None,
            "runtime_sec": 0.0,
            "error": "Scene class not found",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = get_tmp_subdir("render")
    with tempfile.TemporaryDirectory(prefix=f"render_{sample_id[:32]}_", dir=str(tmp_root)) as tmp_dir:
        media_dir = Path(tmp_dir) / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            manim_bin,
            "-ql",
            str(py_file),
            scene_class,
            "--media_dir",
            str(media_dir),
            "-o",
            f"{sample_id}.mp4",
            "--progress_bar",
            "none",
        ]
        started = perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec if timeout_sec and timeout_sec > 0 else None,
            )
        except subprocess.TimeoutExpired as exc:
            runtime_sec = perf_counter() - started
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            error = (stderr or stdout or f"render timed out after {timeout_sec}s").strip()
            return {
                "id": sample_id,
                "file_path": str(py_file),
                "success": False,
                "skipped": False,
                "video_path": None,
                "runtime_sec": runtime_sec,
                "error": f"timeout after {timeout_sec}s: {error[-3500:]}",
            }

        runtime_sec = perf_counter() - started
        if proc.returncode != 0:
            error = (proc.stderr or proc.stdout or "render failed").strip()
            return {
                "id": sample_id,
                "file_path": str(py_file),
                "success": False,
                "skipped": False,
                "video_path": None,
                "runtime_sec": runtime_sec,
                "error": error[-4000:],
            }

        final_mp4 = find_final_mp4(media_dir, f"{sample_id}.mp4")
        if final_mp4 is None:
            return {
                "id": sample_id,
                "file_path": str(py_file),
                "success": False,
                "skipped": False,
                "video_path": None,
                "runtime_sec": runtime_sec,
                "error": "render succeeded but final mp4 not found",
            }

        shutil.move(str(final_mp4), str(dest_video))
        return {
            "id": sample_id,
            "file_path": str(py_file),
            "success": True,
            "skipped": False,
            "video_path": str(dest_video),
            "runtime_sec": runtime_sec,
            "error": None,
        }


def load_existing(results_json: Path):
    if not results_json.exists():
        return {}
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    rows = payload.get("details", []) if isinstance(payload, dict) else []
    return {row["id"]: row for row in rows if isinstance(row, dict) and row.get("id")}


def write_results(results_json: Path, input_dir: Path, output_dir: Path, workers: int, timeout_sec: int | None, rows: list[dict]):
    ordered = sorted(rows, key=lambda row: row["id"])
    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "workers": workers,
        "timeout_sec": timeout_sec,
        "summary": {
            "total": len(ordered),
            "success_count": sum(1 for row in ordered if row.get("success")),
            "skip_count": sum(1 for row in ordered if row.get("skipped")),
            "failure_count": sum(1 for row in ordered if not row.get("success")),
            "avg_runtime_sec": (
                sum(float(row.get("runtime_sec", 0.0)) for row in ordered) / len(ordered) if ordered else 0.0
            ),
        },
        "details": ordered,
    }
    results_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = results_json.with_suffix(results_json.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, results_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--manim-bin", default=os.environ.get("MANIM_BIN", "manim"))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    results_json = Path(args.results_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    py_files = sorted(input_dir.glob("*.py"))
    existing = load_existing(results_json)
    rows = list(existing.values())
    done_ids = set(existing)
    pending = [path for path in py_files if path.stem not in done_ids]

    print(
        f"[START] total={len(py_files)} existing={len(existing)} pending={len(pending)} workers={args.workers} timeout_sec={args.timeout_sec} manim_bin={args.manim_bin}",
        flush=True,
    )
    write_results(results_json, input_dir, output_dir, args.workers, args.timeout_sec, rows)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(render_one, path, output_dir, args.manim_bin, args.timeout_sec): path.stem
            for path in pending
        }
        completed = len(rows)
        for future in as_completed(future_map):
            item = future.result()
            rows.append(item)
            completed += 1
            if completed % max(1, args.save_interval) == 0 or completed == len(py_files):
                write_results(results_json, input_dir, output_dir, args.workers, args.timeout_sec, rows)
            status = "ok" if item.get("success") else "fail"
            extra = " skipped=1" if item.get("skipped") else ""
            print(
                f"[{completed}/{len(py_files)}] {item['id']} {status}{extra} time={item.get('runtime_sec', 0.0):.2f}s",
                flush=True,
            )

    write_results(results_json, input_dir, output_dir, args.workers, args.timeout_sec, rows)
    print(f"[DONE] {results_json}", flush=True)


if __name__ == "__main__":
    main()
