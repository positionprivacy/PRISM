import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from manim_bench.llm_call.llm_call import LLMAPIClient
from repo_config import LLM_CONFIG_PATH, RESULTS_DIR, get_markdown_roots


DEFAULT_CONFIG = LLM_CONFIG_PATH
DEFAULT_OUTPUT_ROOT = RESULTS_DIR
DEFAULT_MD_ROOTS = get_markdown_roots()

INSTRUCTION_TEMPLATE = (
    "你是 Manim 专家，请将 Markdown 讲义转换为一段中文展示的 Manim CE v0.19.0 可运行代码。"
    "要求页面排版逻辑清晰、无元素遮挡，且严禁输出任何代码以外的说明或解释文本。"
    "代码需特别注意规避 LaTeX 语法错误、索引越界等常见报错，确保直接运行。"
    "Markdown 讲义如下：\n\n{markdown}"
)

THINKING_PATTERNS = (
    r"<think(?:ing)?[^>]*>.*?</think(?:ing)?>",
    r"^\s*(?:思考过程|推理过程|Reasoning|Thought process)\s*[:：].*$",
)

CODE_START_PATTERNS = (
    "from manim import",
    "import manim",
    "from manim import *",
    "class ",
)


def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sample"


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def mean_field(rows, key):
    values = [float(item.get(key, 0.0)) for item in rows if item.get(key) is not None]
    return sum(values) / len(values) if values else 0.0


def sha256_text(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest_items(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("details"), list):
            return payload["details"]
        if isinstance(payload.get("results"), list):
            return payload["results"]
    raise ValueError(f"Unsupported manifest format: {path}")


def load_jsonl_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_md_path(sample_id: str, md_roots: list[Path]):
    for root in md_roots:
        candidate = root / f"{sample_id}.md"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def clean_model_output(raw_output: str):
    text = (raw_output or "").strip()
    if not text:
        return ""
    if text == LLMAPIClient.BUSY_MESSAGE:
        return ""

    for pattern in THINKING_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    code_blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if code_blocks:
        text = max(code_blocks, key=len).strip()
    else:
        text = re.sub(r"```(?:python|py)?|```", "", text, flags=re.IGNORECASE).strip()

    lines = text.splitlines()
    start_idx = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped.startswith("from manim import")
            or stripped.startswith("import manim")
            or stripped.startswith("class ")
        ):
            start_idx = idx
            break
    return "\n".join(lines[start_idx:]).strip()


def looks_like_code(text: str):
    stripped = (text or "").lstrip()
    return any(stripped.startswith(prefix) for prefix in CODE_START_PATTERNS)


def build_benchmark_instruction(markdown_text: str):
    return INSTRUCTION_TEMPLATE.format(markdown=markdown_text.strip())


def inject_font_defaults(code: str):
    if "from manim import *" in code:
        return code.replace(
            "from manim import *",
            "from manim import *\nfrom manim import config\nText.set_default(font='SimHei')",
            1,
        )
    if "import manim" in code:
        return code.replace(
            "import manim",
            "import manim\nfrom manim import config\nmanim.Text.set_default(font='SimHei')",
            1,
        )
    return code


def build_tasks(manifest_paths: list[Path], md_roots: list[Path], limit: int | None, shuffle: bool, seed: int):
    dedup = {}
    missing_md = []
    skipped_not_pass = 0

    for manifest_path in manifest_paths:
        for item in load_manifest_items(manifest_path):
            if item.get("audit_success", 1) != 1:
                skipped_not_pass += 1
                continue
            if item.get("visual_structure_pass", 1) != 1:
                skipped_not_pass += 1
                continue

            sample_id = item.get("id")
            if not sample_id:
                continue
            md_path = resolve_md_path(sample_id, md_roots)
            if not md_path:
                missing_md.append(sample_id)
                continue
            if sample_id in dedup:
                continue

            markdown_text = md_path.read_text(encoding="utf-8")
            instruction = build_benchmark_instruction(markdown_text)

            dedup[sample_id] = {
                "id": sample_id,
                "md_path": str(md_path),
                "markdown_sha256": sha256_text(markdown_text),
                "markdown_chars": len(markdown_text),
                "instruction_sha256": sha256_text(instruction),
                "instruction_chars": len(instruction),
                "manifest_path": str(manifest_path),
                "source_file_path": item.get("file_path"),
            }

    tasks = list(dedup.values())
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(tasks)
    else:
        tasks.sort(key=lambda item: item["id"])

    if limit is not None:
        tasks = tasks[:limit]

    return tasks, {
        "manifest_count": len(manifest_paths),
        "selected_count": len(tasks),
        "dedup_count_before_limit": len(dedup),
        "skipped_not_pass": skipped_not_pass,
        "missing_md_count": len(missing_md),
        "missing_md_examples": missing_md[:20],
    }


def build_tasks_from_jsonl(
    jsonl_paths: list[Path],
    limit: int | None,
    shuffle: bool,
    seed: int,
    instruction_field: str,
):
    dedup = {}
    missing_instruction = []

    for jsonl_path in jsonl_paths:
        dataset_tag = safe_name(jsonl_path.stem)
        for item in load_jsonl_rows(jsonl_path):
            source_id = item.get("id")
            instruction = (item.get(instruction_field) or "").strip()
            if not source_id or not instruction:
                missing_instruction.append(source_id or f"<missing-id>@{jsonl_path.name}")
                continue
            sample_id = f"{dataset_tag}__{source_id}"
            if sample_id in dedup:
                continue

            dedup[sample_id] = {
                "id": sample_id,
                "source_id": source_id,
                "dataset_tag": dataset_tag,
                "md_path": None,
                "markdown_sha256": None,
                "markdown_chars": None,
                "instruction_sha256": sha256_text(instruction),
                "instruction_chars": len(instruction),
                "manifest_path": str(jsonl_path),
                "source_file_path": None,
                "instruction_text": instruction,
            }

    tasks = list(dedup.values())
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(tasks)
    else:
        tasks.sort(key=lambda item: item["id"])

    if limit is not None:
        tasks = tasks[:limit]

    return tasks, {
        "jsonl_count": len(jsonl_paths),
        "selected_count": len(tasks),
        "dedup_count_before_limit": len(dedup),
        "missing_instruction_count": len(missing_instruction),
        "missing_instruction_examples": missing_instruction[:20],
    }


def make_client(args):
    client = LLMAPIClient(config_path=str(args.config), model_override=args.model)
    client.max_tokens = args.max_tokens
    client.temperature = args.temperature
    client.max_retries = args.max_retries
    client.timeout = args.timeout
    return client


def process_one(task: dict, args, raw_dir: Path, code_dir: Path, meta_dir: Path, prompt_dir: Path, markdown_dir: Path):
    sample_id = task["id"]
    meta_path = meta_dir / f"{sample_id}.json"
    raw_path = raw_dir / f"{sample_id}.txt"
    code_path = code_dir / f"{sample_id}.py"
    prompt_path = prompt_dir / f"{sample_id}.txt"
    markdown_path = markdown_dir / f"{sample_id}.md"

    if args.resume and meta_path.exists() and raw_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("status") == "success":
            old["resumed"] = True
            return old

    if task.get("instruction_text") is not None:
        markdown_text = None
        instruction = task["instruction_text"]
    else:
        markdown_text = Path(task["md_path"]).read_text(encoding="utf-8")
        instruction = build_benchmark_instruction(markdown_text)
        markdown_path.write_text(markdown_text, encoding="utf-8")
    prompt_path.write_text(instruction, encoding="utf-8")
    client = make_client(args)

    started_at = time.time()
    start_perf = time.perf_counter()
    raw_output = client.call_api_with_text(
        instruction,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    elapsed = time.perf_counter() - start_perf
    usage = client.last_response_usage or {}

    cleaned = clean_model_output(raw_output)
    if cleaned and not looks_like_code(cleaned):
        cleaned = ""
    cleaned = inject_font_defaults(cleaned) if cleaned else ""
    raw_path.write_text(raw_output or "", encoding="utf-8")
    if cleaned:
        code_path.write_text(cleaned, encoding="utf-8")

    result = {
        "id": sample_id,
        "status": "success" if cleaned else "empty_after_cleanup",
        "model": args.model,
        "source_id": task.get("source_id"),
        "dataset_tag": task.get("dataset_tag"),
        "md_path": task["md_path"],
        "manifest_path": task["manifest_path"],
        "source_file_path": task.get("source_file_path"),
        "markdown_sha256": task["markdown_sha256"],
        "markdown_chars": task["markdown_chars"],
        "markdown_snapshot_path": str(markdown_path) if markdown_text is not None else None,
        "instruction_sha256": task["instruction_sha256"],
        "instruction_chars": task["instruction_chars"],
        "prompt_snapshot_path": str(prompt_path),
        "raw_output_path": str(raw_path),
        "cleaned_code_path": str(code_path) if cleaned else None,
        "raw_chars": len(raw_output or ""),
        "cleaned_chars": len(cleaned or ""),
        "llm_time": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "cached_tokens": usage.get("cached_tokens"),
        "started_at": started_at,
        "finished_at": time.time(),
        "resumed": False,
    }
    write_json(meta_path, result)
    return result


def save_snapshot(output_dir: Path, args, tasks: list[dict], results: list[dict], build_info: dict):
    success = [item for item in results if item.get("status") == "success"]
    empty = [item for item in results if item.get("status") == "empty_after_cleanup"]
    summary = {
        "model": args.model,
        "total_tasks": len(tasks),
        "completed": len(results),
        "success_count": len(success),
        "empty_after_cleanup_count": len(empty),
        "avg_llm_time": mean_field(results, "llm_time"),
        "avg_prompt_tokens": mean_field(results, "prompt_tokens"),
        "avg_completion_tokens": mean_field(results, "completion_tokens"),
        "avg_total_tokens": mean_field(results, "total_tokens"),
        "avg_reasoning_tokens": mean_field(results, "reasoning_tokens"),
        "avg_cached_tokens": mean_field(results, "cached_tokens"),
        "build_info": build_info,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "results.json", {"summary": summary, "details": results})


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Manim code from benchmark prompts.")
    parser.add_argument(
        "--audit-results",
        nargs="+",
        help="One or more audit result json files.",
    )
    parser.add_argument(
        "--input-jsonl",
        nargs="+",
        help="One or more jsonl files that already contain instruction/id fields.",
    )
    parser.add_argument(
        "--instruction-field",
        default="instruction",
        help="Field name to read from --input-jsonl rows.",
    )
    parser.add_argument(
        "--md-roots",
        nargs="+",
        default=[str(path) for path in DEFAULT_MD_ROOTS],
        help="Search roots for <sample_id>.md.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path passed to LLMAPIClient config_path.")
    parser.add_argument("--model", default="your-model-name")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.audit_results and not args.input_jsonl:
        raise ValueError("Either --audit-results or --input-jsonl must be provided.")

    manifest_paths = [Path(path) for path in args.audit_results or []]
    input_jsonl_paths = [Path(path) for path in args.input_jsonl or []]
    md_roots = [Path(path) for path in args.md_roots]

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / f"generation_{safe_name(args.model)}_{now_tag()}"
    )
    raw_dir = output_dir / "raw_outputs"
    code_dir = output_dir / "cleaned_scripts"
    meta_dir = output_dir / "meta"
    prompt_dir = output_dir / "prompt_snapshots"
    markdown_dir = output_dir / "markdown_snapshots"
    for path in (output_dir, raw_dir, code_dir, meta_dir, prompt_dir, markdown_dir):
        path.mkdir(parents=True, exist_ok=True)

    if input_jsonl_paths:
        tasks, build_info = build_tasks_from_jsonl(
            jsonl_paths=input_jsonl_paths,
            limit=args.limit,
            shuffle=args.shuffle,
            seed=args.seed,
            instruction_field=args.instruction_field,
        )
    else:
        tasks, build_info = build_tasks(
            manifest_paths=manifest_paths,
            md_roots=md_roots,
            limit=args.limit,
            shuffle=args.shuffle,
            seed=args.seed,
        )

    config_payload = {
        "audit_results": [str(path) for path in manifest_paths],
        "input_jsonl": [str(path) for path in input_jsonl_paths],
        "md_roots": [str(path) for path in md_roots],
        "audit_result_sha256": {str(path): sha256_file(path) for path in manifest_paths},
        "input_jsonl_sha256": {str(path): sha256_file(path) for path in input_jsonl_paths},
        "instruction_field": args.instruction_field,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(Path(args.config)),
        "model": args.model,
        "workers": args.workers,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "max_retries": args.max_retries,
        "timeout": args.timeout,
        "limit": args.limit,
        "seed": args.seed,
        "save_interval": args.save_interval,
        "resume": args.resume,
        "shuffle": args.shuffle,
        "dry_run": args.dry_run,
            "build_info": build_info,
    }
    write_json(output_dir / "run_config.json", config_payload)
    write_json(output_dir / "task_manifest.json", tasks)

    if args.dry_run:
        print(json.dumps({"output_dir": str(output_dir), "config": config_payload}, ensure_ascii=False, indent=2))
        return

    results = []
    results_lock = threading.Lock()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(process_one, task, args, raw_dir, code_dir, meta_dir, prompt_dir, markdown_dir): task
            for task in tasks
        }
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "id": task["id"],
                    "status": "failed",
                    "model": args.model,
                    "source_id": task.get("source_id"),
                    "dataset_tag": task.get("dataset_tag"),
                    "md_path": task["md_path"],
                    "manifest_path": task["manifest_path"],
                    "source_file_path": task.get("source_file_path"),
                    "markdown_sha256": task["markdown_sha256"],
                    "markdown_chars": task["markdown_chars"],
                    "markdown_snapshot_path": str(markdown_dir / f"{task['id']}.md"),
                    "instruction_sha256": task["instruction_sha256"],
                    "instruction_chars": task["instruction_chars"],
                    "prompt_snapshot_path": str(prompt_dir / f"{task['id']}.txt"),
                    "raw_output_path": None,
                    "cleaned_code_path": None,
                    "raw_chars": 0,
                    "cleaned_chars": 0,
                    "llm_time": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "resumed": False,
                }
                write_json(meta_dir / f"{task['id']}.json", result)

            with results_lock:
                results.append(result)
                completed += 1
                if completed % max(1, args.save_interval) == 0 or completed == len(tasks):
                    save_snapshot(output_dir, args, tasks, sorted(results, key=lambda item: item["id"]), build_info)
                    print(
                        f"[{completed}/{len(tasks)}] success="
                        f"{sum(1 for item in results if item.get('status') == 'success')} "
                        f"failed={sum(1 for item in results if item.get('status') == 'failed')} "
                        f"empty={sum(1 for item in results if item.get('status') == 'empty_after_cleanup')}"
                    )

    save_snapshot(output_dir, args, tasks, sorted(results, key=lambda item: item["id"]), build_info)
    print(f"Done. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
