from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_PATTERN = re.compile(r'\bImageMobject\s*\(|/pictures/|\.(png|jpg|jpeg|webp|svg|gif)["\']', re.I)


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def default_report_dir(input_path: Path | None = None) -> Path:
    if input_path is not None:
        return input_path.parent / "reports"
    return Path("reports")


def normalize_code(code: str) -> str:
    stripped = code.strip()
    return stripped + "\n" if stripped else ""


def is_self_wait_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "wait"
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
    )


def annotate_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent


def annotate_original_body_lengths(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            value = getattr(node, field, None)
            if isinstance(value, list):
                setattr(node, f"_orig_{field}_len", len(value))


def count_special_waits(tree: ast.AST) -> tuple[int, int]:
    nonliteral = 0
    nonexpr = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and is_self_wait_call(node.value):
            if not node.value.args:
                nonliteral += 1
                continue
            arg = node.value.args[0]
            if not isinstance(arg, ast.Constant) or not isinstance(arg.value, (int, float)):
                nonliteral += 1
        elif is_self_wait_call(node):
            if not isinstance(getattr(node, "parent", None), ast.Expr):
                nonexpr += 1
    return nonliteral, nonexpr


@dataclass
class WaitTransformStats:
    removed_decimal_waits: int = 0
    capped_integer_waits: int = 0
    inserted_pass_nodes: int = 0


class WaitCleaner(ast.NodeTransformer):
    def __init__(self, decimal_threshold: float, integer_cap: int) -> None:
        self.decimal_threshold = decimal_threshold
        self.integer_cap = integer_cap
        self.stats = WaitTransformStats()

    def visit_Expr(self, node: ast.Expr) -> Any:
        node = self.generic_visit(node)
        if not isinstance(node, ast.Expr) or not is_self_wait_call(node.value):
            return node
        if not node.value.args:
            return node
        arg = node.value.args[0]
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, (int, float)):
            return node

        value = arg.value
        if isinstance(value, float) and value > self.decimal_threshold:
            self.stats.removed_decimal_waits += 1
            return None
        if isinstance(value, int) and value >= self.integer_cap:
            if value != self.integer_cap:
                self.stats.capped_integer_waits += 1
                node.value.args[0] = ast.copy_location(ast.Constant(value=self.integer_cap), arg)
            return node
        return node


class EmptyBodyPassFixer(ast.NodeTransformer):
    BODY_FIELDS = ("body", "orelse", "finalbody")

    def __init__(self) -> None:
        self.inserted = 0

    def generic_visit(self, node: ast.AST) -> Any:
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                original_len = getattr(node, f"_orig_{field}_len", len(old_value))
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        if not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                if field in self.BODY_FIELDS and original_len > 0 and len(new_values) == 0:
                    new_values = [ast.Pass()]
                    self.inserted += 1
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


def transform_waits(code: str, decimal_threshold: float, integer_cap: int) -> tuple[str, WaitTransformStats, int, int]:
    original_tree = ast.parse(code)
    annotate_parents(original_tree)
    skipped_nonliteral, skipped_nonexpr = count_special_waits(original_tree)

    tree = ast.parse(code)
    annotate_original_body_lengths(tree)
    cleaner = WaitCleaner(decimal_threshold=decimal_threshold, integer_cap=integer_cap)
    tree = cleaner.visit(tree)
    ast.fix_missing_locations(tree)

    fixer = EmptyBodyPassFixer()
    tree = fixer.visit(tree)
    ast.fix_missing_locations(tree)

    cleaner.stats.inserted_pass_nodes = fixer.inserted
    return normalize_code(ast.unparse(tree)), cleaner.stats, skipped_nonliteral, skipped_nonexpr


def compile_ok(code: str) -> bool:
    compile(code, "<curated_output>", "exec")
    return True


@dataclass
class WaitCleanFileReport:
    path: str
    output_path: str
    backup_path: str | None
    total_rows: int = 0
    rows_with_output: int = 0
    changed_rows: int = 0
    unchanged_rows: int = 0
    parse_fail_rows: int = 0
    compile_fail_after_transform_rows: int = 0
    original_compile_fail_rows: int = 0
    removed_decimal_waits: int = 0
    capped_integer_waits: int = 0
    inserted_pass_nodes: int = 0
    skipped_nonliteral_waits: int = 0
    skipped_nonexpr_waits: int = 0


def clean_waits_file(
    input_path: Path,
    output_path: Path,
    decimal_threshold: float,
    integer_cap: int,
    backup_original: bool,
) -> WaitCleanFileReport:
    backup_path = None
    if backup_original and input_path == output_path:
        backup_path = input_path.with_name(input_path.name + f".bak_wait_clean_{timestamp_tag()}")
        shutil.copy2(input_path, backup_path)

    report = WaitCleanFileReport(
        path=str(input_path),
        output_path=str(output_path),
        backup_path=str(backup_path) if backup_path else None,
    )
    cleaned_rows: list[dict] = []

    for row in load_jsonl(input_path):
        report.total_rows += 1
        output = row.get("output")
        if not isinstance(output, str):
            cleaned_rows.append(row)
            continue

        report.rows_with_output += 1
        try:
            compile_ok(output)
        except SyntaxError:
            report.original_compile_fail_rows += 1

        try:
            new_output, stats, skipped_nonliteral, skipped_nonexpr = transform_waits(
                output,
                decimal_threshold=decimal_threshold,
                integer_cap=integer_cap,
            )
        except SyntaxError:
            report.parse_fail_rows += 1
            cleaned_rows.append(row)
            continue

        report.skipped_nonliteral_waits += skipped_nonliteral
        report.skipped_nonexpr_waits += skipped_nonexpr

        if new_output != normalize_code(output):
            try:
                compile_ok(new_output)
            except SyntaxError:
                report.compile_fail_after_transform_rows += 1
                cleaned_rows.append(row)
                continue
            row["output"] = new_output
            report.changed_rows += 1
            report.removed_decimal_waits += stats.removed_decimal_waits
            report.capped_integer_waits += stats.capped_integer_waits
            report.inserted_pass_nodes += stats.inserted_pass_nodes
        else:
            report.unchanged_rows += 1
        cleaned_rows.append(row)

    write_jsonl(output_path, cleaned_rows)
    return report


@dataclass
class ReplaceImageReport:
    target: str
    output_path: str
    backup_path: str | None
    target_language: str
    total_rows_before: int
    image_rows_before: int
    image_rows_after: int
    selected_total: int
    duplicate_ids_after: int
    unused_candidates_before: dict[str, int]
    unused_candidates_after: dict[str, int]
    selected_candidates: dict[str, int]
    mapping_examples: list[dict]


def build_candidate_row(label: str, index: int, row: dict, language: str) -> dict:
    line_id = f"line_{index:06d}"
    source_id = f"{label}__{line_id}"
    instruction = row.get("instruction") or row.get("user_input") or ""
    return {
        "id": f"{language}__{source_id}",
        "language": language,
        "source_id": source_id,
        "instruction": instruction,
        "user_input": instruction,
        "output": row.get("output", ""),
        "output_source": f"{label}_jsonl",
        "output_source_path": row.get("_source_path"),
    }


def has_image_output(row: dict, language: str) -> bool:
    return bool(row.get("language") == language and IMAGE_PATTERN.search(row.get("output", "") or ""))


def replace_image_rows(
    target_jsonl: Path,
    output_jsonl: Path,
    candidate_specs: list[tuple[str, Path]],
    target_language: str,
    backup_original: bool,
) -> ReplaceImageReport:
    rows = load_jsonl(target_jsonl)
    current_ids = {row.get("id") for row in rows if row.get("id")}
    target_image_rows_before = sum(1 for row in rows if has_image_output(row, target_language))

    candidate_pools: dict[str, list[dict]] = {}
    for label, path in candidate_specs:
        pool = []
        for index, row in enumerate(load_jsonl(path), start=1):
            paired_ids = {
                f"zh__{label}__line_{index:06d}",
                f"en__{label}__line_{index:06d}",
            }
            if current_ids & paired_ids:
                continue
            if IMAGE_PATTERN.search(row.get("output", "") or ""):
                continue
            normalized = dict(row)
            normalized["_source_path"] = str(path.resolve())
            pool.append(build_candidate_row(label=label, index=index, row=normalized, language=target_language))
        candidate_pools[label] = pool

    available_before = {label: len(pool) for label, pool in candidate_pools.items()}
    available_total = sum(available_before.values())
    if available_total < target_image_rows_before:
        raise RuntimeError(
            f"Not enough replacement candidates: need {target_image_rows_before}, have {available_total}"
        )

    selected_counts = {label: 0 for label, _ in candidate_specs}
    selected_rows: list[dict] = []
    remaining = target_image_rows_before
    for label, _ in candidate_specs:
        take = min(len(candidate_pools[label]), remaining)
        selected_rows.extend(candidate_pools[label][:take])
        selected_counts[label] += take
        candidate_pools[label] = candidate_pools[label][take:]
        remaining -= take
        if remaining == 0:
            break
    if remaining != 0:
        raise RuntimeError(f"Replacement selection failed: still need {remaining}")

    selected_iter = iter(selected_rows)
    mapping_examples: list[dict] = []
    replaced_rows: list[dict] = []
    for row in rows:
        if has_image_output(row, target_language):
            new_row = next(selected_iter)
            if len(mapping_examples) < 10:
                mapping_examples.append({"old_id": row.get("id"), "new_id": new_row.get("id")})
            replaced_rows.append(new_row)
        else:
            replaced_rows.append(row)

    backup_path = None
    if backup_original and target_jsonl == output_jsonl:
        backup_path = target_jsonl.with_name(target_jsonl.name + f".bak_image_replace_{timestamp_tag()}")
        shutil.copy2(target_jsonl, backup_path)

    write_jsonl(output_jsonl, replaced_rows)

    ids_after = [row.get("id") for row in replaced_rows if row.get("id")]
    image_rows_after = sum(1 for row in replaced_rows if has_image_output(row, target_language))
    return ReplaceImageReport(
        target=str(target_jsonl),
        output_path=str(output_jsonl),
        backup_path=str(backup_path) if backup_path else None,
        target_language=target_language,
        total_rows_before=len(rows),
        image_rows_before=target_image_rows_before,
        image_rows_after=image_rows_after,
        selected_total=len(selected_rows),
        duplicate_ids_after=len(ids_after) - len(set(ids_after)),
        unused_candidates_before=available_before,
        unused_candidates_after={label: len(pool) for label, pool in candidate_pools.items()},
        selected_candidates={label: count for label, count in selected_counts.items() if count > 0},
        mapping_examples=mapping_examples,
    )


def add_common_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--report-dir", type=Path, help="Directory for JSON reports.")


def save_report(report_dir: Path, prefix: str, payload: dict) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{prefix}_{timestamp_tag()}.json"
    write_json(report_path, payload)
    return report_path


def parse_candidate_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate spec must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("candidate label cannot be empty")
    return label, Path(raw_path).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset curation utilities for Manim-Bench.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean_waits = subparsers.add_parser(
        "clean-waits",
        help="Remove long decimal self.wait calls and cap long integer waits with AST-aware rewriting.",
    )
    clean_waits.add_argument("--input-jsonl", nargs="+", type=Path, required=True)
    clean_waits.add_argument("--output-dir", type=Path)
    clean_waits.add_argument("--in-place", action="store_true")
    clean_waits.add_argument("--decimal-threshold", type=float, default=3.0)
    clean_waits.add_argument("--integer-cap", type=int, default=10)
    add_common_report_arguments(clean_waits)

    replace_images = subparsers.add_parser(
        "replace-image-rows",
        help="Replace image-containing rows with unused no-image rows from candidate JSONL files.",
    )
    replace_images.add_argument("--target-jsonl", type=Path, required=True)
    replace_images.add_argument("--candidate-jsonl", nargs="+", type=parse_candidate_spec, required=True)
    replace_images.add_argument("--output-jsonl", type=Path)
    replace_images.add_argument("--in-place", action="store_true")
    replace_images.add_argument("--target-language", default="zh")
    add_common_report_arguments(replace_images)
    return parser


def resolve_output_path(input_path: Path, output_dir: Path | None, in_place: bool) -> Path:
    if in_place:
        return input_path
    if output_dir is None:
        raise ValueError("either --in-place or --output-dir must be provided")
    return output_dir / input_path.name


def run_clean_waits(args: argparse.Namespace) -> None:
    if args.in_place and args.output_dir is not None:
        raise ValueError("--in-place and --output-dir cannot be used together")

    reports = []
    for input_path in args.input_jsonl:
        output_path = resolve_output_path(input_path, args.output_dir, args.in_place)
        report = clean_waits_file(
            input_path=input_path,
            output_path=output_path,
            decimal_threshold=args.decimal_threshold,
            integer_cap=args.integer_cap,
            backup_original=args.in_place,
        )
        reports.append(asdict(report))

    report_dir = args.report_dir or default_report_dir(args.input_jsonl[0] if args.input_jsonl else None)
    report_path = save_report(
        report_dir,
        prefix="wait_cleanup",
        payload={
            "command": "clean-waits",
            "decimal_threshold": args.decimal_threshold,
            "integer_cap": args.integer_cap,
            "reports": reports,
        },
    )
    print(json.dumps({"reports_saved_to": str(report_path), "files": len(reports)}, ensure_ascii=False, indent=2))


def run_replace_image_rows(args: argparse.Namespace) -> None:
    if args.in_place and args.output_jsonl is not None:
        raise ValueError("--in-place and --output-jsonl cannot be used together")
    output_jsonl = args.target_jsonl if args.in_place else args.output_jsonl
    if output_jsonl is None:
        raise ValueError("either --in-place or --output-jsonl must be provided")

    report = replace_image_rows(
        target_jsonl=args.target_jsonl,
        output_jsonl=output_jsonl,
        candidate_specs=args.candidate_jsonl,
        target_language=args.target_language,
        backup_original=args.in_place,
    )
    report_dir = args.report_dir or default_report_dir(args.target_jsonl)
    report_path = save_report(
        report_dir,
        prefix="image_row_replacement",
        payload={"command": "replace-image-rows", "report": asdict(report)},
    )
    print(
        json.dumps(
            {
                "reports_saved_to": str(report_path),
                "image_rows_before": report.image_rows_before,
                "image_rows_after": report.image_rows_after,
                "selected_total": report.selected_total,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "clean-waits":
        run_clean_waits(args)
    elif args.command == "replace-image-rows":
        run_replace_image_rows(args)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
