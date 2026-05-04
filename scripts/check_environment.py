#!/usr/bin/env python3
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys

REQUIRED_IMPORTS = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("manim", "manim"),
    ("PIL", "Pillow"),
    ("jieba", "jieba"),
    ("sentence_transformers", "sentence-transformers"),
    ("sklearn", "scikit-learn"),
]
REQUIRED_OCR_IMPORTS = [
    ("rapidocr_onnxruntime", "rapidocr-onnxruntime"),
]
OPTIONAL_IMPORTS = [
    ("openai", "openai"),
    ("paddleocr", "paddleocr"),
]
REQUIRED_COMMANDS = ["manim", "ffmpeg"]
OPTIONAL_COMMANDS = ["latex", "xelatex", "dvisvgm"]
ENV_VARS = [
    "MANIM_BENCH_LLM_CONFIG",
    "MANIM_BENCH_DATA_DIR",
    "MANIM_BENCH_RESULTS_DIR",
    "PADVC_HF_CACHE",
    "PADVC_ZH_MODEL",
    "PADVC_EN_MODEL",
    "PADVC_OCR_CACHE_DIR",
]


def has_import(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def command_version(cmd: str) -> str:
    path = shutil.which(cmd)
    if not path:
        return "missing"
    try:
        output = subprocess.run(
            [cmd, "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        first = output.stdout.splitlines()[0] if output.stdout else path
        return first[:160]
    except Exception:
        return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Manim-Bench runtime dependencies.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when optional OCR fallback or LaTeX components are missing.",
    )
    args = parser.parse_args()

    failed = False
    optional_failed = False
    print(f"Python: {sys.version.split()[0]}")

    print("\nPython packages:")
    for module, package in REQUIRED_IMPORTS:
        ok = has_import(module)
        print(f"  [{'ok' if ok else 'missing'}] {package}")
        failed = failed or not ok
    for module, package in REQUIRED_OCR_IMPORTS:
        ok = has_import(module)
        print(f"  [{'ok' if ok else 'missing'}] {package} (default OCR backend)")
        failed = failed or not ok
    for module, package in OPTIONAL_IMPORTS:
        ok = has_import(module)
        note = "optional generation client" if package == "openai" else "optional OCR fallback"
        print(f"  [{'ok' if ok else 'missing'}] {package} ({note})")
        optional_failed = optional_failed or not ok

    print("\nExternal commands:")
    for cmd in REQUIRED_COMMANDS:
        version = command_version(cmd)
        ok = version != "missing"
        print(f"  [{'ok' if ok else 'missing'}] {cmd}: {version}")
        failed = failed or not ok
    for cmd in OPTIONAL_COMMANDS:
        version = command_version(cmd)
        ok = version != "missing"
        print(f"  [{'ok' if ok else 'missing'}] {cmd}: {version} (needed for TeX-heavy scenes)")
        optional_failed = optional_failed or not ok

    print("\nEnvironment variables:")
    for name in ENV_VARS:
        value = os.environ.get(name)
        print(f"  {name}={value or '<unset>'}")

    if failed or (args.strict and optional_failed):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
