from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESULTS_DIR = Path(os.environ.get("MANIM_BENCH_RESULTS_DIR", PROJECT_ROOT / "results"))
TMP_DIR = Path(os.environ.get("MANIM_BENCH_TMP_DIR", PROJECT_ROOT / ".tmp"))
DATA_DIR = Path(os.environ.get("MANIM_BENCH_DATA_DIR", PROJECT_ROOT / "data"))
DOCS_DIR = PROJECT_ROOT / "docs"
LLM_CONFIG_PATH = Path(
    os.environ.get(
        "MANIM_BENCH_LLM_CONFIG",
        PROJECT_ROOT / "manim_bench" / "llm_call" / "config.json",
    )
)


def get_python_bin() -> str:
    return os.environ.get("MANIM_BENCH_PYTHON", sys.executable)


def get_manim_bin() -> str:
    return os.environ.get("MANIM_BENCH_MANIM_BIN", "manim")


def get_score_python() -> str:
    return os.environ.get("MANIM_BENCH_SCORE_PYTHON", sys.executable)


def get_tmp_subdir(name: str) -> Path:
    path = TMP_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_markdown_roots() -> list[Path]:
    values: list[Path] = [DATA_DIR / "markdowns"]
    extra = os.environ.get("MANIM_BENCH_MD_ROOTS", "").strip()
    if extra:
        for item in extra.split(os.pathsep):
            item = item.strip()
            if item:
                values.append(Path(item))
    return values


def get_analysis_dir(name: str = "analysis") -> Path:
    path = RESULTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path
