import re
from collections import Counter


ERROR_TYPE_ORDER = [
    "empty_output",
    "provider_error",
    "timeout_error",
    "format_contamination",
    "syntax_error",
    "api_hallucination",
    "api_misuse",
    "text_render_error",
    "asset_error",
    "value_error",
    "other_runtime_error",
    "other_error",
]

ERROR_STAGE_ORDER = [
    "llm_call",
    "llm_output",
    "parse",
    "render",
    "postprocess",
    "other",
]

ERROR_TYPE_LABELS = {
    "empty_output": "Empty output",
    "provider_error": "Provider/server",
    "timeout_error": "Timeout",
    "format_contamination": "Format contamination",
    "syntax_error": "Syntax error",
    "api_hallucination": "API hallucination",
    "api_misuse": "API misuse",
    "text_render_error": "Text render error",
    "asset_error": "Asset error",
    "value_error": "Value/semantic error",
    "other_runtime_error": "Other runtime",
    "other_error": "Other error",
}

ERROR_STAGE_LABELS = {
    "llm_call": "LLM call",
    "llm_output": "LLM output",
    "parse": "Python parse",
    "render": "Manim render",
    "postprocess": "Post-process",
    "other": "Other",
}

_PROVIDER_PATTERNS = (
    "server busy",
    "service unavailable",
    "provider busy",
    "rate limit",
    "too many requests",
    "temporarily unavailable",
    "overloaded",
    "upstream",
    "bad gateway",
    "gateway timeout",
    "status code 429",
    "status code 502",
    "status code 503",
    "status code 504",
)

_TEXT_RENDER_PATTERNS = (
    "latex error converting to dvi",
    "unknown tag",
    "continuation byte",
    "invalid continuation byte",
    "markup",
    "pangocairo",
    "pangomarkup",
    "tex error",
)

_ASSET_PATTERNS = (
    "must specify file for svgmobject",
    "no such file or directory",
    "file not found",
    "cannot find image",
    "cannot open resource",
    "failed to load svg",
)

_API_MISUSE_PATTERNS = (
    "unexpected keyword argument",
    "unexpected positional argument",
    "unexpected keyword",
    "got multiple values for keyword argument",
    "multiple values for keyword argument",
    "has no attribute",
    "object has no attribute",
    "module 'manim",
    "module 'manim.utils",
    "module 'manim.'",
    "got an unexpected keyword",
    "missing 1 required positional argument",
    "missing required positional argument",
    "takes ",
)

_VALUE_PATTERNS = (
    "too few rows and columns",
    "a.any() or a.all()",
    "must be str or falsy value",
    "argument 'stretch'",
    "argument 'x'",
    "argument 'y'",
    "argument 'z'",
    "keyword argument 'font_size'",
    "values for keyword argument",
    "could not broadcast",
)

_PROSE_HINTS = {
    "replace",
    "using",
    "ensure",
    "positioning",
    "scale",
    "keep",
    "avoid",
    "make",
    "center",
    "render",
    "set",
    "adjust",
    "move",
    "use",
    "add",
}


def _clean_text(value):
    return str(value or "").strip()


def _contains_any(text, patterns):
    lowered = _clean_text(text).lower()
    return any(pattern in lowered for pattern in patterns)


def _line_looks_like_prose(line):
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    if stripped.startswith(("```", "<think", "</think", "Thought process", "Reasoning:", "思考过程", "推理过程")):
        return True
    if re.match(r"^[-*+]\s+[A-Za-z\u4e00-\u9fff]", stripped):
        return True
    if re.match(r"^\d+\.\s+[A-Za-z\u4e00-\u9fff]", stripped):
        return True
    if re.search(r"\)\s*-\s+[A-Za-z\u4e00-\u9fff]", stripped):
        return True
    words = re.findall(r"[A-Za-z]+", stripped)
    if words and words[0].lower() in _PROSE_HINTS and len(words) >= 3 and "=" not in stripped:
        return True
    if len(words) >= 5 and not any(ch in stripped for ch in "()[]=,.:"):
        return True
    return False


def looks_like_format_contamination(code, parse_error=None):
    error_text = _clean_text(parse_error).lower()
    if "invalid character" in error_text:
        return True

    text = _clean_text(code)
    if not text:
        return False

    for line in text.splitlines():
        if _line_looks_like_prose(line):
            return True

    if any(marker in text for marker in ("```", "<think", "</think>")):
        return True

    return False


def classify_result_error(detail, code=None):
    if detail.get("pass"):
        return {
            "error_stage": None,
            "error_type": None,
            "error_type_display": None,
            "error_flags": [],
        }

    error_msg = _clean_text(detail.get("error_msg"))
    error_trace = _clean_text(detail.get("error_trace"))
    parse_error = _clean_text(detail.get("code_parse_error"))
    hallucination_count = int(detail.get("hallucination_count", 0) or 0)
    combined = "\n".join(part for part in (error_msg, error_trace, parse_error) if part).lower()
    flags = []

    if parse_error:
        flags.append("code_parse_error")
    if hallucination_count > 0:
        flags.append("api_hallucination_detected")

    if "empty model output after cleanup" in combined or "empty response" in combined:
        error_stage = "llm_output"
        error_type = "empty_output"
    elif _contains_any(combined, _PROVIDER_PATTERNS):
        error_stage = "llm_call"
        error_type = "provider_error"
    elif "timed out" in combined or "timeout" in combined:
        error_stage = "render"
        error_type = "timeout_error"
    elif parse_error:
        error_stage = "parse"
        if looks_like_format_contamination(code, parse_error=parse_error):
            error_type = "format_contamination"
            flags.append("format_contamination_detected")
        else:
            error_type = "syntax_error"
    elif hallucination_count > 0:
        error_stage = "render"
        error_type = "api_hallucination"
    elif _contains_any(combined, _ASSET_PATTERNS):
        error_stage = "render"
        error_type = "asset_error"
    elif _contains_any(combined, _TEXT_RENDER_PATTERNS):
        error_stage = "render"
        error_type = "text_render_error"
    elif _contains_any(combined, _API_MISUSE_PATTERNS):
        error_stage = "render"
        error_type = "api_misuse"
    elif _contains_any(combined, _VALUE_PATTERNS):
        error_stage = "render"
        error_type = "value_error"
    elif error_msg or error_trace:
        error_stage = "render"
        error_type = "other_runtime_error"
    else:
        error_stage = "other"
        error_type = "other_error"

    return {
        "error_stage": error_stage,
        "error_type": error_type,
        "error_type_display": ERROR_TYPE_LABELS[error_type],
        "error_flags": sorted(set(flags)),
    }


def annotate_result_error(detail, code=None):
    detail.update(classify_result_error(detail, code=code))
    return detail


def aggregate_error_breakdown(details):
    total = len(details)
    failed = [item for item in details if not item.get("pass")]
    failure_count = len(failed)
    type_counts = Counter()
    stage_counts = Counter()

    for item in failed:
        error_type = item.get("error_type")
        error_stage = item.get("error_stage")
        if error_type:
            type_counts[error_type] += 1
        if error_stage:
            stage_counts[error_stage] += 1

    return {
        "failure_count": failure_count,
        "failure_rate": failure_count / total if total else 0.0,
        "error_type_counts": {
            key: type_counts.get(key, 0) for key in ERROR_TYPE_ORDER
        },
        "error_type_rates_overall": {
            key: (type_counts.get(key, 0) / total if total else 0.0) for key in ERROR_TYPE_ORDER
        },
        "error_type_rates_within_failures": {
            key: (type_counts.get(key, 0) / failure_count if failure_count else 0.0)
            for key in ERROR_TYPE_ORDER
        },
        "error_stage_counts": {
            key: stage_counts.get(key, 0) for key in ERROR_STAGE_ORDER
        },
        "error_stage_rates_overall": {
            key: (stage_counts.get(key, 0) / total if total else 0.0) for key in ERROR_STAGE_ORDER
        },
        "error_stage_rates_within_failures": {
            key: (stage_counts.get(key, 0) / failure_count if failure_count else 0.0)
            for key in ERROR_STAGE_ORDER
        },
    }
