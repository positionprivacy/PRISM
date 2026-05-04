#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 <input_dir> <run_dir> <task_manifest> <sample_jsonl> <padvc_params_json> <td_params_json> <label>" >&2
  exit 1
fi

INPUT_DIR="$1"
RUN_DIR="$2"
TASK_MANIFEST="$3"
SAMPLE_JSONL="$4"
PADVC_PARAMS="$5"
TD_PARAMS="$6"
LABEL="$7"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIM_BIN="${MANIM_BIN:-manim}"
RENDER_SCRIPT="${RENDER_SCRIPT:-${SCRIPT_DIR}/render_directory.py}"
AUDIT_SCRIPT="${AUDIT_SCRIPT:-${SCRIPT_DIR}/audit_batch.py}"
SCORE_PADVC_SCRIPT="${SCORE_PADVC_SCRIPT:-${SCRIPT_DIR}/score_padvc.py}"
MERGE_PADVC_SCRIPT="${MERGE_PADVC_SCRIPT:-${SCRIPT_DIR}/merge_padvc_shards.py}"
SCORE_TD_SCRIPT="${SCORE_TD_SCRIPT:-${SCRIPT_DIR}/score_td.py}"
TEXT_EXPANSION_SCRIPT="${TEXT_EXPANSION_SCRIPT:-${SCRIPT_DIR}/compute_text_expansion.py}"

RENDER_WORKERS="${RENDER_WORKERS:-8}"
AUDIT_WORKERS="${AUDIT_WORKERS:-8}"
PADVC_SHARDS="${PADVC_SHARDS:-8}"
TD_WORKERS="${TD_WORKERS:-16}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
RENDER_TIMEOUT_SEC="${RENDER_TIMEOUT_SEC:-600}"
AUDIT_TIMEOUT_SEC="${AUDIT_TIMEOUT_SEC:-900}"
AUDIT_SAVE_IMAGES="${AUDIT_SAVE_IMAGES:-0}"

PADVC_DEVICE="${PADVC_DEVICE:-cpu}"
PADVC_OCR_BACKEND="${PADVC_OCR_BACKEND:-rapidocr}"
PADVC_P="${PADVC_P:-0.7}"
PADVC_EVENT_THRESHOLD_MODE="${PADVC_EVENT_THRESHOLD_MODE:-absolute}"
PADVC_EVENT_THRESHOLD_ABS="${PADVC_EVENT_THRESHOLD_ABS:-50000.0}"
PADVC_EVENT_THRESHOLD_RATIO="${PADVC_EVENT_THRESHOLD_RATIO:-0.08}"
PADVC_DELTA_MODE="${PADVC_DELTA_MODE:-positive}"
PADVC_TEXT_DILATE="${PADVC_TEXT_DILATE:-7}"
PADVC_RAPID_DET_LIMIT_SIDE_LEN="${PADVC_RAPID_DET_LIMIT_SIDE_LEN:-736}"

VIDEO_DIR="${RUN_DIR}/videos"
RENDER_RESULTS="${RUN_DIR}/render_results.json"
AUDIT_DIR="${RUN_DIR}/audit"
AUDIT_RESULTS="${AUDIT_DIR}/results.json"
OCR_CACHE_DIR="${RUN_DIR}/ocr_cache"
PADVC_SHARD_ROOT="${RUN_DIR}/padvc_shards"
PADVC_FINAL_DIR="${RUN_DIR}/padvc_final"
TD_DIR="${RUN_DIR}/td_final"
LOG_DIR="${RUN_DIR}/logs"
RUN_LOG="${RUN_DIR}/run.log"

mkdir -p "${RUN_DIR}" "${VIDEO_DIR}" "${AUDIT_DIR}" "${OCR_CACHE_DIR}" "${PADVC_SHARD_ROOT}" "${PADVC_FINAL_DIR}" "${TD_DIR}" "${LOG_DIR}"

padvc_common_args=(
  --norm-params "${PADVC_PARAMS}"
  --device "${PADVC_DEVICE}"
  --p "${PADVC_P}"
  --event-threshold-mode "${PADVC_EVENT_THRESHOLD_MODE}"
  --event-threshold-abs "${PADVC_EVENT_THRESHOLD_ABS}"
  --event-threshold-ratio "${PADVC_EVENT_THRESHOLD_RATIO}"
  --delta-mode "${PADVC_DELTA_MODE}"
  --text-dilate "${PADVC_TEXT_DILATE}"
  --ocr-backend "${PADVC_OCR_BACKEND}"
  --rapid-det-limit-side-len "${PADVC_RAPID_DET_LIMIT_SIDE_LEN}"
)
if [[ -n "${PADVC_RAPIDOCR_PACKAGE_PATH:-}" ]]; then
  padvc_common_args+=(--rapidocr-package-path "${PADVC_RAPIDOCR_PACKAGE_PATH}")
fi
if [[ "${PADVC_RAPID_USE_CLS:-0}" == "1" ]]; then
  padvc_common_args+=(--rapid-use-cls)
fi
if [[ "${PADVC_RAPID_NO_REC:-0}" == "1" ]]; then
  padvc_common_args+=(--rapid-no-rec)
fi
if [[ "${PADVC_STICKY_PEAK_RESCUE:-0}" == "1" ]]; then
  padvc_common_args+=(--sticky-peak-rescue)
fi

audit_extra_args=()
if [[ "${AUDIT_SAVE_IMAGES}" == "0" ]]; then
  audit_extra_args+=(--no-images)
fi

{
  echo "[START $(date '+%F %T')] label=${LABEL}"
  echo "host=$(hostname)"
  echo "repo_root=${REPO_ROOT}"
  echo "input_dir=${INPUT_DIR}"
  echo "run_dir=${RUN_DIR}"
  echo "task_manifest=${TASK_MANIFEST}"
  echo "sample_jsonl=${SAMPLE_JSONL}"
  echo "padvc_params=${PADVC_PARAMS}"
  echo "td_params=${TD_PARAMS}"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  echo "manim=$(${MANIM_BIN} --version 2>&1 | head -n 1)"
  echo "render_workers=${RENDER_WORKERS} audit_workers=${AUDIT_WORKERS} padvc_shards=${PADVC_SHARDS} td_workers=${TD_WORKERS}"
  echo "padvc_ocr_backend=${PADVC_OCR_BACKEND} padvc_p=${PADVC_P} padvc_delta_mode=${PADVC_DELTA_MODE}"
} | tee -a "${RUN_LOG}"

cd "${REPO_ROOT}"

echo "[STEP] render begin $(date '+%F %T')" | tee -a "${RUN_LOG}"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${RENDER_SCRIPT}" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${VIDEO_DIR}" \
  --results-json "${RENDER_RESULTS}" \
  --workers "${RENDER_WORKERS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --manim-bin "${MANIM_BIN}" \
  --timeout-sec "${RENDER_TIMEOUT_SEC}" \
  >> "${LOG_DIR}/render.log" 2>&1
echo "[STEP] render end $(date '+%F %T')" | tee -a "${RUN_LOG}"

echo "[STEP] audit begin $(date '+%F %T')" | tee -a "${RUN_LOG}"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${AUDIT_SCRIPT}" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${AUDIT_DIR}" \
  --workers "${AUDIT_WORKERS}" \
  --timeout-sec "${AUDIT_TIMEOUT_SEC}" \
  --save-interval "${SAVE_INTERVAL}" \
  "${audit_extra_args[@]}" \
  >> "${LOG_DIR}/audit.log" 2>&1
echo "[STEP] audit end $(date '+%F %T')" | tee -a "${RUN_LOG}"

echo "[STEP] padvc shards begin $(date '+%F %T')" | tee -a "${RUN_LOG}"
declare -a pids=()
declare -a shard_names=()
for (( shard=0; shard<PADVC_SHARDS; shard++ )); do
  shard_name=$(printf 'shard_%02d' "${shard}")
  shard_dir="${PADVC_SHARD_ROOT}/${shard_name}"
  shard_log="${LOG_DIR}/${shard_name}.log"
  mkdir -p "${shard_dir}"
  PYTHONUNBUFFERED=1 PADVC_OCR_CACHE_DIR="${OCR_CACHE_DIR}" "${PYTHON_BIN}" -u "${SCORE_PADVC_SCRIPT}" \
    --task-manifest "${TASK_MANIFEST}" \
    --audit-results "${AUDIT_RESULTS}" \
    --render-results "${RENDER_RESULTS}" \
    --output-dir "${shard_dir}" \
    --sample-jsonl "${SAMPLE_JSONL}" \
    --ocr-cache-dir "${OCR_CACHE_DIR}" \
    --num-shards "${PADVC_SHARDS}" \
    --shard-index "${shard}" \
    --quiet-padvc \
    --skip-existing \
    "${padvc_common_args[@]}" \
    > "${shard_log}" 2>&1 &
  pids+=("$!")
  shard_names+=("${shard_name}")
done

padvc_status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  shard_name="${shard_names[$idx]}"
  if wait "${pid}"; then
    rc=0
  else
    rc=$?
  fi
  echo "[STEP] ${shard_name} exit=${rc} $(date '+%F %T')" | tee -a "${RUN_LOG}"
  if [[ ${rc} -ne 0 ]]; then
    padvc_status=${rc}
  fi
done
if [[ ${padvc_status} -ne 0 ]]; then
  echo "[STEP] padvc shards failed exit=${padvc_status}" | tee -a "${RUN_LOG}"
  exit "${padvc_status}"
fi
echo "[STEP] padvc shards end $(date '+%F %T')" | tee -a "${RUN_LOG}"

echo "[STEP] merge padvc begin $(date '+%F %T')" | tee -a "${RUN_LOG}"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${MERGE_PADVC_SCRIPT}" \
  --input-dir "${PADVC_SHARD_ROOT}" \
  --output-dir "${PADVC_FINAL_DIR}" \
  >> "${LOG_DIR}/merge_padvc.log" 2>&1
echo "[STEP] merge padvc end $(date '+%F %T')" | tee -a "${RUN_LOG}"

echo "[STEP] td begin $(date '+%F %T')" | tee -a "${RUN_LOG}"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${SCORE_TD_SCRIPT}" \
  --input-jsonl "${PADVC_FINAL_DIR}/padvc_scores.jsonl" \
  --output-dir "${TD_DIR}" \
  --params-json "${TD_PARAMS}" \
  --workers "${TD_WORKERS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --skip-existing \
  >> "${LOG_DIR}/td.log" 2>&1
echo "[STEP] td end $(date '+%F %T')" | tee -a "${RUN_LOG}"

GENERATION_DIR="${GENERATION_DIR:-}"
if [[ -z "${GENERATION_DIR}" && "$(basename "${INPUT_DIR}")" == "cleaned_scripts" ]]; then
  candidate_dir="$(cd "${INPUT_DIR}/.." && pwd)"
  if [[ -f "${candidate_dir}/results.json" && -d "${candidate_dir}/prompt_snapshots" ]]; then
    GENERATION_DIR="${candidate_dir}"
  fi
fi
if [[ -n "${GENERATION_DIR}" ]]; then
  echo "[STEP] text expansion begin $(date '+%F %T') generation_dir=${GENERATION_DIR}" | tee -a "${RUN_LOG}"
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${TEXT_EXPANSION_SCRIPT}" \
    --generation-dir "${GENERATION_DIR}" \
    --skip-analysis-refresh \
    >> "${LOG_DIR}/text_expansion.log" 2>&1
  echo "[STEP] text expansion end $(date '+%F %T')" | tee -a "${RUN_LOG}"
else
  echo "[STEP] text expansion skipped: set GENERATION_DIR or pass a cleaned_scripts input under a generation run" | tee -a "${RUN_LOG}"
fi

echo "[DONE $(date '+%F %T')] label=${LABEL}" | tee -a "${RUN_LOG}"
