#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 <input_dir> <run_dir> <task_manifest> <audit_results> <sample_jsonl> <norm_params> <label>" >&2
  exit 1
fi

INPUT_DIR="$1"
RUN_DIR="$2"
TASK_MANIFEST="$3"
AUDIT_RESULTS="$4"
SAMPLE_JSONL="$5"
NORM_PARAMS="$6"
LABEL="$7"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIM_BIN="${MANIM_BIN:-manim}"
RENDER_SCRIPT="${RENDER_SCRIPT:-${SCRIPT_DIR}/render_directory.py}"
SCORE_SCRIPT="${SCORE_SCRIPT:-${SCRIPT_DIR}/score_padvc.py}"
MERGE_SCRIPT="${MERGE_SCRIPT:-${SCRIPT_DIR}/merge_padvc_shards.py}"
RENDER_WORKERS="${RENDER_WORKERS:-8}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
RENDER_TIMEOUT_SEC="${RENDER_TIMEOUT_SEC:-600}"

VIDEO_DIR="${RUN_DIR}/videos"
RENDER_RESULTS="${RUN_DIR}/results.json"
OCR_CACHE_DIR="${RUN_DIR}/ocr_cache"
SHARD_ROOT="${RUN_DIR}/padvc_shards"
FINAL_DIR="${RUN_DIR}/padvc_final"
LOG_DIR="${RUN_DIR}/logs"
RUN_LOG="${RUN_DIR}/run.log"

mkdir -p "${RUN_DIR}" "${VIDEO_DIR}" "${OCR_CACHE_DIR}" "${SHARD_ROOT}" "${FINAL_DIR}" "${LOG_DIR}"

{
  echo "[START $(date '+%F %T')] label=${LABEL}"
  echo "host=$(hostname)"
  echo "pwd=$(pwd)"
  echo "input_dir=${INPUT_DIR}"
  echo "run_dir=${RUN_DIR}"
  echo "task_manifest=${TASK_MANIFEST}"
  echo "audit_results=${AUDIT_RESULTS}"
  echo "sample_jsonl=${SAMPLE_JSONL}"
  echo "norm_params=${NORM_PARAMS}"
  echo "python=$(${PYTHON_BIN} --version 2>&1)"
  echo "manim=$(${MANIM_BIN} --version 2>&1 | head -n 1)"
  echo "render_workers=${RENDER_WORKERS}"
  echo "num_shards=${NUM_SHARDS}"
  echo "save_interval=${SAVE_INTERVAL}"
  echo "render_timeout_sec=${RENDER_TIMEOUT_SEC}"
} | tee -a "${RUN_LOG}"

cd "${REPO_ROOT}"

echo "[STEP] render_only begin $(date '+%F %T')" | tee -a "${RUN_LOG}"
set +e
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${RENDER_SCRIPT}" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${VIDEO_DIR}" \
  --results-json "${RENDER_RESULTS}" \
  --workers "${RENDER_WORKERS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --manim-bin "${MANIM_BIN}" \
  --timeout-sec "${RENDER_TIMEOUT_SEC}" \
  >> "${LOG_DIR}/render.log" 2>&1
render_status=$?
set -e
echo "[STEP] render_only end $(date '+%F %T') exit=${render_status}" | tee -a "${RUN_LOG}"
if [[ ${render_status} -ne 0 ]]; then
  exit "${render_status}"
fi

echo "[STEP] padvc_shards begin $(date '+%F %T')" | tee -a "${RUN_LOG}"
declare -a pids=()
declare -a shard_ids=()
for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  shard_name=$(printf 'shard_%02d' "${shard}")
  shard_dir="${SHARD_ROOT}/${shard_name}"
  shard_log="${LOG_DIR}/${shard_name}.log"
  mkdir -p "${shard_dir}"
  PYTHONUNBUFFERED=1 PADVC_OCR_CACHE_DIR="${OCR_CACHE_DIR}" "${PYTHON_BIN}" -u "${SCORE_SCRIPT}" \
    --task-manifest "${TASK_MANIFEST}" \
    --audit-results "${AUDIT_RESULTS}" \
    --render-results "${RENDER_RESULTS}" \
    --output-dir "${shard_dir}" \
    --sample-jsonl "${SAMPLE_JSONL}" \
    --norm-params "${NORM_PARAMS}" \
    --ocr-cache-dir "${OCR_CACHE_DIR}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard}" \
    --quiet-padvc \
    --skip-existing \
    > "${shard_log}" 2>&1 &
  pids+=("$!")
  shard_ids+=("${shard_name}")
done

set +e
padvc_status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  shard_name="${shard_ids[$idx]}"
  wait "${pid}"
  rc=$?
  echo "[STEP] ${shard_name} exit=${rc} $(date '+%F %T')" | tee -a "${RUN_LOG}"
  if [[ ${rc} -ne 0 ]]; then
    padvc_status=${rc}
  fi
done
set -e

if [[ ${padvc_status} -ne 0 ]]; then
  echo "[STEP] padvc_shards failed exit=${padvc_status}" | tee -a "${RUN_LOG}"
  exit "${padvc_status}"
fi
echo "[STEP] padvc_shards end $(date '+%F %T')" | tee -a "${RUN_LOG}"

echo "[STEP] merge begin $(date '+%F %T')" | tee -a "${RUN_LOG}"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${MERGE_SCRIPT}" \
  --input-dir "${SHARD_ROOT}" \
  --output-dir "${FINAL_DIR}" \
  >> "${LOG_DIR}/merge.log" 2>&1
echo "[STEP] merge end $(date '+%F %T')" | tee -a "${RUN_LOG}"

echo "[DONE $(date '+%F %T')] label=${LABEL}" | tee -a "${RUN_LOG}"
