#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_NAME=""
PREV_ARG=""
for ARG in "$@"; do
    if [[ "${PREV_ARG}" == "--output-name" ]]; then
        OUTPUT_NAME="${ARG}"
    fi
    if [[ "${ARG}" == --output-name=* ]]; then
        OUTPUT_NAME="${ARG#--output-name=}"
    fi
    PREV_ARG="${ARG}"
done

if [[ -z "${OUTPUT_NAME}" ]]; then
    echo "缺少 --output-name 参数。"
    exit 1
fi

MODEL_OUTPUT_DIR="${PROJECT_DIR}/output/model"
mkdir -p "${MODEL_OUTPUT_DIR}"

LOG_PATH="${ADL_LOG_PATH:-${MODEL_OUTPUT_DIR}/${OUTPUT_NAME}.adl.log}"
DETAILED_REPORT_PATH="${ADL_DETAILED_REPORT_PATH:-${MODEL_OUTPUT_DIR}/${OUTPUT_NAME}.adl.json}"
mkdir -p "$(dirname "${LOG_PATH}")" "$(dirname "${DETAILED_REPORT_PATH}")"

cd "${PROJECT_DIR}"
export TOKENIZERS_PARALLELISM=false

echo "开始执行 ADL TraceDR 训练"
echo "项目目录: ${PROJECT_DIR}"
echo "日志路径: ${LOG_PATH}"
echo "详细报告: ${DETAILED_REPORT_PATH}"

# 目的：先在远端实例内对齐依赖，再进入统一训练入口，避免 ADL 新机器缺包。
uv sync
uv run rerank-tracedr-train-adl \
    --detailed-report-path "${DETAILED_REPORT_PATH}" \
    --log-path "${LOG_PATH}" \
    "$@" 2>&1 | tee "${LOG_PATH}"
