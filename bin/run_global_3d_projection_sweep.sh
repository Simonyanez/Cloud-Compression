#!/usr/bin/env bash
#
# Comprehensive Final Experiment Sweep: Global 3D Codebook Projection
#
# Full search matrix:
#   Block sizes : 4, 8, 16, 32
#   K values    : 3, 4, 6, 8, 12, 16
#   Gammas      : 250 to 7500 (12 fine-grained steps)
#   R-D Iters   : 5 (deep centroid refinement)
#   Q-Step      : 24

set -euo pipefail

# Get directory of the running script (fallback to pwd if run interactively)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${0}}")" 2>/dev/null && pwd)"

# If SCRIPT_DIR turns out to be root / or empty, default to current working directory
if [ -z "${SCRIPT_DIR}" ] || [ "${SCRIPT_DIR}" = "/" ]; then
    PROJECT_ROOT="$(pwd)"
else
    # Assuming script is in <PROJECT_ROOT>/scripts or <PROJECT_ROOT>
    if [ -f "${SCRIPT_DIR}/config/base_config.yaml" ]; then
        PROJECT_ROOT="${SCRIPT_DIR}"
    else
        PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." 2>/dev/null && pwd)"
    fi
fi

export PROJECT_ROOT
echo "PROJECT_ROOT: ${PROJECT_ROOT}"

# Python entrypoint script
PY_SCRIPT="${PY_SCRIPT:-${PROJECT_ROOT}/scripts/run_global_3d_projection_fixed.py}"

# Experiment Hyperparameters (Overridable via environment variables)
BLOCK_SIZES="${BLOCK_SIZES:-4,8,16,32}"
K_VALUES="${K_VALUES:-3,4,6,8,12,16}"
GAMMAS="${GAMMAS:-250,500,750,1000,1500,2000,2500,3000,3500,4000,5000,7500}"
Q_STEP="${Q_STEP:-24}"
RD_ITERS="${RD_ITERS:-5}"
SAMPLE_RATIO="${SAMPLE_RATIO:-0.01}"
N_STRATA="${N_STRATA:-5}"

# Directories & Output Files
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/final_sweep_${TIMESTAMP}}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/config/base_config.yaml}"
OUTPUT_NAME="${OUTPUT_NAME:-global_3d_projection_final_results.md}"
LOG_FILE="${OUTPUT_DIR}/experiment_run.log"

# Setup Output Directory
mkdir -p "${OUTPUT_DIR}"

# Track total execution time
SECONDS=0

# Log pipeline output simultaneously to stdout and logfile
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo " Starting Final Comprehensive Experiment Sweep"
echo " Timestamp    : $(date)"
echo " Python Script: ${PY_SCRIPT}"
echo " Config Path  : ${CONFIG_PATH}"
echo " Output Dir   : ${OUTPUT_DIR}"
echo " Report File  : ${OUTPUT_DIR}/${OUTPUT_NAME}"
echo " Log File     : ${LOG_FILE}"
echo "------------------------------------------------------------"
echo " Hyperparameters:"
echo "   Block Sizes    : ${BLOCK_SIZES}"
echo "   K Values       : ${K_VALUES}"
echo "   Gamma0 Sweep   : ${GAMMAS}"
echo "   Quant Step     : ${Q_STEP}"
echo "   R-D Refine     : ${RD_ITERS} iterations"
echo "   Sample Ratio   : ${SAMPLE_RATIO} (${N_STRATA} strata)"
echo "============================================================"

# Pre-flight check
if [ ! -f "${PY_SCRIPT}" ]; then
    echo "ERROR: Python script not found at ${PY_SCRIPT}" >&2
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Configuration file not found at ${CONFIG_PATH}" >&2
    exit 1
fi

# Execute full Python sweep pass
python3 "${PY_SCRIPT}" \
    --block-sizes "${BLOCK_SIZES}" \
    --k-values "${K_VALUES}" \
    --gammas "${GAMMAS}" \
    --q-step "${Q_STEP}" \
    --rd-iters "${RD_ITERS}" \
    --sample-ratio "${SAMPLE_RATIO}" \
    --n-strata "${N_STRATA}" \
    --config "${CONFIG_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-name "${OUTPUT_NAME}"

ELAPSED=$SECONDS
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECS=$((ELAPSED % 60))

echo ""
echo "============================================================"
echo " Experiment completed successfully!"
echo " Total Elapsed Time: ${HOURS}h ${MINUTES}m ${SECS}s"
echo " Results Report written to: ${OUTPUT_DIR}/${OUTPUT_NAME}"
echo " Execution Log written to:   ${LOG_FILE}"
echo "============================================================"

# ------------------------------------------------------------------
# Optional: Isolated Block-by-Block Execution Mode
#
# If working with massive point clouds or limited memory, run each 
# block size as an isolated subprocess. Uncomment below to enable.
# ------------------------------------------------------------------
# IFS=',' read -ra B_ARRAY <<< "${BLOCK_SIZES}"
# for BSIZE in "${B_ARRAY[@]}"; do
#     echo ">>> Processing Block Size B=${BSIZE} <<<"
#     python3 "${PY_SCRIPT}" \
#         --block-sizes "${BSIZE}" \
#         --k-values "${K_VALUES}" \
#         --gammas "${GAMMAS}" \
#         --q-step "${Q_STEP}" \
#         --rd-iters "${RD_ITERS}" \
#         --sample-ratio "${SAMPLE_RATIO}" \
#         --n-strata "${N_STRATA}" \
#         --config "${CONFIG_PATH}" \
#         --output-dir "${OUTPUT_DIR}" \
#         --output-name "global_3d_projection_B${BSIZE}_results.md"
# done
