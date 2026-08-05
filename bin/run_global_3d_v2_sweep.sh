#!/usr/bin/env bash
# ============================================================
#  Global 3D Projection v2 Sweep
#  Fixes: magnitude normalisation, live label update,
#         boundary-aware grid search, degenerate projection guard
# ============================================================
set -euo pipefail

REPO="/home/simao/Documents/Repositories/Cloud-Compression"
VENV="${REPO}/.venv"
SCRIPT="${REPO}/scripts/run_global_3d_projection_v2.py"

# --- Sweep hyperparameters (edit as needed) ---
BLOCK_SIZES="${1:-16,32}"
K_VALUES="${2:-4,8,12,16}"
GAMMAS="${3:-250,500,750,1000,1500,2000,2500,3000,3500,4000,5000,7500}"
Q_STEP="${4:-24}"
RD_ITERS="${5:-5}"
DEGEN_THRESH="${6:-0.001}"

# --- Output paths ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="${REPO}/results/v2_sweep_${TIMESTAMP}"
REPORT_FILE="${OUT_DIR}/global_3d_v2_results.md"
LOG_FILE="${OUT_DIR}/run.log"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo " Global 3D Projection v2 Sweep"
echo " Timestamp   : $(date)"
echo " Script      : ${SCRIPT}"
echo " Block sizes : ${BLOCK_SIZES}"
echo " K values    : ${K_VALUES}"
echo " Gamma sweep : ${GAMMAS}"
echo " Q step      : ${Q_STEP}"
echo " R-D iters   : ${RD_ITERS}"
echo " Degen thresh: ${DEGEN_THRESH}"
echo " Output dir  : ${OUT_DIR}"
echo " Report file : ${REPORT_FILE}"
echo " Log file    : ${LOG_FILE}"
echo "============================================================"

source "${VENV}/bin/activate"

python "${SCRIPT}" \
    --block-sizes  "${BLOCK_SIZES}" \
    --k-values     "${K_VALUES}" \
    --gammas       "${GAMMAS}" \
    --q-step       "${Q_STEP}" \
    --rd-iters     "${RD_ITERS}" \
    --degen-thresh "${DEGEN_THRESH}" \
    --output-dir   "${OUT_DIR}" \
    --output-name  "global_3d_v2_results.md" \
    2>&1 | tee "${LOG_FILE}"

echo "============================================================"
echo " Results Report : ${REPORT_FILE}"
echo " Execution Log  : ${LOG_FILE}"
echo "============================================================"
