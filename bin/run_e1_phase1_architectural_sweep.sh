#!/usr/bin/env bash
# ============================================================
#  E1 Gain-Filter Sweep — Phase 1: Architectural Search
#  Goal: Identify optimal (B, K) spatial-angular architecture
# ============================================================
set -euo pipefail

REPO="/home/simao/Documents/Repositories/Cloud-Compression"
VENV="${REPO}/.venv"
SCRIPT="${REPO}/scripts/exp_e1_gain_filter.py"

# --- Phase 1 Hyperparameters ---
BLOCK_SIZES="${1:-8 16 32}"
K_VALUES="${2:-4 6 8 10}"
TAU_RETRAIN="${3:-0.5,1.0}"
GAMMAS="${4:-1000,2000,4000}"
Q_STEP="${5:-24}"

# --- Output paths ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="${REPO}/results/e1_phase1_sweep_${TIMESTAMP}"
LOG_FILE="${OUT_DIR}/phase1_run.log"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo " E1 Gain-Filter Sweep — Phase 1: Architectural Search"
echo " Timestamp   : $(date)"
echo " Script      : ${SCRIPT}"
echo " Block sizes : ${BLOCK_SIZES}"
echo " K values    : ${K_VALUES}"
echo " Tau retrain : ${TAU_RETRAIN}"
echo " Gamma sweep : ${GAMMAS}"
echo " Q step      : ${Q_STEP}"
echo " Output dir  : ${OUT_DIR}"
echo " Log file    : ${LOG_FILE}"
echo "============================================================"

source "${VENV}/bin/activate"

for B in ${BLOCK_SIZES}; do
    for K in ${K_VALUES}; do
        echo "" | tee -a "${LOG_FILE}"
        echo "============================================================" | tee -a "${LOG_FILE}"
        echo " Phase 1 Run: B=${B}, K=${K}..." | tee -a "${LOG_FILE}"
        echo "============================================================" | tee -a "${LOG_FILE}"
        
        python "${SCRIPT}" \
            --block-size "${B}" \
            --k "${K}" \
            --q-step "${Q_STEP}" \
            --tau-retrain "${TAU_RETRAIN}" \
            --gammas "${GAMMAS}" \
            2>&1 | tee -a "${LOG_FILE}"
    done
done

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo " Phase 1 Architectural Sweep Completed!" | tee -a "${LOG_FILE}"
echo " Execution Log : ${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
