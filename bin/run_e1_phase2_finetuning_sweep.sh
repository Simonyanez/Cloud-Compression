#!/usr/bin/env bash
# ============================================================
#  E1 Gain-Filter Sweep — Phase 2: Fine-Tuning Sweep
#  Goal: Full tau & gamma RD curve generation for top (B, K)
# ============================================================
set -euo pipefail

REPO="/home/simao/Documents/Repositories/Cloud-Compression"
VENV="${REPO}/.venv"
SCRIPT="${REPO}/scripts/exp_e1_gain_filter.py"

# --- Phase 2 Hyperparameters (specify winning B and K from Phase 1) ---
BLOCK_SIZES="${1:-16 32}"
K_VALUES="${2:-6 8}"
TAU_RETRAIN="${3:-0.5,1.0,2.0,3.0,5.0}"
GAMMAS="${4:-500,1000,2000,4000,6000,8000}"
Q_STEP="${5:-24}"

# --- Output paths ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="${REPO}/results/e1_phase2_sweep_${TIMESTAMP}"
LOG_FILE="${OUT_DIR}/phase2_run.log"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo " E1 Gain-Filter Sweep — Phase 2: Fine-Tuning Sweep"
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
        echo " Phase 2 Fine-Tuning Run: B=${B}, K=${K}..." | tee -a "${LOG_FILE}"
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
echo " Phase 2 Fine-Tuning Sweep Completed!" | tee -a "${LOG_FILE}"
echo " Execution Log : ${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
