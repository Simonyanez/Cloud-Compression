#!/usr/bin/env bash
# Step 3 (E10 diagnostic deep-dive plan): densify the Q-step grid for each sequence's own best
# (B_c*, K=24) config from tab:e10_bd_summary_multiframe. Reuses the already-trained/frozen
# codebook (skip-if-in-DB logic in exp_e10_multisequence_benchmark.py) -- pure new-Q encoding,
# no retraining. Same stride=50 as the existing K=24 data, so new points land on the same frames.
set -uo pipefail
cd /home/simao/Documents/Repositories/Cloud-Compression
PY=/home/simao/.pyenv/versions/.pcadc_venv/bin/python3
LOG=experiments/e10_q_densification.log
DB=results/exp_e10_multisequence.db
NEW_Q="16,20,28,40"

# name:B_c (K=24 for all, from tab:e10_bd_summary_multiframe)
CONFIGS=(
  "longdress:16" "loot:16" "redandblack:16" "soldier:16"
  "andrew10:8" "david10:16" "phil10:16" "ricardo10:16" "sarah10:16"
  "andrew9:8" "david9:16" "phil9:16" "ricardo9:16" "sarah9:16"
)

echo "[$(date -Is)] Starting Q densification (Q=$NEW_Q) for 14 sequences at their K=24 best B_c." >> "$LOG"
for entry in "${CONFIGS[@]}"; do
    IFS=":" read -r name bc <<< "$entry"
    echo "[$(date -Is)] === $name  B_c=$bc  K=24  Q=$NEW_Q ===" >> "$LOG"
    "$PY" scripts/exp_e10_multisequence_benchmark.py \
        --db-path "$DB" \
        --sequences "$name" --child-sizes "$bc" --k-values 24 \
        --stride 50 --q-values "$NEW_Q" \
        >> "$LOG" 2>&1
done
echo "[$(date -Is)] Q densification: all 14 sequences complete." >> "$LOG"
