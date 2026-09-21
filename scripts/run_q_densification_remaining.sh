#!/usr/bin/env bash
# Resume of run_q_densification.sh -- the original background run died silently after 8/14
# sequences (no OOM in dmesg, no reboot, no traceback in its log -- cause undetermined) while
# training sarah10's new-Q adapters. Re-run only the 6 sequences with zero new-Q coverage.
set -uo pipefail
cd /home/simao/Documents/Repositories/Cloud-Compression
PY=/home/simao/.pyenv/versions/.pcadc_venv/bin/python3
LOG=experiments/e10_q_densification.log
DB=results/exp_e10_multisequence.db
NEW_Q="16,20,28,40"

# name:B_c -- the 6 sequences with new_Q_coverage=0/4, K=24 for all
CONFIGS=(
  "sarah10:16"
  "andrew9:8" "david9:16" "phil9:16" "ricardo9:16" "sarah9:16"
)

echo "[$(date -Is)] RESUME: Q densification for the 6 remaining sequences." >> "$LOG"
for entry in "${CONFIGS[@]}"; do
    IFS=":" read -r name bc <<< "$entry"
    echo "[$(date -Is)] === $name  B_c=$bc  K=24  Q=$NEW_Q ===" >> "$LOG"
    "$PY" scripts/exp_e10_multisequence_benchmark.py \
        --db-path "$DB" \
        --sequences "$name" --child-sizes "$bc" --k-values 24 \
        --stride 50 --q-values "$NEW_Q" \
        >> "$LOG" 2>&1
    echo "[$(date -Is)] === $name exit code $? ===" >> "$LOG"
done
echo "[$(date -Is)] Q densification RESUME: all 6 remaining sequences complete." >> "$LOG"
