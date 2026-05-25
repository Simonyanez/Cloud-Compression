#!/bin/bash

# Usage: ./bin/run_vanishing_tests.sh [start_c] [end_c] [block_size] [sample_frac] [max_iters] [mode] [qsteps...]
# Example: ./bin/run_vanishing_tests.sh 1 8 8 0.005 20 draft 12 24 44 64

START_C=${1:-1}
END_C=${2:-8}
BLOCK_SIZE=${3:-8}
SAMPLE_FRAC=${4:-0.005}
MAX_ITERS=${5:-20}
MODE=${6:-"draft"}

# Handle QSTEPS: if parameters remain, use them, otherwise default
shift 6
if [ $# -gt 0 ]; then
    QSTEPS=$@
else
    QSTEPS="12 24 44 64"
fi

# Get the project root directory
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$ROOT_DIR"

echo "=== Starting Multi-Q-Step Vanishing Cluster Tests ($START_C to $END_C clusters) ==="
echo "[*] Block Size:        $BLOCK_SIZE"
echo "[*] Optimization Mode: ${MODE^^}"
echo "[*] Sample Fraction:   $SAMPLE_FRAC"
echo "[*] Max Iterations:    $MAX_ITERS"
echo "[*] Q-Steps:           $QSTEPS"

for (( C=$START_C; C<=$END_C; C++ ))
do
    echo ""
    echo "--------------------------------------------------------"
    echo "[*] RUNNING TEST SUITE FOR C = $C CLUSTERS"
    echo "--------------------------------------------------------"
    
    python3 scripts/test_vanishing_clusters.py \
        --clusters "$C" \
        --block_size "$BLOCK_SIZE" \
        --sample_frac "$SAMPLE_FRAC" \
        --max_iters "$MAX_ITERS" \
        --mode "$MODE" \
        --qsteps $QSTEPS
    
    if [ $? -ne 0 ]; then
        echo "[!] Error occurred during suite for $C clusters. Continuing..."
    fi
done

echo ""
echo "=== All suites completed! Check the timestamped 'results' folder for output. ==="
