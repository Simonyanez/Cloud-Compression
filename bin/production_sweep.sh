#!/bin/bash

# PRODUCTION-GRADE RD-CLUSTERING SENSITIVITY SWEEP
# Master sequence for full Rate-Distortion and Capacity Analysis.
# Total experiments: 126 (3 Resolution x 6 Quantization x 7 Capacity)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESUME_DIR=$1

if [ -n "$RESUME_DIR" ]; then
    if [ ! -d "$RESUME_DIR" ]; then
        echo "[!] Error: Resume directory '$RESUME_DIR' does not exist."
        exit 1
    fi
    OUTPUT_DIR="$RESUME_DIR"
    SKIP_FLAG="--skip-existing"
    echo "[*] RESUMING SWEEP in: $OUTPUT_DIR"
else
    OUTPUT_DIR="production_sweep_$TIMESTAMP"
    SKIP_FLAG=""
    mkdir -p "$OUTPUT_DIR"
    echo "[*] STARTING NEW SWEEP in: $OUTPUT_DIR"
fi

# CONTEXTUAL PARAMS
SAMPLE_RATE=0.005  # 0.5% sample (will be overridden by Cochran floor if too small)
MAX_ITERS=40       # Increased limit for absolute convergence
MODE="production"  # Enable polish and high population

echo "=========================================================================="
echo "    STARTING PRODUCTION GRADE RD-CLUSTERING SWEEP"
echo "    Target: $OUTPUT_DIR"
echo "=========================================================================="

# 1. Block Sizes (Resolution)
for B in 4 8 16
do
    # 2. Quantization Steps (Noise)
    for Q in 12 24 36 44 48 64
    do
        # 3. Cluster Sizes (Capacity)
        for C in 1 2 4 6 8 10 12
        do
            echo ""
            echo "[*] RUNNING: B=$B | Q=$Q | C=$C"
            echo "----------------------------------------------------------"
            
            python3 scripts/run_exp.py \
                --block_size "$B" \
                --q_step "$Q" \
                --clusters "$C" \
                --sample_rate "$SAMPLE_RATE" \
                --max_iters "$MAX_ITERS" \
                --mode "$MODE" \
                --out_dir "$OUTPUT_DIR" \
                $SKIP_FLAG
            
            # Error check
            if [ $? -ne 0 ]; then
                echo "[!] CRITICAL ERROR in run B$B_C$C_Q$Q. Moving to next..." >> "$OUTPUT_DIR/error_log.txt"
            fi
        done
    done
done

echo "=========================================================================="
echo "    PRODUCTION SWEEP COMPLETED SUCCESSFULLY"
echo "=========================================================================="
