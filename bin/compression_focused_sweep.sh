#!/bin/bash

# COMPRESSION-FOCUSED RD-CLUSTERING SWEEP
# Prioritizing Large blocks (B16, B32) and Spatial Regularization.

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="compression_sweep_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

# CONSTANTS
SAMPLE_RATE=0.02   # 2% sample (Higher for B32 to ensure enough blocks)
MAX_ITERS=5       # Slightly reduced for B32 speed
Q_STEP=24

echo "=========================================================="
echo "    STARTING COMPRESSION-FOCUSED PRODUCTION SWEEP"
echo "    Target: $OUTPUT_DIR"
echo "=========================================================="

for B in 16 32
do
    for C in 2 4
    do
        # Lambda Proportional (1.0 = Balanced, 5.0 = Heavy Compression)
        for L in 1.0 5.0
        do
            # Beta Penalty (500 = Moderate, 2000 = Strong RLE-style)
            for BETA in 500 2000
            do
                echo ""
                echo "[*] RUNNING: B=$B | C=$C | L=$L | Beta=$BETA"
                echo "----------------------------------------------------------"
                
                python3 scripts/run_exp_v2.py \
                    --block_size "$B" \
                    --q_step "$Q_STEP" \
                    --clusters "$C" \
                    --lambda_prop "$L" \
                    --beta "$BETA" \
                    --sample_rate "$SAMPLE_RATE" \
                    --max_iters "$MAX_ITERS" \
                    --out_dir "$OUTPUT_DIR" \
                    --skip-existing
                
                if [ $? -ne 0 ]; then
                    echo "[!] ERROR in B$B C$C L$L Beta$BETA" >> "$OUTPUT_DIR/error_log.txt"
                fi
            done
        done
    done
done

echo "=========================================================="
echo "    COMPRESSION SWEEP COMPLETED SUCCESSFULLY"
echo "=========================================================="
