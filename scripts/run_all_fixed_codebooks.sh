#!/bin/bash

SWEEP_FOLDER="production_sweep_20260526_123142"
OUT_DIR="fixed_codebook_full_results"

# Block sizes and Cluster counts from the sweep
BLOCK_SIZES=(4 8 16)
CLUSTERS=(1 2 4 6 8 10 12)

echo "=========================================================="
echo "   STARTING FULL FIXED CODEBOOK PIPELINE EVALUATION"
echo "   Source: $SWEEP_FOLDER"
echo "   Target: $OUT_DIR"
echo "=========================================================="

for B in "${BLOCK_SIZES[@]}"
do
    for C in "${CLUSTERS[@]}"
    do
        # Check if codebook exists for this B, C at Q24
        CODEBOOK="$SWEEP_FOLDER/result_B${B}_C${C}_Q24.json"
        
        if [ -f "$CODEBOOK" ]; then
            echo ""
            echo "[*] TARGET: B=$B | C=$C"
            echo "----------------------------------------------------------"
            
            python3 scripts/test_fixed_codebook.py \
                --block_size "$B" \
                --clusters "$C" \
                --codebook_q 24 \
                --sweep_folder "$SWEEP_FOLDER" \
                --out_dir "$OUT_DIR"
                
            if [ $? -ne 0 ]; then
                echo "[!] Error in B${B}_C${C}. Skipping to next..."
            fi
        else
            echo "[?] Skipping B${B}_C${C} (No Q24 codebook found)"
        fi
    done
done

echo "=========================================================="
echo "   FULL PIPELINE EVALUATION COMPLETED"
echo "=========================================================="
