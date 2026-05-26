#!/bin/bash

# Master Timestamp Only
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="sweep_results_$TIMESTAMP"

# Single Flattened Destination
mkdir -p "$OUTPUT_DIR"

echo "[*] Sensitivity Analysis Starting..."
echo "[*] Results will be saved to: $OUTPUT_DIR"

# Global Script Variables
SAMPLE_RATE=0.001
MAX_ITERS=30
FRAME_REF="reference_frame.raw"

# Sequential Execution Matrix (10 Exploits)

# Phase 1: B=8 Baseline and Boundary Sweep
echo "[*] Phase 1: B=8 Baseline and Boundary Sweep"
python3 scripts/run_exp.py --block_size 8 --q_step 24 --clusters 1 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"
python3 scripts/run_exp.py --block_size 8 --q_step 24 --clusters 4 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"
python3 scripts/run_exp.py --block_size 8 --q_step 24 --clusters 8 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"
python3 scripts/run_exp.py --block_size 8 --q_step 64 --clusters 4 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"

# Phase 2: B=4 Cross-Comparison
echo "[*] Phase 2: B=4 Cross-Comparison"
python3 scripts/run_exp.py --block_size 4 --q_step 24 --clusters 1 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"
python3 scripts/run_exp.py --block_size 4 --q_step 24 --clusters 4 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"
python3 scripts/run_exp.py --block_size 4 --q_step 24 --clusters 8 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"

# Phase 3: B=16 Cross-Comparison
echo "[*] Phase 3: B=16 Cross-Comparison"
python3 scripts/run_exp.py --block_size 16 --q_step 24 --clusters 1 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"
python3 scripts/run_exp.py --block_size 16 --q_step 24 --clusters 4 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"
python3 scripts/run_exp.py --block_size 16 --q_step 24 --clusters 8 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"

echo "[+] Sensitivity Analysis Suite Completed."
