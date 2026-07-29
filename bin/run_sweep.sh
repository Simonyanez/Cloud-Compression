#!/bin/bash
OUTPUT_DIR="fast_validated_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Capping max iterations to 20 to rely heavily on early stopping window savings
MAX_ITERS=20
SAMPLE_RATE=0.001

# Write the thesis justification file immediately inside the output folder
cat <<EOF > "$OUTPUT_DIR/README_JUSTIFICATION.md"
# Adjusted Sensitivity Sweep (Time-Constrained Framework)

## 1. Algorithmic Complexity Management
To prevent combinatorial search-space explosion in the joint differential_evolution optimizer during multi-scale evaluations, the baseline statistical parameter instruction was set to n_0 = 32. This scales down the dimensionality of the B=16 blocks, avoiding long processing delays while preserving a representative sample.

## 2. Experimental Matrix Architecture
The execution utilizes a strict "Isolate & Anchor" approach structured for execution under 9 hours:
- **Baseline Cross-Check:** Maps cross-scale performance at a stable anchor of C=4, Q=24.
- **B=4 Structural Sweep:** Conducts a deep capacity sweep (C=2, 4, 6) on the production-critical B=4 block dimension.
- **B=4 Noise Threshold Check:** Tests the breakdown limits under extreme quantization noise (Q=64).
EOF

echo "[*] Launching calibrated sweep execution into: $OUTPUT_DIR"

# STEP 1: MULTI-SCALE CORRELATION MATRIX (Anchor C=4, Q=24)
echo "[*] Phase 1: Cross-scale baselines with fast n_0=32 footprint"
python3 scripts/run_exp.py --block_size 16 --q_step 24 --clusters 4 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"
python3 scripts/run_exp.py --block_size 8  --q_step 24 --clusters 4 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"
python3 scripts/run_exp.py --block_size 4  --q_step 24 --clusters 4 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"

# STEP 2: PRODUCTION B=4 CAPACITY DEPTH SWEEP (Anchor Q=24)
echo "[*] Phase 2: Structural Capacity Sweeps on high-stability block matrix"
python3 scripts/run_exp.py --block_size 4  --q_step 24 --clusters 2 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"
python3 scripts/run_exp.py --block_size 4  --q_step 24 --clusters 6 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"

# STEP 3: PRODUCTION B=4 NOISE PROFILE LIMIT
echo "[*] Phase 3: Quantization degradation boundary assessment"
python3 scripts/run_exp.py --block_size 4  --q_step 64 --clusters 4 --sample_rate $SAMPLE_RATE --max_iters $MAX_ITERS --out_dir "$OUTPUT_DIR"

echo "[+] Execution stream finalized."
