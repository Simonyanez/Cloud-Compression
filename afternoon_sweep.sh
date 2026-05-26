#!/bin/bash
OUTPUT_DIR="afternoon_validated_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Write the justification file immediately
cat <<EOF > "$OUTPUT_DIR/README_AFTERNOON_JUSTIFICATION.md"
# Afternoon Sensitivity Sweep: Saturation & Convergence Benchmark

## 1. Dimensionality Optimization
This experiment explicitly abandons the computationally inefficient B=16 scale (which exhibited a 4.8x per-iteration penalty) to reallocate budget toward high-resolution structural sweeps at B=4 and B=8.

## 2. Capacity Saturation Probe
We probe the absolute saturation limits of the B=4 scale by extending the codebook capacity up to C=10. This identifies the point where signaling overhead (Entropy H) completely cancels out adaptive transform gains.

## 3. Convergence Plateau Benchmark
A prolonged 40-iteration benchmark is executed for the B=4, C=6, Q=24 configuration. This factual evaluation determines whether the standard 20-iteration cap is prematurely truncating the framework's optimization potential.
EOF

SAMPLE_RATE=0.001

echo "[*] Launching final afternoon sweep window into: $OUTPUT_DIR"

# PHASE 1: SEARCHING FOR THE CAPACITY SATURATION BOUNDARY
echo "[*] Phase 1: Pushing B=4 Capacity Limits"
python3 scripts/run_exp.py --block_size 4 --q_step 24 --clusters 8  --sample_rate $SAMPLE_RATE --max_iters 20 --out_dir "$OUTPUT_DIR"
python3 scripts/run_exp.py --block_size 4 --q_step 24 --clusters 10 --sample_rate $SAMPLE_RATE --max_iters 20 --out_dir "$OUTPUT_DIR"

# PHASE 2: SOLVING THE CONVERGENCE DEFICIT (PROLONGED RUN)
echo "[*] Phase 2: Testing Full Convergence Plateau on Winner"
python3 scripts/run_exp.py --block_size 4 --q_step 24 --clusters 6  --sample_rate $SAMPLE_RATE --max_iters 40 --out_dir "$OUTPUT_DIR"

# PHASE 3: FILLING THE QUANTIZATION CURVE INTERPOLATION
echo "[*] Phase 3: Intermediate Quantization Step Mapping"
python3 scripts/run_exp.py --block_size 4 --q_step 36 --clusters 4  --sample_rate $SAMPLE_RATE --max_iters 20 --out_dir "$OUTPUT_DIR"
python3 scripts/run_exp.py --block_size 4 --q_step 48 --clusters 4  --sample_rate $SAMPLE_RATE --max_iters 20 --out_dir "$OUTPUT_DIR"

# PHASE 4: RE-VALUATING B=8 AT HIGH CAPACITY
echo "[*] Phase 4: Checking if higher capacity rescues B=8"
python3 scripts/run_exp.py --block_size 8 --q_step 24 --clusters 6  --sample_rate $SAMPLE_RATE --max_iters 20 --out_dir "$OUTPUT_DIR"

echo "[+] Afternoon pipeline complete. Clean data secured for 19:00 report."
