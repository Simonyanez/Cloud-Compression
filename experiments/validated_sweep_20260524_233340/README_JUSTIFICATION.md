# Experiment Justification: Validated Sensitivity Sweep

## 1. Statistical Sampling Floor (Cochran)
This experiment replaces uniform percentage sampling with a finite-population Cochran statistical floor. 
By enforcing a dynamic minimum sample count based on confidence intervals (95% confidence, 10% MoE), we eliminate the "block-starvation" bias that caused crashes in large block sizes (B=16) during previous runs.

## 2. "Isolate & Anchor" Strategy
To narrow the sensitivity analysis, this sweep follows a dual-phase isolation:
- **Phase 1 (Capacity Sweep):** Anchors noise at Q=24 and sweeps cluster capacity ( \in \{2, 4, 6\}$) across all block sizes.
- **Phase 2 (Quantization Sweep):** Anchors capacity at C=4 and sweeps quantization noise ( \in \{36, 48, 64\}$) across all block sizes.

This matrix identifies whether adaptive gains are driven primarily by codebook size or structural resolution.
