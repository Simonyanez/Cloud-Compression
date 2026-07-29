# Afternoon Sensitivity Sweep: Saturation & Convergence Benchmark

## 1. Dimensionality Optimization
This experiment explicitly abandons the computationally inefficient B=16 scale (which exhibited a 4.8x per-iteration penalty) to reallocate budget toward high-resolution structural sweeps at B=4 and B=8.

## 2. Capacity Saturation Probe
We probe the absolute saturation limits of the B=4 scale by extending the codebook capacity up to C=10. This identifies the point where signaling overhead (Entropy H) completely cancels out adaptive transform gains.

## 3. Convergence Plateau Benchmark
A prolonged 40-iteration benchmark is executed for the B=4, C=6, Q=24 configuration. This factual evaluation determines whether the standard 20-iteration cap is prematurely truncating the framework's optimization potential.
