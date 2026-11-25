# PCADC and RA-GFT: actual architectural relationship

Not part of the compiled thesis — a working note answering "is PCADC compatible with RA-GFT
somehow," grounded in what's actually in this repository, written alongside the RAHT baseline
experiment (see `scripts/exp_raht_baseline.py`, `src/pcadc/raht.py`) on
`feature/raht-baseline-comparison`. It also serves as groundwork for a deferred, later RA-GFT
baseline benchmark (see the plan/PR description for that branch) — this note is the "what would
that even mean" answer that benchmark will build on.

## The relationship is closer than "conceptually related work" — it's shared lineage

`estado_del_arte.tex` cites RA-GFT (Pavez, Girault, Ortega, Chou, ICIP 2020) as related work
PCADC positions itself against. That's true, but incomplete: PCADC's own code has a direct,
traceable history with RA-GFT's own reference implementation, not just a citation relationship.

- `matlab_code/*.m` (28 files) is a full, vendored, working RA-GFT reference implementation —
  `Region_Adaptive_GFT.m` / `iRegion_Adaptive_GFT.m` (forward/inverse transform),
  `encoder_RAGFT_RLGR_large_blocks.m` (a complete encoder with RLGR entropy coding and a
  quantization/PSNR sweep), `compute_GFT.m`, `bf_graph.m`, `w2l.m`, plus octree/Morton/PLY I/O
  helpers. `matlab_code/readme.md` credits it directly to the RA-GFT authors and cites the same
  ICIP 2020 paper.
- `src/utils/color.py:59-60` — the exact RGB↔YUV conversion matrices `src/pcadc/color.py`'s
  `Colourist` still uses today carry the comment *"Functions below translated from MATLAB to
  Python from https://github.com/STAC-USC/RA-GFT"* — i.e. PCADC's own color-conversion code is
  a direct port of RA-GFT's.
- Git history shows the same lineage for the structural machinery: `f1ce6a2` ("First Translation
  to Python") ported the MATLAB RA-GFT code (including a `Region_Adaptive_GFT.py` that no longer
  exists under that name); `7ae8e60` ("First Refactoring Effort") and `a7187cb` ("Reorganizing
  and Testing Python Functions") progressively reshaped that port into what `src/pcadc/`
  actually is now (`blocks.py`, `graph.py`, `transforms.py`, `decider.py`).

So the honest framing isn't "PCADC and RA-GFT are two independent, comparable methods." It's:
**PCADC started as a Python port of RA-GFT and diverged from there.** The comparison worth
making is not "are they compatible" in the abstract, but "what did PCADC keep from that lineage,
and what did it replace."

## What's shared today

- **Voxel/block partitioning conventions and PLY/Morton-order I/O** — same basic data model.
- **The YUV color transform** — verbatim, as noted above.
- **The graph-Laplacian machinery in spirit**: `StructuralGraph`'s inverse-distance-weighted
  adjacency and `ConnectedGFTProcessor`'s `L = D - A` eigendecomposition are the same family of
  operation RA-GFT's `compute_GFT.m`/`bf_graph.m` perform — a graph built from geometry, then
  eigendecomposed for a per-region transform basis.

## Where PCADC actually diverged, and why that's the real design question

RA-GFT's defining property — the one the thesis's own comparison table
(`tab:comparativo_metodos`) doesn't fully capture — is that it's **multiresolution and
hierarchical**: it builds a family of nested partitions (a tree over regions), computes a
region-local transform at each level, and **promotes a single low-pass (DC) coefficient up to
the next-coarser level**, exactly the same "propagate DC, terminate AC" structure this
session's RAHT implementation (`src/pcadc/raht.py`) uses, just with a full graph-Laplacian
transform at each node instead of a Haar-style weighted merge.

**PCADC does not have this.** Its two-level $B_p$/$B_c$ hierarchy (`MortonBlockPartition`
called separately at two block sizes, glued together externally) signals one mode decision per
parent and inherits it to children — but it never computes a transform at the parent level and
propagates a DC term down/up between levels the way RA-GFT does. Each child block's GFT is
self-contained; nothing about it is derived from, or feeds into, a coarser-level transform.
PCADC's real contribution — the directional codebook, self-loops, and signal-adaptive mode
selection via gating/Viterbi — was built entirely on top of the *flat*, single-level GFT it kept
from the RA-GFT port, not on the multiresolution structure RA-GFT actually contributes.

This is exactly the same "principal limitation" this session's RAHT experiment surfaced
independently and for an unrelated reason (see `results/raht_baseline_comparison.json`): PCADC's
one-decision-per-parent-block design doesn't share information *across* scales the way a real
hierarchical transform (RA-GFT or RAHT) does. Two different investigations this session — one
comparing against RAHT numerically, one tracing RA-GFT's code lineage — point at the same gap.

## What integrating RA-GFT's structure into PCADC would concretely require

Not attempted here (see the deferred-work note on this experiment's branch) — but concretely,
not abstractly, it would mean:

1. Extending `MortonBlockPartition`'s two-level $B_p$/$B_c$ split into a genuine nested tree
   (more than two levels), the way RA-GFT's own partition hierarchy works.
2. Computing a `ConnectedGFTProcessor`-style transform *at every level* of that tree, not just at
   the child-block leaf level as today, with the DC coefficient of each level's transform
   becoming an input to the next-coarser level's transform — a structural change to
   `transforms.py`'s data flow, not just a new `GFTProcessorStrategy` subclass.
3. Deciding where PCADC's own contribution — self-loops keyed to a directional codebook — would
   attach in that new structure: applied at every level independently (most faithful to today's
   design, but multiplies the codebook-training and per-level RDO cost), or only at the finest
   level (cheaper, but then the coarser levels are un-adapted RA-GFT, and the self-loop mechanism
   never sees the redundancy the hierarchy is designed to exploit).

Item 3 is the real open design question, and it's the same shape of question this session's
Tier-2 RAHT+PCADC discussion (see the branch's plan) already surfaced for RAHT: **PCADC's
adaptation mechanism was built for a flat transform, and attaching it to any hierarchical one —
RA-GFT or RAHT — is genuinely new design work, not a drop-in extension.**
