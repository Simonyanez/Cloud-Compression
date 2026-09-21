# Briefing for Fable 5.1 — PCADC Thesis Final Review

Purpose of this document: give a future Claude Fable 5.1 session everything it needs to review
this thesis end-to-end **without spending tokens re-deriving repo structure, re-finding figures,
or re-establishing terminology**. Attach this file first, then the chapter `.tex` files you want
reviewed. Everything below was verified by direct inspection of the repo on 2026-09-18 (paths,
line numbers, DB schema, image inventory) — not recalled from memory.

---

## 1. Where things are

### 1.1 Thesis source
- Root: `docs/thesis/`. Entry point `main.tex` — `\input`s chapters in order:
  `introduccion.tex` (932 words) → `marco_teorico.tex` (6,682w) → `estado_del_arte.tex` (5,218w)
  → `metodologia.tex` (11,002w) → `resultados.tex` (11,994w) → `conclusiones.tex` (1,148w).
  Body total ≈ 37,000 words (~48–55K tokens raw). `\bibliography{library}` at `main.tex:126`.
- `resultados.tex` is by far the largest and most figure-dense chapter (the E10 multi-sequence
  benchmark results) — most of Fable's "does this figure match this claim" work will center here.
- Compile: `pdflatex` × 2 from `docs/thesis/` (bibtex has been run historically; `main.bbl` exists
  and is current relative to `library.bib`).

### 1.2 Code — the SMALL relevant set (out of ~130 scripts in `scripts/`)
The repo has a long tail of exploratory/legacy scripts from earlier experiment generations
(E1–E9, `oracle_*`, `run_*_sweep.py`, `test_*.py`, `absolute_brute_force_upper_bound.py`, etc.).
**Ignore all of those unless a specific figure caption points at one.** The scripts that actually
produced the current thesis content are:

| Script | Produces |
|---|---|
| `scripts/exp_e10_multisequence_benchmark.py` | The E10 benchmark itself — writes to `results/exp_e10_multisequence.db` and `results/exp_e10/bitstreams/`. This is the source of truth for all BPV/PSNR/BD-Rate numbers in `resultados.tex`. Contains `_worker_decode_block` / `_worker_decode_baseline` (real-decode primitives), `SEQUENCES`, `get_bp`, `make_pc`, `load_ply_from_zip`. |
| `scripts/generate_e10_thesis_figures.py` | Most `img/resultados/e10_*.png` figures (RD curves, k-sweep, overhead breakdown, driver correlations, adapter distribution, codebook directions, convergence frequency, compute time, adjacency z-score). |
| `scripts/tier2_block_diagnostics.py` | Block-level diagnostics behind `sec:e10_block_dependence`-area discussion; canonical reference for the "real decode, not proxy" methodology (`find_best_config_meta`, `load_codebook_and_adapters`, `_gft_two_modes`). Writes `scripts/_tier2_block_records.json` / `_tier2_summary.json`. |
| `scripts/generate_qualitative_comparison.py` | The 6 `e10_qualitative_*.png` figures (§Análisis Cualitativo). Q=32, K=24, frame_idx=1. Per-family camera `VIEW_CONFIG` dict (8iVFB vs. MVUB use different up-axes). Reads/writes `scripts/_qualitative_cache/*.npz`. |
| `scripts/exp_e9_multi_entropy_suite.py` | Defines `decode_block_attributes` — the canonical real-decode function reused by both `tier2_block_diagnostics.py` and `generate_qualitative_comparison.py`. Only Y (luma) is ever GFT-coded; U,V are lossless passthrough. |
| `scripts/generate_annex_figures.py` | The `es_fig_annex*` figures cited in `estado_del_arte.tex`/`metodologia.tex` (spherical centroids, spatial layout, cost evolution, trellis/CABAC diagrams). |
| `scripts/oracle_real_rlgr_and_compaction.py` | The `oracle_*` figures (oracle RD, sensitivity matrix, gain-by-blocksize, real-vs-proxy RD). |
| `docs/report/generate_report_figures_e10.py`, `docs/report/generate_inset_rd.py` | Source of the `es_fig_annex1`/`es_fig_annex2` pair specifically (separate small dir from `scripts/`, don't confuse the two). |

Core library (`src/pcadc/` or `pcadc/` importable module — used by all scripts above):
`blocks.py` (`Block`, `MortonBlockPartition`), `graph.py` (`StructuralGraph`, `AttributeGraph`),
`color.py` (`Colourist` — RGB↔YUV, `Approximator` — PCA tangent-plane fit).

### 1.3 Data
- **Primary result DB**: `results/exp_e10_multisequence.db` (8.5MB, last written 2026-09-17 — the
  authoritative source for every number in the E10 results section). Tables: `sequences` (14 rows
  — `andrew9/10, david9/10, sarah9/10, phil9/10, ricardo9/10` [MVUB] + `longdress, loot,
  redandblack, soldier` [8iVFB]), `aggregate_results` (the BPV/PSNR/BD-Rate table — keyed by
  `seq_id, child_b, K, q_step`, has `aggregate_bpv/psnr_Y`, `baseline_bpv/psnr_Y`, `bd_rate_pct`),
  `cluster_centroids`, `mode_adapters`, `kmeans_iterations`, `adapter_opt_steps`, `frame_results`.
  Query this directly instead of re-parsing any script's stdout if Fable needs to double-check a
  specific number.
- **Ablation DB**: `results/exp_e10_ablation_noselfloop.db` — the no-self-loop ablation referenced
  in the results chapter.
- **Bitstreams**: `results/exp_e10/bitstreams/<seq_name>/frame_XXXX/` (real encoded bitstreams,
  one dir per sequence per frame) and `results/exp_e10/block_parquet/` (per-block Parquet
  records). Note: **no bitstream reader exists in the repo** — all "decoded" figures/PSNR numbers
  come from forward-GFT-then-quantize (mathematically identical to a true round trip through the
  lossless entropy coder, since RLGR/arithmetic coding never lose information) — this is the
  established "real decode, not proxy" convention; if a figure caption or reviewer question
  implies an actual `.bin` read happened, that's a misunderstanding to correct, not a bug to find.
- **Small intermediate JSON caches** (all in `scripts/`, all reproducible from the DB + scripts
  above, safe for Fable to read directly instead of re-running anything):
  `_e10_thesis_table_data.json`, `_e10_bd_classic_tables.json`, `_e10_bd_ratio_tables.json`,
  `_e10_bd_multiframe_data.json`, `_e10_k_sweep` source data, `_e10_driver_data.json`,
  `_e10_adapter_dist_data.json`, `_e10_convergence_data.json`, `_e10_compute_time_data.json`,
  `_e10_cost_by_bpbck.json`, `_e10_info_diag_records.json`/`_summary.json`,
  `_tier2_block_records.json`/`_summary.json`, `_e10_rd_files.json`.
- `scripts/_qualitative_cache/*.npz` — cached decode results (points + colors + errors) for the
  6 qualitative-figure sequences at Q=24/32/64, keyed `<seq>_Q<Q>.npz`.

### 1.4 Figures — inventory and status
Full listing cross-checked against every `\includegraphics{...}` call in all 6 `.tex` chapters
(2026-09-18). **83 images are actually included in the compiled thesis; 98 exist on disk in
`img/resultados/` alone** — the ~30 unreferenced ones below are legacy/superseded, safe to ignore
unless Fable is specifically asked to audit for dead assets:

Unreferenced (not in any `\includegraphics` call, confirmed by direct grep):
`convergence_velocity.png`, `cost_evolution_B{4,8,16}.png`, `hamming_evolution_B{4,8,16}.png`,
`slp_bounds_B{4,8,16}.png`, `slp_split_B{4,8}.png`, `slw_bounds_B{4,8,16}.png`,
`slw_split_B{4,8}.png`, `gain_parameters_analysis.png`, `parameter_drifts.png`,
`efficiency_frontier.png`, `rd_curves_comparison.png`, `oracle_gain_by_blocksize.png`,
`oracle_real_vs_proxy_rd.png`, `e10_rd_longdress_bc32.png` (dropped when the RD-curves section was
redesigned to exclude B_c=32), `e10_block_dependence_graw.png`, `e10_block_dependence_npts.png`,
`e10_selfloop_compaction_mi.png`, `e10_temporal_stability.png` — these last 4 look current-generation
(`e10_` prefix) but are **not** embedded anywhere; worth asking the user whether they were meant
to replace something or are simply cut content.

Live figure families in `resultados.tex` (all confirmed embedded): `e10_rd_<seq>_bc{4,8,16}.png`
(42 images, the redesigned 7-page 2×3 RD-curve spreads), `e10_qualitative_{longdress,loot,phil9,
ricardo9,phil10,ricardo10}.png` (the 4-panel qualitative comparisons, Q=32), `e10_k_sweep.png`,
`e10_driver_correlations.png`, `e10_overhead_breakdown.png`, `e10_codebook_directions.png`,
`e10_adapter_distribution.png`, `e10_convergence_frequency.png`, `e10_adjacency_zscore.png`,
`e10_compute_time.png`, plus the `oracle_*` and `es_fig*`/`es_fig_annex*` families used in the
earlier chapters (Oracle/upper-bound discussion, method diagrams).

Pipeline/method diagrams (not code-generated, externally authored): `docs/thesis/img/metodología/
Full-Scheme.png`, `PCADC_Algorithm_{Hor,Ver}.png`, `PCADC_Algorithm.svg`, `Bloque_Ejemplo.png`,
`Grafo_Ejemplo.png`, `Norm_Ejemplo.png`, `PCADC_drawio_update.html` (source for the diagrams),
plus dataset thumbnails (`longdress_thumb.png`, etc. — one per sequence, externally sourced
cover-image crops, not decoded output).

---

## 2. Context Fable needs to make sense of the figures/argument

- **Pipeline vocabulary**: B_p/B_c = parent/child block side length in the octree partition
  (`B_p` fixed per depth via `get_bp(depth)`, `B_c` swept — the thesis tests 4/8/16, dropped 32 as
  uninformative). PCA tangent-plane extraction per block (`extract_block_tangent`) feeds a
  spherical k-means codebook (K clusters of tangent directions) → per-block mode assignment via a
  self-loop-augmented attribute graph (`AttributeGraph`) + gating/Viterbi selection → GFT
  (graph Fourier transform) of the Y (luma) channel only → quantize (step Q) → RLGR + order-1
  arithmetic coding. "Baseline" = mode-0 only (no self-loop, no adaptive mode selection) — the
  ablation floor every codec number is compared against.
- **Two dataset families with DIFFERENT coordinate conventions** — this tripped up the qualitative
  figures once already: 8iVFB sequences (`longdress`, `loot`, `redandblack`, `soldier`) have
  `V[:,1]` as the vertical/up axis; MVUB sequences (`andrew/david/sarah/phil/ricardo`, each at
  depth 9 and depth 10) have `V[:,2]` as vertical and `V[:,1]` as depth. If Fable is asked to
  reason about any 3D-scatter figure or camera angle, this distinction matters.
  The "6 representative sequences" used for the qualitative section are `longdress`, `loot` (8iVFB)
  + `phil9`, `ricardo9`, `phil10`, `ricardo10` (MVUB, same two subjects at both octree depths).
- **BD-Rate / BD-PSNR**: standard Bjøntegaard-delta convention — % bitrate savings / dB gain at
  matched quality/rate, integrated over the RD curve, not a single-point comparison. This is why
  qualitative-figure PSNR gaps can look tiny (~0.1–0.2 dB at a single fixed Q) while the RD-curve
  section reports meaningful BD-Rate gains — **both are true simultaneously and this is explained
  explicitly in the qualitative section's intro paragraph**; don't flag the small single-Q PSNR
  gap as a contradiction without re-reading that paragraph first.
- **Q=32 for the qualitative figures specifically**: chosen (Q=24 too fine → invisible artifacts;
  Q=64 too coarse → codec/baseline gap itself collapses) as the point where compression artifacts
  are visible AND the codec's advantage over baseline is near its max. This reasoning is stated in
  the section's intro paragraph in `resultados.tex`; if Fable proposes changing Q again, it should
  re-derive this trade-off, not assume more compression is always more illustrative.
- **LaTeX authoring conventions in this repo**: `%[OBSERVACIÓN: ...]` = an inline author note
  flagging something for later attention (not user-facing); `% [NOTA: texto redactado por IA]`
  = marks a passage drafted by an LLM, worth a closer clarity/register pass; `\paragraph{}` used as
  a sub-subsection-style grouping header within a `\subsection` (not a true LaTeX paragraph lead-in)
  — this repo's `\clearpage` placement convention is `\clearpage` **before** a `\paragraph{}` that
  starts a new figure spread, never after (an earlier bug here caused orphaned headings — see §4).
  Multi-panel figures are always pre-composed into one PNG via matplotlib subplots and included as
  a single `\includegraphics`, never assembled as a LaTeX-side grid — there is a known, previously
  reproduced-and-fixed `multicols`-inside-`figure[H]` ordering bug in this LaTeX setup; if Fable
  proposes any LaTeX-side multi-image grid, flag that risk.
- **Remaining genuinely-unstarted content**: `resultados.tex` around `\section{Experimentos
  Propuestos}` → `\subsection{Comparación con el Estándar G-PCC (RAHT)}` carries a
  `% Estado: NO INICIADO — requiere ejecutar TMC13 sobre las mismas secuencias` marker (line ~1283
  as of 2026-09-18, will drift with edits — search for `NO INICIADO` rather than trusting the line
  number) — this is real future work, not a stale note. A handful of other `[PLACEHOLDER]`-looking
  comments earlier in the same file (search `PLACEHOLDER`) are **stale leftover authoring notes**
  next to sections that are actually fully written with real data (Pearson correlation matrix,
  K-sweep, B_c scale analysis, overhead breakdown, Morton-adjacency mode-sharing) — worth a cleanup
  pass to delete the dead comments, but not a content gap.

---

## 3. Known issues to flag or fix (found during this briefing pass, not yet corrected)

1. **Two `\ref{}`/`\cite{}` typos** (real bug, trivial fix, not a missing-citation problem):
   - `introduccion.tex:48` — `...independiente de la señal\ref{zhang2014point}` should be
     `\cite{zhang2014point}`.
   - `marco_teorico.tex:132` — `...si minimiza $L(\gamma)$\ref{goyal2001theoretical}.` should be
     `\cite{goyal2001theoretical}`.
   - Confirmed genuine typos, not absent bibliography entries: both keys are correctly defined in
     `library.bib` and correctly used with `\cite{}` elsewhere in the same or other chapters
     (`marco_teorico.tex:79,111`; `estado_del_arte.tex:178,435`). This is what produced the
     "Reference X undefined" (not "Citation X undefined") pdflatex warnings — the wording
     distinguishes a dangling `\ref` from a dangling `\cite`. One-character-class fix each; doesn't
     need Fable's reasoning, just flagging so it gets fixed before or independent of the Fable pass.

2. **Six genuinely missing bibliography entries** (real content gap, needs real citations added,
   confirmed absent from BOTH `library.bib` and `Exported Items.bib`): `miller1955note`
   (`marco_teorico.tex:59`, Miller-Madow bias correction), `graziosi2020overview`
   (`estado_del_arte.tex:20,91,138`, G-PCC/video-codec-reuse discussion), `dequeiroz2016raht`
   (`estado_del_arte.tex:32,75,434`, RAHT), `quach2019learning`, `sheng2021deeppcac`,
   `fang2022_3dac` (all three at `estado_del_arte.tex:275` + `sheng2021deeppcac` again at line 437
   — the deep-learning PCAC related-work paragraph). All six are used correctly with `\cite{}` (not
   a typo) — the entries themselves are simply missing. `library.bib` has 15 entries;
   `Exported Items.bib` has 9, all of which turn out to be duplicates of existing `library.bib`
   entries under different key-naming conventions (e.g. `zhang_point_2014` vs. `zhang2014point`) —
   it does **not** contain any of the 6 missing keys, so it's not a case of "forgot to merge the
   second file." These 6 need real bibliography entries sourced and added to `library.bib`.

3. **Dangling section reference**: `\ref{sec:e10_block_dependence}` is used three times
   (`resultados.tex:1090,1093,1182`) but no `\label{sec:e10_block_dependence}` exists anywhere in
   the document — this resolves to `??` in the compiled PDF / triggers an undefined-reference
   warning. Likely the labeled section was renamed or restructured and lost its label. Needs a
   `\label{sec:e10_block_dependence}` added at the intended section (probably the block-level
   diagnostics discussion this same subsection is building toward, around `e10_adjacency_zscore` /
   the driver-correlations discussion — confirm the right anchor point before adding the label).

4. **`fig:enter-label`** used as a literal, un-renamed auto-generated label twice
   (`metodologia.tex:211,333`) — harmless (LaTeX only warns on genuinely duplicate labels within
   the same document, and grep shows these are two distinct instances of the same generic string,
   so may or may not actually collide depending on how many total unrenamed labels exist) but worth
   renaming to something descriptive for maintainability; low priority.

---

## 4. Priority task: deepen the analysis with already-computed data, then draft the unfinished conclusions

This is the primary deliverable for this Fable session — the audit tasks in §5 below are
secondary, worth folding in only if budget/context allows.

**What "conclusiones.tex" currently looks like** (checked 2026-09-18, 129 lines): `§Síntesis de
Resultados`, `§De RD-Clustering a la Metodología Final`, and `§Limitaciones y Honestidad sobre lo
que Falta` are all **fully written already**, with real numbers and section cross-references —
don't touch these except where task (a) below finds something to strengthen. Two sections are
genuinely incomplete and explicitly marked:
- `§Trabajo Futuro` (line 101) — literal placeholder text: `[POR COMPLETAR: priorizar entre
  E11--E15 según el tiempo disponible...]`.
- `§Reflexión Final` (line 109) — literal placeholder: `[POR COMPLETAR: esta sección es
  deliberadamente tuya.]`, followed by 3 explicit open questions (how strongly to claim
  "competitive with the state of the art" given E12/RAHT was never run; how to frame the
  RD-Clustering → final-methodology pivot to the committee; whether any numeric/scope commitment
  from the original thesis proposal needs to be referenced) that a prior Claude session
  deliberately left unanswered because they depend on the student's own judgment, voice, and
  relationship with their committee — **not** facts derivable from the data.

**Task (a) — deepen the analysis using data that's already computed, no new experiments:**
Point Fable at the JSON caches and DB tables in §1.3 above (not just the prose in `resultados.tex`)
and ask it to find 2–3 places where the underlying data supports a more specific or more
interesting claim than what's currently narrated — e.g. patterns in `_e10_driver_data.json` or
`_e10_adapter_dist_data.json` that are plotted but not fully discussed in text, per-sequence
detail in `aggregate_results` that's currently only summarized in aggregate. Hard constraint:
**only cite numbers that actually appear in the attached data, and name the exact source file/
table for each one** — this task is explicitly about using data that already exists, not about
running anything new.

**Task (b) — draft `§Trabajo Futuro`:** an evidence-based prioritization among the proposed future
experiments (E11–E15, defined in `resultados.tex`'s `\section{Experimentos Propuestos}`), with
reasoning grounded in what `§Limitaciones` already establishes as the biggest open gap. A prior
session's own read (left in the placeholder comment) was that E12 (RAHT/G-PCC comparison via
TMC13) matters most because it's the only path to an external-state-of-the-art claim — Fable
should form its own view from the evidence rather than just repeating that.

**Task (c) — draft `§Reflexión Final`:** treat this as **first-person draft language for the
student to edit and own, not a final ghostwritten section** — the file itself says this section is
deliberately the student's. For each of the 3 open questions, Fable should propose a defensible
position *grounded strictly in what the data supports* (e.g. on the "competitive with state of the
art" question, the honest claim without E12 is "consistently improves over its own structural
baseline across 14 real sequences with an interpretable, bounded-overhead adaptation mechanism" —
not "beats the state of the art"), clearly labeled as a suggestion, not asserted as the student's
actual opinion.

**Task (d) — expand the bibliography via web search, grounded in what's actually cited/claimed:**
Requires the `web_search` tool enabled for this Fable session (in Claude Code, `WebSearch` is
available automatically if Fable is run as the session model; in a plain API/chat session, this
needs the `web_search` tool declared explicitly — without it, Fable cannot do this task honestly
and should say so rather than inventing plausible-sounding citations). Sources to ground the search
in, so it isn't a blind literature search:
- The 6 already-confirmed missing keys (§3, item 2) — find the real papers for
  `miller1955note`, `graziosi2020overview`, `dequeiroz2016raht`, `quach2019learning`,
  `sheng2021deeppcac`, `fang2022_3dac` and add correctly-formatted `library.bib` entries.
- A broader pass over all 5 non-conclusions chapters for claims that currently have no citation but
  probably should (standard results stated as fact, comparisons to named methods, any "es sabido
  que..." / "estudios previos han mostrado..." phrasing without a `\cite{}` attached).
- `docs/report/main_es.pdf` (4,516-word Spanish "reporte técnico" — an earlier intermediate report
  for this same project, predates the thesis; referenced by name in `resultados.tex`'s commented-out
  `% \subsection{Figuras Extendidas del Reporte Técnico...}` block, lines ~1189–1250) — it has no
  bibliography of its own, but may reference methods/comparisons worth a real citation in the thesis.
- Hard constraint, same as everywhere else in this briefing: **never fabricate a citation** — every
  new `library.bib` entry must come from an actual web-search result (real title/authors/venue/year/
  DOI-or-URL), and if Fable can't find a confident match for a claim, it should flag the claim as
  needing a citation rather than inventing one.

**Task (e) — diagrams and methodology narrative:** Review the diagrams actually embedded in
`metodologia.tex` (only 5 of the 6 files in `img/metodología/` are used —
confirm current `\includegraphics` calls before assuming any list is current: `Full-Scheme.png`,
`PCADC_Algorithm_Hor.png`, `Bloque_Ejemplo.png`, `Grafo_Ejemplo.png`, `Norm_Ejemplo.png`;
`PCADC_Algorithm_Ver.png`/`.svg` exist on disk but are **not** currently included anywhere) against
the surrounding narrative text for accuracy, consistency of terminology with the rest of the
thesis, and clarity — flag any place where the diagram and the prose describe the pipeline
differently, and propose narrative rewrites directly.
- **Explicit scope limit (user instruction, 2026-09-18): the existing PNGs are not to be touched.**
  Do not edit, replace, or propose edits to `img/metodología/PCADC_drawio_update.html` (the draw.io
  source) or any existing `.png`/`.svg` in that folder — leave all of them exactly as they are, even
  if a narrative mismatch is found; fix the mismatch by rewriting the *text* to match the diagram,
  not the other way around, unless a new diagram is genuinely warranted (next bullet).
  The **only** diagram edits in scope are **new TikZ/PGF figures added directly in `metodologia.tex`**
  — either to illustrate something the current 5 diagrams don't cover, or to sit alongside an
  existing PNG as a supplementary figure. Compiles natively with `pdflatex`, no external renderer
  needed, and never touches the existing image files.
- **Only propose a new TikZ diagram where it's actually necessary or clearly adds explanatory
  value** — e.g. a step in the pipeline the prose describes but no current figure shows, or a
  concept that's genuinely hard to follow in text alone. This is not a "produce more diagrams"
  task — if the 5 existing PNGs already cover the pipeline adequately, the right output for Task
  (e) may just be narrative-text fixes with zero new figures, and that's a fine outcome.

**Task (f) — compile and verify:** you're running Fable as the model inside Claude Code (confirmed
2026-09-18), so it has real bash and file-write access to this repo — no attach-files step needed,
it should read files directly. After applying bibliography and/or narrative/TikZ changes, run
`pdflatex` twice from `docs/thesis/` (and `bibtex main` once if new `library.bib` entries were
added, before the second `pdflatex` pass) and confirm: no new "undefined reference"/"undefined
citation" warnings beyond what's already known (§3), no compile errors, and no unexpected page-count
swing. Since bibliography edits touch the same mechanism, also fix the two `\ref{}`/`\cite{}` typos
and the dangling `sec:e10_block_dependence` label from §3 while in there — one line each.

## 5. Candidate final-review tasks/questions for Fable

Rank-ordered by expected value for a $20 budget (see cost note below — recommend spending most of
it on task #1 as one consolidated pass with the others folded in as sub-questions, rather than
running each separately):

1. **Global numeric-claim consistency audit**: cross-check every BPV/PSNR/BD-Rate number quoted in
   prose (abstract, results narrative, conclusions) against the actual figures/tables it's citing —
   flag any mismatch, stale number left over from an earlier experiment run, or number quoted with
   the wrong sequence/B_c/K attached.
2. **Cross-reference/dangling-label audit**: every `\ref{}`/`\cite{}` in the document — flag
   undefined targets (should already catch the two known issues above, useful as a check that the
   audit methodology actually works), and any `\ref{}` that should be `\cite{}` or vice versa.
3. **Abstract vs. results consistency**: does the abstract's claimed contribution/results match what
   `resultados.tex`/`conclusiones.tex` actually show, including any hedges or scope limits added
   during the results chapter that the abstract doesn't reflect?
4. **Narrative coherence across chapters**: does `marco_teorico.tex` establish every concept
   `resultados.tex` relies on (self-loop graphs, spherical k-means, GFT modes, gating/Viterbi)
   before it's used? Does `estado_del_arte.tex`'s positioning of PCADC against RAHT/deep-learning
   methods match how `resultados.tex` and `conclusiones.tex` frame the comparison?
5. **Redundancy check**: any explanation, definition, or figure-description repeated near-verbatim
   across chapters that should be consolidated or cross-referenced instead.
6. **Academic writing / clarity pass, targeted at `[NOTA: texto redactado por IA]`-marked
   passages**: these are flagged as LLM-drafted and likely to benefit most from a register/clarity
   pass — search for that marker across all chapters first, review those passages specifically
   rather than the whole document uniformly.
7. **§Experimentos Propuestos review**: is the scope/framing of the still-unstarted G-PCC/RAHT
   comparison (`NO INICIADO` marker) well-motivated and clearly bounded as *future work*, not
   accidentally read as a claimed result?
8. **Bibliography formatting**: once §4 Task (d)'s new entries are added, verify every entry
   (old and new) has correct venue/year/author formatting consistent with the rest of `library.bib`.

**Budget note** (carried over from the earlier pricing discussion): thesis body is ~37K words
(~48–55K tokens raw text, well within Fable's 1M context even with this briefing doc, the JSON
data files, and full figure-caption text attached). At Fable 5.1 pricing ($10/1M in, $50/1M out),
spend the bulk of the $20 on §4 (the analysis-deepening + conclusions draft — this is generative,
output-token-heavy work, the expensive kind) at `effort: high`. If there's budget and context room
left, fold in §5's items 1, 3–7 (skip 2 and 8 — those are mechanical/bibliographic, already
resolved by direct grep in §3, not worth Fable's judgment) as secondary questions in the *same*
call rather than a separate one, since the context (chapters + data) is already loaded.

---

## 6. The actual call to make to Fable

Confirmed: you're running this inside **Claude Code**, switching the session model to Fable 5.1 —
so it has real bash, file-read/write, and web-search tool access to this repo directly. No file
attachment step needed; it reads `FABLE_BRIEFING.md` and everything else itself. Effort defaults to
adaptive/high inside Claude Code; no separate `effort` parameter to set.

Paste this as the first message:

> Read `docs/thesis/FABLE_BRIEFING.md` first — it maps out where everything is (code, figures,
> data, DB schema) and what you need to know about this codec (PCADC) and this repo's conventions,
> so you don't need to re-explore the repo from scratch. Then do these in order:
>
> 1. Read `resultados.tex`'s analysis sections alongside the JSON data files in `scripts/`
>    (`_e10_driver_data.json`, `_e10_adapter_dist_data.json`, `_e10_thesis_table_data.json`,
>    `_e10_bd_classic_tables.json`, `_e10_bd_ratio_tables.json`, `_tier2_summary.json`) and the
>    `results/exp_e10_multisequence.db` tables listed in the briefing §1.3. Find 2–3 places where
>    the underlying data supports a more specific or interesting claim than what's currently
>    written in prose (briefing §4, Task (a)). Only use numbers that actually appear in that data,
>    and cite the exact source file/table for every number — do not invent or estimate anything.
>    Show me each one as "current text → proposed text" before applying it.
> 2. Using that (plus the already-written §Síntesis de Resultados and §Limitaciones sections of
>    `conclusiones.tex`), write `§Trabajo Futuro`: an evidence-based prioritization among the
>    proposed future experiments E11–E15, with your own reasoning — don't just repeat the
>    placeholder comment's guess that E12 matters most, form your own view from the evidence.
> 3. Write `§Reflexión Final` as first-person language I can edit and own, not a final section. The
>    three open questions are in `conclusiones.tex` (line ~109) and briefing §4 Task (c). For each,
>    propose a defensible position grounded strictly in what the data supports, labeled clearly as
>    a suggestion — especially on how strongly to claim competitiveness with the state of the art,
>    since Experiment E12 (the only external-codec comparison) was never run.
> 4. Expand the bibliography (briefing §4 Task (d)) using real web search: find real sources for the
>    6 confirmed-missing keys (`miller1955note`, `graziosi2020overview`, `dequeiroz2016raht`,
>    `quach2019learning`, `sheng2021deeppcac`, `fang2022_3dac`), plus any other claim across the 5
>    chapters (and `docs/report/main_es.pdf` if relevant) that states something as fact without a
>    citation and should have one. Every new `library.bib` entry must come from an actual search
>    result — real title/authors/venue/year — never fabricated; if you can't find a confident
>    match, flag the claim instead of inventing a source.
> 5. Review the 5 diagrams actually used in `metodologia.tex` against the narrative around them
>    (briefing §4 Task (e)): flag any mismatch between what a diagram shows and what the text says.
>    **Do not edit or replace any existing `.png`/`.svg`, and do not touch
>    `PCADC_drawio_update.html`** — where the mismatch is real, fix it by rewriting the narrative
>    text to match the diagram. The only diagram edits in scope are **new TikZ/PGF figures added
>    directly in `metodologia.tex`**, and only where genuinely necessary or clearly useful for the
>    explanation (e.g. a pipeline step the prose covers but no current figure shows) — this is not
>    a "produce more diagrams" task; if the existing 5 already cover it, propose zero new figures.
> 6. After applying the bibliography, narrative, and any TikZ changes, run `pdflatex` twice from
>    `docs/thesis/` (`bibtex main` once in between if you added `library.bib` entries) and confirm
>    no new undefined-reference/citation warnings and no compile errors. While you're touching the
>    bibliography, also fix the two `\ref{}`/`\cite{}` typos and the dangling
>    `sec:e10_block_dependence` label from briefing §3 — same mechanism, one line each.
> 7. If you have budget and context room left after all that, also do items 1 and 3–7 from the
>    briefing's §5 candidate-task list (skip 2 and 8, already resolved). Keep each finding short and
>    cite `file:line`.
>
> Match the existing academic Spanish register throughout — don't shift tone. Flag anything you're
> not confident about instead of guessing, and never fabricate a number, a citation, or a compile
> result you didn't actually verify.

This orders the expensive generative work first (drafting real sections, sourcing real citations,
fixing real diagram/text mismatches), the mechanical fix-and-verify step next, and the cheaper
audit-style checks last — so if the pass runs long or the budget is tighter than expected, the
highest-value output lands before anything gets cut.

**How to actually start this**: since Fable is priced much higher per token than the model running
this session, start it in a **fresh Claude Code session** (new terminal `claude` invocation, or
`/clear` in this one) rather than continuing here — mid-session model switches keep this entire
conversation's history loaded, which would burn Fable-priced tokens re-reading work it doesn't need
(RD-curve layout iterations, qualitative-figure tuning, etc.). In the fresh session, run `/model`,
pick Fable 5.1 from the list, then paste the message above.
