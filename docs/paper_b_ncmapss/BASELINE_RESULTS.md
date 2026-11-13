# N-CMAPSS baseline ladder — four subsets, development and sealed test

Status: **complete**. Development 480 rows (240 cells × 2 evaluation points) at
`git_sha = b889bd2` under protocol blob `a3c40c63…` (13th generation). Sealed test
80 rows (40 cells × 2) at `git_sha = ad53371`, opened under protocol blob
`6ec1c01…` (15th generation, which adds §6.10 defining `epoch*` for the
full-device test model). Every cell carries a checkpoint.

The two batches legitimately carry different blobs: the test model trains on all
six development engines, so it has no inner split, and §6.10 had to exist before
the test set was opened. It was committed at 04:00 and the set opened at 04:05
(+222 s, mechanically verified from the opens record). The development rows keep
`a3c40c63` permanently — `protocol_blob` is a per-row field, so no later anchor
can rewrite them.

The test units were opened once, after the freeze, and their numbers are
**descriptive**: they were never used to pick a model, a stopping rule or a
configuration. Where test and development disagree, both are reported.

## The ladder

Four rungs, from trivial to the frozen DSPT configuration. The ladder exists to
answer one question — *how much of any improvement is just the cycle axis?* —
because that is what the C-MAPSS audit found to dominate there.

| rung | model | inputs | purpose |
|---|---|---|---|
| **N0** | age-only | cumulative cycle only | the floor. If a structural model cannot beat this, nothing else matters |
| **N1** | vanilla Transformer | flight statistics + cycle | same family as the C-MAPSS `VAN_*` arms |
| **N2** | DSPT Base | same as N1, DSPT architecture | isolates the architecture term |
| **N3** | frozen DSPT | input layer adapted only; MS/SA switches frozen at the C-MAPSS values | the transfer claim |

## Rules

1. **N3 is frozen.** Its MS / SA / HI switches are the C-MAPSS values, fixed
   before any result exists. Outcomes may not be used to re-decide them.
2. All rungs use the identical split, the identical folds and the identical
   metric. Deviations are recorded as deviations.
3. Every rung is reported per fold and per engine with dispersion. A single mean
   is not a result.
4. N0 must be reported even though it is uninteresting: it is the reference that
   says how much of the performance is the cycle axis.

## Test results (main table)

Sealed test units, `fixed_passes:200`, three-seed mean (N0: one seed, no seed
axis). Prediction instant = end of flight; label = raw RUL (the 125 cap is
inactive on all four subsets, `max(Y)` = 62–88).

| subset | N0 | N1 | N2 | N3 | test engines | flights |
|---|---:|---:|---:|---:|---:|---:|
| DS02 | **9.31** | 25.85 | 14.61 | 18.26 | 3 | 202 |
| DS05 | **7.96** | 32.66 | 14.48 | 12.35 | 4 | 327 |
| DS06 | **7.86** | 25.33 | 7.89 | 10.59 | 4 | 322 |
| DS07 | **9.84** | 24.10 | 11.36 | 11.45 | 4 | 344 |

Same cells at `inner_val` (`epoch*`):

| subset | N0 | N1 | N2 | N3 |
|---|---:|---:|---:|---:|
| DS02 | **8.91** | 23.98 | 20.91 | 23.15 |
| DS05 | **7.81** | 24.81 | 13.98 | 15.16 |
| DS06 | **7.86** | 23.78 | 8.95 | 9.68 |
| DS07 | **10.49** | 25.73 | 14.88 | 16.54 |

**Age-only is first in all eight subset × evaluation-point combinations**, not by
a margin that a reader can wave away: on DS05 the clock is 7.96 against 14.48 for
the best structural rung, a factor of 1.8.

## Paired contrasts on the test set (15 engines)

Computed per engine from `pred.csv`, not from ledger aggregates. Positive means
the first rung is worse. All p-values below were re-derived here as exact
two-sided sign tests and reproduce the reported figures.

| contrast | mean, `fixed` | agree | p | mean, `inner_val` | agree | p |
|---|---:|---:|---:|---:|---:|---:|
| N1 − N0 | +17.70 | 13/15 | 7.4e−3 | +16.69 | **15/15** | 6.1e−5 |
| N2 − N0 | +3.66 | 11/15 | 0.118 | +5.79 | 12/15 | **0.035** |
| N3 − N0 | +4.52 | 11/15 | 0.118 | +7.34 | 12/15 | **0.035** |
| N3 − N2 | +0.87 | 8/15 | **1.00** | +1.55 | 10/15 | 0.302 |
| N1 − N2 | +14.04 | 14/15 | 9.8e−4 | — | — | — |

**N3 − N2 at p = 1.00 is the exact value, not a rounding**: with 8 of 15 folds
one way and 7 the other, the two-sided sign test is precisely 1. The frozen
MS+SA configuration is not merely "not significant" against DSPT-Base — its sign
cannot be separated from a coin.

## Development results (internal evidence, LOEO)

Six LOEO folds per subset, `fixed_passes:200`, per-cell mean over three seeds:

| subset | N0 | N1 | N2 | N3 |
|---|---:|---:|---:|---:|
| DS02 | **9.01** | 26.32 | 10.66 | 13.53 |
| DS05 | **13.96** | 22.62 | 15.95 | 17.24 |
| DS06 | **11.24** | 22.48 | 13.88 | 14.01 |
| DS07 | **6.50** | 23.35 | 13.92 | 11.72 |

At `inner_val` (DS02 recomputed from the sealed ledger; the other three from the
batch summary): DS02 **8.91** / 22.23 / 11.45 / 18.65; DS05 14.08 / 22.62 /
**14.03** / 17.24 (N2 ahead of N0 by 0.05); DS06 and DS07 as in the test-style
pattern — N0 first.

Pooled over 24 folds:

| contrast | mean, `fixed` | agree | p | mean, `inner_val` | agree | p |
|---|---:|---:|---:|---:|---:|---:|
| N1 − N0 | +13.52 | 23/24 | 3.0e−6 | +13.29 | **24/24** | 1.2e−7 |
| N2 − N0 | +3.43 | 18/24 | **0.023** | +3.59 | 16/24 | 0.152 |
| N3 − N0 | +3.95 | 17/24 | 0.064 | +6.41 | 16/24 | 0.152 |
| N3 − N2 | +0.52 | 13/24 | 0.839 | +2.82 | 15/24 | 0.307 |

## What holds across development and test, and what does not

The two levels agree on **direction** for every contrast, which is the point of
running both. They do not agree on **magnitude**, and the disagreement is not
uniform:

| subset | dev N2 − N0 | test N2 − N0 |
|---|---:|---:|
| DS02 | +1.65 | +5.30 |
| DS05 | +1.99 | +6.52 |
| DS06 | +2.64 | **+0.03** |
| DS07 | +7.42 | +1.52 |

The pooled means (+3.43 dev, +3.66 test) look like agreement, but they are means
over a range that spans **0.03 to 6.52** — a factor of 200. On DS06 the gap
vanishes on test; on DS02 and DS05 it triples. **The pooled agreement is not
evidence that the dev→test shift is one mechanism**; it is evidence that the sign
is stable while the size is set by the subset. Any claim of the form "the gap is
about 3.5" is a statement about the average of four very different numbers.

The one thing the two levels agree on without qualification is that **no
structural rung beats the clock in any subset, at either evaluation point.**

## The mechanism: the clock follows lifetime distance

Age-only predicts from the cycle index alone, so its error on a held-out engine
is set by how far that engine's lifetime sits from the mean lifetime of its
training engines. Over all 24 development folds, **Spearman ρ = 0.983** between
per-fold error and signed lifetime distance, with error spanning 1.47 to 22.04
and distance spanning 0.0 to 23.8. The four worst folds are all at
|distance| > 20; the four best are all at |distance| < 1.5.

The clock is not *learning* the mapping. Its error is a readout of a property of
the fold that is fixed before training starts. That is the quantitative form of
D043's "six engines are only enough for limited-scope development evidence", and
it is why every number here is reported per fold and per engine.

The same mechanism reappears on the test set as **bias direction**: DS02's test
engines are shorter-lived than its development engines (58/75/66 against 62–88),
and N0 overestimates by +6.48 there; DS07 runs the other way at −7.37.

## Pre-registered falsification check

These three conditions were written into this file before any result existed.

1. **"N3 does not beat N1 by more than the cross-fold spread"** — **not
   triggered.** N3 beats N1 in all four subsets at both evaluation points.
2. **"N0 alone matches N2/N3, repeating the C-MAPSS saturation finding"** —
   **triggered, and more strongly than on any single subset.** N0 is first in all
   eight test combinations. By the rule written in advance, the architecture
   contribution is **not measurable on these development sets**, the same
   saturation the C-MAPSS audit found.
3. **"N3's advantage disappears once the cycle axis is removed from every
   rung"** — **not tested.** No cycle-ablated rung was run; N0 is the cycle-*only*
   reference, not its complement.

## What this does not show

- **Not "architecture is useless".** The supported statement is: under this
  protocol, at this scale, against this baseline set, **an age-only clock cannot
  be beaten**, and the frozen MS+SA configuration has no measurable effect. That
  is a statement about a measurement protocol, not a universal claim about
  structure.
- **No attribution of the vanilla gap.** N1 and N2 differ in three ways at once —
  input representation, condition embedding, dual-span auxiliary loss. Only the
  condition embedding was isolated (`N1c`, diagnostic). The other two remain
  unattributed.
- **No claim that DSPT is harmful.** The test-set contrasts that reach p<0.05
  (N2−N0 and N3−N0 at `inner_val`) are not matched at `fixed_passes:200`
  (p=0.118), so the two columns disagree about significance while agreeing on
  direction. Both are reported; neither is selected.
- **No cross-subset comparison of absolute levels.** Each subset has its own
  lifetime distribution and fold difficulty. The pooled tests pool *directions*,
  not magnitudes.

## Provenance

- Development: `experiments/ncmapss/<subset>/runs_dev.csv`, `git_sha = b889bd2`,
  `protocol_blob = a3c40c63…`. The DS02 column was recomputed here from the
  sealed `registry/runs_DS02.csv`.
- Test: `experiments/ncmapss/<subset>/runs_test.csv`, `git_sha = ad53371`,
  `protocol_blob = 6ec1c01…` (15th generation; §6.10 defines the test model's
  `epoch*` as `round_half_up` of the six LOEO values, frozen before opening).
  `code_md5` differs from the development manifest because the test path did not
  exist at that freeze; the diff is `load_split(..., fitted=)` and
  `fit_rung(..., no_holdout=True)` only — loss, optimiser, hyper-parameters,
  architecture and label construction are unchanged (recorded in
  `REGISTRY_NOTES` v20).
- Both evaluation points come from **one training run** with two weight
  snapshots; the 240 development pairs share no checkpoint sha256.
- `DS02_N0_TEST_f200 = 9.31` matches an earlier code-validation run at 9.310.
  That is **the same point appearing twice, not independent evidence.**
