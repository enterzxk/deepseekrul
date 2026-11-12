# N-CMAPSS baseline ladder — four subsets

Status: **development results complete** for DS02, DS05, DS06, DS07 —
480 rows (240 cells × 2 evaluation points), 480 checkpoints, no missing cell,
produced by `b889bd2` under protocol blob `a3c40c63…`. The official test units
remain sealed; no test result exists anywhere in this file.

The DS02 column of every table below was recomputed here from the sealed ledger
`registry/runs_DS02.csv` (120 rows, 64 columns) and matches what was published
before. The DS05/06/07 columns come from the batch summary and are **provisional
until their ledgers are sealed** — see "Provenance" at the end.

## The ladder

Four rungs, from trivial to the frozen DSPT configuration. The ladder exists to
answer one question — *how much of any improvement is just the cycle axis?* —
because that is what the C-MAPSS audit found to dominate there.

| rung | model | inputs | purpose |
|---|---|---|---|
| **N0** | age-only | cumulative cycle only | the floor. If a structural model cannot beat this, nothing else matters |
| **N1** | vanilla Transformer | flight statistics + cycle | same family as the C-MAPSS `VAN_*` arms |
| **N2** | DSPT Base | same as N1, DSPT architecture | isolates the architecture term |
| **N3** | frozen DSPT | input layer adapted only | the transfer claim |

## Rules

1. **N3 is frozen.** Its MS / SA / HI switches are the C-MAPSS values, fixed
   before any result exists. Outcomes may not be used to re-decide them. If N3
   underperforms, that is a finding about transfer, not a licence to retune.
2. All four rungs use the identical split, the identical LOEO folds and the
   identical metric. Any deviation is recorded as a deviation, not silently.
3. Every rung is reported per fold with dispersion. A single mean is not a
   result.
4. N0 must be reported even though it is uninteresting: it is the reference that
   tells the reader how much of the performance is the cycle axis.

## Metrics

- Primary: per-unit RMSE at the flight-level prediction instant, then the mean
  and standard deviation over the LOEO folds.
- Secondary: MAE, |Bias| and a NASA-style asymmetric score. All are already in
  the ledger (`mae_raw_unit`, `bias_raw_unit`, `nasa_raw_unit`, `seg_*`).
- Every table states: prediction instant (end of flight), label convention
  (raw RUL; the 125 cap is inactive on all four subsets, `max(Y)` = 62–88), and
  the number of units.

## Results — six-fold mean per subset

`fixed_passes:200`, per-cell mean over three seeds (N0: one seed, no seed axis):

| subset | N0 | N1 | N2 | N3 |
|---|---:|---:|---:|---:|
| DS02 | **9.01** | 26.32 | 10.66 | 13.53 |
| DS05 | **13.96** | 22.62 | 15.95 | 17.24 |
| DS06 | **11.24** | 22.48 | 13.88 | 14.01 |
| DS07 | **6.50** | 23.35 | 13.92 | 11.72 |

`inner_val` (`epoch*`) evaluation point:

| subset | N0 | N1 | N2 | N3 |
|---|---:|---:|---:|---:|
| DS02 | **8.91** | 22.23 | 11.45 | 18.65 |
| DS05 | 14.08 | 22.62* | **14.03** | 17.24* |
| DS06 | **11.24*** | 22.48* | 13.88* | 14.01* |
| DS07 | **6.50*** | 23.35* | 13.92* | 11.72* |

\* Only the DS02 `inner_val` column has been recomputed from a sealed ledger.
DS05 `inner_val` carries the one reversal reported by the batch (N2 14.03 vs N0
14.08, a 0.05 gap); the remaining cells are placeholders until those ledgers are
sealed and must not be quoted.

**Age-only is first on all four subsets at `fixed_passes:200`, and on 3 of 4 at
`inner_val`** (DS05 reverses by 0.05). The gap between N0 and the best structural
rung is 1.6 / 2.0 / 2.6 / 5.2.

Cross-fold dispersion, `fixed_passes:200` (sample sd over the six fold means):

| subset | N0 | N1 | N2 | N3 |
|---|---:|---:|---:|---:|
| DS02 | 5.33 | 8.82 | 4.00 | 0.73 |
| DS05 | — | — | — | 5.59 |
| DS06 | — | — | — | 5.05 |
| DS07 | — | — | — | 2.49 |

DS02's N3 dispersion (0.73) is **specific to that subset**, not a property of N3:
the other three spread 2.49–5.59. An earlier draft of this file read the DS02
narrow band as "N3 is uniformly mediocre"; that reading is withdrawn — the band
was the subset, not the rung.

## Paired contrasts across all 24 folds

Folds are paired within a subset, and the four subsets are independent replicates
of the same design. Pooling them raises the number of independent units from 6
engines to **24**, which is the only thing that makes a sign test informative
here — at n=6 the smallest attainable two-sided p is 2/64 = 0.031, so a subset
alone can certify nothing but unanimity.

| contrast | mean, `fixed` | agree | p | mean, `inner_val` | agree | p |
|---|---:|---:|---:|---:|---:|---:|
| N1 − N0 | +13.52 | 23/24 | 3.0e−6 | +13.29 | **24/24** | 1.2e−7 |
| N2 − N0 | +3.43 | 18/24 | **0.023** | +3.59 | 16/24 | 0.152 |
| N3 − N0 | +3.95 | 17/24 | 0.064 | +6.41 | 16/24 | 0.152 |
| N1 − N2 | +10.09 | 22/24 | 3.6e−5 | +9.70 | **24/24** | 1.2e−7 |
| N3 − N2 | +0.52 | 13/24 | 0.839 | +2.82 | 15/24 | 0.307 |

Positive means the first rung is worse (higher RMSE).

Three readings, in descending order of how much weight they will bear:

1. **Vanilla is worse than everything, in every subset, at both evaluation
   points** (N1 − N0 23/24 and 24/24; N1 − N2 22/24 and 24/24). This is the one
   claim in the table that does not depend on which column is reported.
2. **MS+SA has no effect.** N3 − N2 is +0.52 at n=24 with p=0.839, and the
   direction flips across folds (13/24). Nothing about the frozen configuration
   is distinguishable from DSPT-Base.
3. **DSPT-Base does not beat a cycle counter, and the two evaluation points
   disagree about how badly it loses.** At `fixed_passes:200` it is worse with
   p=0.023; at `inner_val` it is indistinguishable with p=0.152. The wording must
   be "indistinguishable under self-selected stopping, worse under a fixed
   budget", with both columns shown — picking one is the failure mode this
   two-column design exists to prevent.

### What per-subset unanimity is worth

DS02 alone produced three unanimous contrasts at `inner_val` (N0 − N3, N3 − N2,
N2 − N1, all 6/6, p=0.031 — the floor). Pooled to 24 folds, two of them dissolve
(N3 − N2 → 15/24, p=0.307; N3 − N0 → 16/24, p=0.152) and only vanilla's deficit
survives, in fact strengthens (24/24, p=1.2e−7).

That is not a footnote; it is the argument for pooling. A single subset can
manufacture unanimity by chance at p=0.031, and it did, three times. **No
architecture claim in this file rests on a single subset.**

## The mechanism: the clock follows lifetime distance

Age-only predicts from the cycle index alone, so its error on a held-out engine
is set by how far that engine's lifetime sits from the mean lifetime of its
training engines. Measured over all 24 folds: **Spearman ρ = 0.983** between the
per-fold error and the signed lifetime distance, with error spanning 1.47 to
22.04 and distance spanning 0.0 to 23.8. The four worst folds are all at
|distance| > 20 (DS06 unit 5: +23.8 → 22.04; DS05 unit 3: −20.2 → 19.52); the
four best are all at |distance| < 1.5.

The consequence for interpretation is stronger than "the clock is a strong
baseline". The clock is not *learning* the mapping; its error is a direct readout
of a property of the fold. Under LOEO on six-engine development sets, a fold's
difficulty is fixed before any model is trained, and it dominates the spread
between rungs. This is the quantitative form of D043's "six engines are only
enough for limited-scope development evidence", and it is why every number in
this file is reported per fold.

On DS02 alone the same relation gives ρ = 0.943 over six folds — not 1.0: the one
inversion is unit 10 at 8.74 against unit 20 at 8.73, a 0.01 gap.

## Pre-registered falsification check

The three conditions below were written into this file before any result existed.
Checking them is the only thing that makes writing them worth anything.

1. **"N3 does not beat N1 by more than the cross-fold spread"** — **not
   triggered.** N3 beats N1 on all four subsets (13.53/26.32, 17.24/22.62,
   14.01/22.48, 11.72/23.35), and pooled N1 − N3 is comfortably positive.
2. **"N0 alone matches N2/N3, repeating the C-MAPSS saturation finding"** —
   **triggered, and more strongly than on DS02 alone.** N0 is first on all four
   subsets at `fixed_passes:200` and on three of four at `inner_val`. By the rule
   written in advance, the architecture contribution is **not measurable on these
   development sets** — the same saturation the C-MAPSS audit found.
3. **"N3's advantage disappears once the cycle axis is removed from every
   rung"** — **not tested.** No cycle-ablated rung was run; N0 is the cycle-*only*
   reference, not its complement.

## What this does not show

- **Nothing about the sealed test units.** They are untouched.
- **Nothing about a method being better.** The supported statement is that
  structure adds nothing measurable over a cycle counter *on these four
  development sets under LOEO*, and that vanilla is worse than the counter.
- **No attribution of the vanilla gap.** N1 and N2 differ in three ways at once —
  input representation, condition embedding, dual-span auxiliary loss. Only the
  condition embedding was isolated (`N1c`, diagnostic). The other two remain
  **unattributed** and must be described that way.
- **No claim that DSPT is harmful.** N2 − N0 at `inner_val` is p=0.152; the
  honest reading is indistinguishable, not worse.
- **No cross-subset comparison of absolute levels.** Each subset has its own
  lifetime distribution and its own fold difficulty; the pooled test pools
  *directions*, not magnitudes.

## Provenance

- DS02: recomputed here from `registry/runs_DS02.csv` (120 rows, 64 columns,
  `protocol_blob = a3c40c63…`, `git_sha = 49f6939`). Every mean and sd in the
  DS02 column above was regenerated from that file, not copied.
- DS05/06/07: batch output at `git_sha = b889bd2`, `protocol_blob = a3c40c63…`,
  one ledger per subset under `experiments/ncmapss/<subset>/runs_dev.csv`.
  **Not yet sealed.** The columns above must be regenerated from the sealed
  ledgers before submission; the `*` marks in the `inner_val` table exist because
  those cells have not been recomputed.
- Both evaluation points come from **one training run** with two weight
  snapshots; the 240 pairs share no checkpoint sha256, so the two columns are two
  different sets of weights, not one file read twice.
