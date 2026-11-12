# N-CMAPSS DS02 baseline ladder

Status: **DS02 development results complete** — 120/120 cells on 2026-11-12,
both evaluation points present for every cell. The official test units remain
sealed; no test result exists.

## The ladder

Four rungs, from trivial to the frozen DSPT configuration. The ladder exists to
answer one question — *how much of any improvement is just the cycle axis?* —
because that is what the C-MAPSS audit found to dominate there.

| rung | model | inputs | purpose |
|---|---|---|---|
| **N0** | age-only | cumulative cycle only | the floor. If a structural model cannot beat this, nothing else matters |
| **N1** | vanilla Transformer | flight statistics + cycle | same family as the C-MAPSS `VAN_*` arms |
| **N2** | DSPT Base | same as N1, DSPT architecture | isolates the architecture term |
| **N3** | frozen DSPT | DS02-adapted input layer only | the transfer claim |

## Rules

1. **N3 is frozen.** Its MS / SA / HI switches are the C-MAPSS values, fixed
   before any DS02 result exists. DS02 outcomes may not be used to re-decide
   them. If N3 underperforms, that is a finding about transfer, not a licence to
   retune.
2. All four rungs use the identical split, the identical LOEO folds and the
   identical metric. Any deviation is recorded as a deviation, not silently.
3. Every rung is reported per fold with dispersion. A single mean is not a
   result.
4. N0 must be reported even though it is uninteresting: it is the reference that
   tells the reader how much of the performance is the cycle axis.

## Metrics

- Primary: per-unit RMSE at the flight-level prediction instant, then the mean
  and standard deviation over the six LOEO folds.
- Secondary: MAE, and a NASA-style asymmetric score only if the label convention
  is stated alongside it.
- Every table states: prediction instant (end of flight), label convention
  (capped / raw / no cap), and the number of units.

## Results — DS02 development set (120/120, 2026-11-12)

Six LOEO folds over the official dev units `[2,5,10,16,18,20]`. Prediction
instant = end of flight; label = raw RUL (the 125 cap is inactive on DS02, since
`max(Y)` = 88 < 125). Each cell is the arithmetic mean of three seeds
(101/202/303); **N0 has no seed axis** — fixed seed 101, one run per fold. The
official test units are sealed and contribute nothing here.

`fixed_passes:200` evaluation point:

| held-out unit | `max(Y)` | distance from training mean | N0 | N1 | N2 | N3 |
|---|---|---|---|---|---|---|
| 2 | 74 | +0.8 | 2.46 | 20.56 | 5.29 | 14.28 |
| 5 | 88 | +17.6 | 16.74 | 34.80 | 16.14 | 14.60 |
| 10 | 81 | +9.2 | 8.74 | 26.96 | 7.72 | 13.14 |
| 16 | 62 | −13.6 | 13.12 | 18.34 | 13.50 | 13.31 |
| 18 | 70 | −4.0 | 4.27 | 38.84 | 9.15 | 13.09 |
| 20 | 65 | −10.0 | 8.73 | 18.40 | 12.18 | 12.77 |
| **mean** | | | **9.01** | **26.32** | **10.66** | **13.53** |
| **sd** | | | **5.33** | **8.82** | **4.00** | **0.73** |

The same cells at the `inner_val` (`epoch*`) point give N0 8.91 / N1 22.23 /
N2 11.45 / N3 18.65. **Both points order the rungs identically
(N0 < N2 < N3 < N1)**; the absolute levels move (N1 by 4.1, N3 by 5.1), so
neither column substitutes for the other and both are reported.

### What the table supports, with its scope

1. **Age-only and DSPT-Base are comparable on DS02** — 9.01 against 10.66, a gap
   of 1.6 with cross-fold sd of 5.33 and 4.00. Scope: DS02 dev, six LOEO folds,
   `fixed_passes:200`. It does **not** say the cycle axis beats the architecture:
   fold 1 (2.46 against 5.29) is the fold whose held-out lifetime sits 0.8 from
   the training mean, and it is not representative of the other five.
2. **Vanilla is clearly worse than DSPT-Base** — 26.32 against 10.66, and N2 is
   better in **all six folds** (18.34–38.84 against 7.72–16.14). Scope: this
   contrast mixes three changes — input representation (flat 51-dim vector
   against sensor tokens), the condition embedding, and the dual-span auxiliary
   loss. Only the condition embedding was isolated (`N1c`, diagnostic arm). The
   other two remain **unattributed** and must be described as such.
3. **MS+SA has no consistent effect** — paired N3−N2 per fold: +9.00, −1.53,
   +5.42, −0.19, +3.94, +0.59; mean **+2.87**, four folds worse and two better,
   two-sided sign test **p ≈ 0.69**. This is **not** a negative transfer and must
   not be written as one: the direction flips with the held-out unit, and
   averaging it would manufacture a direction that no fold shows.

### Methodology finding: the lifetime-distance confound

Age-only error tracks |held-out `max(Y)` − mean `max(Y)` of the five training
units| with Spearman **ρ = 0.943** — not 1.0: the single inversion is unit 10 at
8.74 against unit 20 at 8.73, a 0.01 difference — over a 6.8x error span (2.46
to 16.74). Any cross-fold ranking is dominated by this quantity; it is the
quantitative form of D043's "six engines are only enough for limited-scope
development evidence", and the reason every result here is reported per fold.

A second shape worth recording: **the frozen rung is by far the flattest**
(12.77–14.60, sd 0.73) while N0 spans 2.46–16.74 (sd 5.33). The confound acts on
the rungs that follow the cycle axis; N3 does not follow it, so it inherits
neither the good folds nor the bad ones — it is uniformly mediocre. **A
low-variance rung and a low-error rung are not the same thing**, and N3's small
spread is not evidence in its favour.

### Pre-registered falsification check

The three conditions under "What would falsify the transfer claim" were written
before any number existed. Checking them is the only thing that makes writing
them worth anything.

1. **"N3 does not beat N1 by more than the cross-fold spread"** — **not
   triggered.** N3 (13.53) beats N1 (26.32) by 12.8, and the fold ranges do not
   overlap (N3 ≤ 14.60, N1 ≥ 18.34).
2. **"N0 alone matches N2/N3, repeating the C-MAPSS saturation finding"** —
   **triggered.** N0 (9.01) is not merely comparable but better than N2 (10.66)
   and N3 (13.53). By the rule written in advance, the architecture contribution
   is **not measurable on this dataset** — the same saturation the C-MAPSS audit
   found.
3. **"N3's advantage disappears once the cycle axis is removed from every
   rung"** — **not tested.** No cycle-ablated rung was run; N0 is the
   cycle-*only* reference, not its complement.

Secondary table (MAE, Bias, NASA score, early/mid/late segments) is not yet
reported; it is required before any of the above enters the manuscript.


## What would falsify the transfer claim

- N3 does not beat N1 by more than the cross-fold spread.
- N0 alone matches N2/N3, which would repeat the C-MAPSS saturation finding and
  make the architecture contribution unmeasurable on this dataset too.
- N3's advantage disappears once the cycle axis is removed from every rung.
