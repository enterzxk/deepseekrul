# N-CMAPSS DS02 baseline ladder

Status: **skeleton, no numbers yet**. No DS02 result has been produced at the
time of writing.

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

## Results table (PENDING)

Every cell is run with **≥3 model seeds**; the cell holds the seed-mean and the
seed standard deviation is reported alongside (a single-seed fold is not a
result). Per-fold seed spread and cross-fold spread are both reported, and the
two are never conflated.

| rung | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | fold 6 | mean | sd |
|---|---|---|---|---|---|---|---|---|
| N0 age-only | | | | | | | | |
| N1 vanilla | | | | | | | | |
| N2 DSPT Base | | | | | | | | |
| N3 frozen DSPT | | | | | | | | |

Secondary table (same folds, same seeds): MAE, Bias, NASA score, and
early/mid/late lifetime segment errors, each with the label convention named.

## What would falsify the transfer claim

- N3 does not beat N1 by more than the cross-fold spread.
- N0 alone matches N2/N3, which would repeat the C-MAPSS saturation finding and
  make the architecture contribution unmeasurable on this dataset too.
- N3's advantage disappears once the cycle axis is removed from every rung.
