# N-CMAPSS DS02 final results

Status: **empty by design**. This file is opened only after the freeze manifest
is written and committed, and only for the three official test units
`[11, 14, 15]`.

## Freeze procedure

Before any official-test number exists, the following must be committed on
branch `ncmapss-ds02`:

1. The exact configuration of N0–N3 (config files, not prose).
2. The analysis rule: which metric, which aggregation, which interval method,
   and what counts as "insufficient power" rather than "no difference".
3. The list of runs that will be evaluated, with their run ids.
4. This file, containing only the procedure — no results.

Only then are the test units opened. If the procedure changes afterwards, the
change is recorded as a deviation with its reason, and the affected numbers are
marked.

## Results (PENDING — do not fill before the freeze)

| rung | unit 11 | unit 14 | unit 15 | mean | sd |
|---|---|---|---|---|---|
| N0 age-only | | | | | |
| N1 vanilla | | | | | |
| N2 DSPT Base | | | | | |
| N3 frozen DSPT | | | | | |

Every cell states: prediction instant (end of flight), label convention, and the
number of flights contributing.

## Reporting rules

- A number from the official test units is descriptive. It is never used to pick
  a model, a hyper-parameter, or an ablation variant.
- If the LOEO development result and the official-test result disagree, both are
  reported; the disagreement is the finding, not something to reconcile away.
- No DS02 number is compared against a C-MAPSS number without the protocol
  statement required by `CROSS_DATASET_COMPARISON.md`.
