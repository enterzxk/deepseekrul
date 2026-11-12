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
- **The capped criterion may be inactive on DS02.** DS02 RUL is counted in
  flights, and its lifetimes may be far shorter than the 125-cycle cap inherited
  from C-MAPSS. If no unit's raw RUL ever exceeds 125, then
  `y_true_cap125 == y_true_raw` everywhere and the capped criterion carries no
  information. In that case the capped column is reported as
  `inactive (no unit exceeds the cap)` — not printed as a second set of numbers
  identical to the raw one, and never described as "the stricter criterion".
  The measured lifetime distribution is reported first so a reader can check the
  claim rather than take it on trust.
- Symmetrically, if some units do exceed 125, the two criteria are reported
  side by side with both caps named, and the difference between them is stated
  as a property of the label definition, not as a property of the model.
