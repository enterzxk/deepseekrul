# N-CMAPSS DS02 final results

Status: **empty by design**. This file is opened only after the freeze manifest
is written and committed, and only for the three official test units
`[11, 14, 15]`.

## Reporting rule: the held-out lifetime confound

This is the most important methodological constraint in the DS02 design, and it
was found by measurement, not by argument.

With six dev engines under LOEO, the age-only rung (N0) predicts RUL from the
cycle index alone. Its error on a held-out engine is driven almost entirely by
**how far that engine's lifetime sits from the mean lifetime of the five training
engines** — the "average cycle → RUL" curve is exact for an engine whose lifetime
equals the training mean, and degrades in proportion to the gap.

Measured on fold 1: the held-out engine is unit 2 with a lifetime of 75 flights,
and the five training engines average 74.2. N0 scores **2.45** there — better
than the linear ridge reference (5.19) and far better than DSPT-Base (6.81) or the
frozen DSPT (6.93). Holding out unit 16 (lifetime 63) instead would push N0 far
worse, because the training mean no longer matches.

Consequences, and they are reporting rules, not caveats:

1. **Every result is reported per fold, with that fold's held-out lifetime and
   the training-set mean lifetime printed beside it.** A single averaged number
   is not a result and may not appear as one.
2. **A ranking claim must survive the lifetime-distance check.** If the ordering
   of N1/N2/N3 across folds tracks the held-out engines' distance from the
   training mean rather than the architecture, the ordering is not evidence about
   the architecture.
3. **The heterogeneity is a finding, not noise to be averaged away.** It is the
   concrete form of what D043 records as "six engines are only enough for
   limited-scope development evidence".

Any attempt to remove this effect by tuning the budget, the features or the fold
assignment would be exactly the move D043 forbids: the numbers would look better
and the question would no longer be answerable.

## Reporting rule: two evaluation points, neither used for selection

Every cell is trained **once** to 200 passes, and then evaluated at **two**
points:

- `inner_val` — argmin of the mean inner validation curve, floored at 20 passes;
- `fixed_passes:200` — the final pass.

Both are reported **side by side, at the same font size**, with the stopping rule
named in the header. The body text **may not pick whichever column looks better
and draw the conclusion from it**. If the two columns order the rungs
differently, that disagreement is itself the finding.

Why both rather than one, measured before any full batch:

- `epoch*` lands between **20 and 94** passes, while the outer curve's minimum
  sits at **157–198** — a systematic offset of roughly 2x;
- the five inner per-fold minima spread by **13–19x** (N2 seed 202: 0.294 to
  3.804).

On six engines, when to stop is neither estimable nor transferable. Reporting a
single point hides that; reporting both is the honest option, and it costs
nothing because both come from the same training run.

Ledger consequence: **two rows per cell** and two prediction files suffixed
`_ev<epoch>`, distinguished by `selection_rule`.

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
- **Every reported number is accompanied by its learning curve.** Because the
  budget is fixed and there is no early stopping, the reported model is the one
  at the end of that budget. Each row therefore also carries the holdout
  minimum and the epoch at which it occurs, labelled
  `diagnostic only - not used for model selection`. This shows whether the
  fixed endpoint sits on a slope or at a floor; a fixed budget hides
  over-fitting only when the curve is hidden with it.
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
