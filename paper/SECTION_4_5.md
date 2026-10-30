# 4.5 Where the protocol decides the answer

Draft. Numbers in the E6 convention (Δ = RMSE(first) − RMSE(second), engine-clustered
100k resamples, Monte-Carlo error ≤ 0.02).

Three protocol choices are usually treated as housekeeping. Each of them moves the
result by more than the architectural differences the benchmark is used to compare.

## 4.5.1 How many predictions per engine

A comparison usually gives each held-out engine one prediction. Giving it several
costs nothing at training time — the model is already trained — and narrows the
interval, because the engine-level variance is shared across its cutoffs while the
residual noise is not.

We re-analysed the stored runs, restricting to engines that contribute every cutoff in
the design, so the only thing that changes is observations per engine:

| contrast | one cutoff | two cutoffs (same engines) | width ratio | four cutoffs |
|---|---:|---:|---:|---:|
| FD002 A vs BE | 3.362 | 2.528 | **0.75** | 2.328 (0.69, 123 engines) |
| FD002 A vs PE | 3.508 | 2.303 | **0.66** | 1.923 (0.55, 123 engines) |
| FD001 A vs BE | 4.635 | 3.282 | **0.71** | 3.503 (0.76, 42 engines) |

Two cutoffs instead of one cut the interval to two-thirds to three-quarters of its
width on the *same* engines and the *same* trained models. On FD001, adding all four
cutoffs is counter-productive because the 180-cycle cutoff removes too many short
engines — a reminder that the design has to be checked, not assumed.

This is the one place in this paper where a change makes the benchmark strictly
better without any new data, and it should be the default: **report at least two
cutoffs per engine and cluster on engines.**

## 4.5.2 Where the cutoffs sit

The same contrast, same models, same engines, evaluated under the deployment-style
matched cutoff and under fixed absolute cutoffs:

| evaluation design | ΔRMSE (BE − A, FD002) | 95% CI |
|---|---:|---|
| matched (random cutoff in [40, 0.95T]) | **+2.693** | [+1.352, +4.096] |
| absolute cutoff 100 | +0.002 | [−1.672, +1.690] |
| absolute cutoffs 100 + 140 | −0.625 | [−1.891, +0.638] |

The time channel is worth about 2.7 RMSE when the evaluation samples the whole life,
and indistinguishable from zero when it samples mid-life. This is not a contradiction:
early in life the capped label is nearly constant, so a coordinate that encodes age
predicts it almost perfectly; at mid-life the cap no longer binds and the remaining
variance is carried by other features.

The consequence for comparison tables is direct: two papers can report opposite
conclusions about the same design choice and both be internally correct, because they
cut at different places. A comparison table that does not state its cutoff policy is
not interpretable, and "matched cutoff" must not be treated as a synonym for
"absolute cutoff at 100".

## 4.5.3 How "severe" is defined

The frozen definition is a late prediction with error > 10 cycles. A common proxy in
earlier reports of this project was "late and RUL ≤ 30". On identical predictions the
proxy reports 0.071 where the frozen definition reports 0.345 — the proxy understates
severe late predictions by roughly a factor of five, because it only counts the last
30 cycles while the large errors occur earlier.

We report the frozen definition throughout and list the proxy here so that the
discrepancy is on the record rather than buried in a superseded table.

## 4.5.4 What the cap does

The 125-cycle cap is a modelling convention, not part of the benchmark. On a fixed set
of predictions, changing the cap from 125 to none moves RMSE from 12.06 to 26.9; the
cap at 80 gives 24.8. A comparison between two papers that use different caps is
therefore not a comparison of methods, and a paper that does not state its cap has not
reported a number that can be reproduced.

## 4.5.5 Summary

Four protocol choices — predictions per engine, cutoff placement, severe definition,
cap — each move the metric by an amount comparable to or larger than the method effects
the benchmark is used to measure. None of them is a modelling contribution, and all of
them are free to fix. §5 turns them into a checklist.
