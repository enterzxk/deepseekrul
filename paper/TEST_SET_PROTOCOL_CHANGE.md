# Protocol change: opening the official test set

Status: **AUTHORIZED 2026-10-31T08:57+08:00** — human replied to the request for
confirmation with: "好的你继续". Read as authorisation for `test_FD002` + `RUL_FD002`
only, under the protocol fixed in §3 below and the commitments in §4.

Execution log:
* 2026-10-31T09:40 — authorisation recorded; feature-pipeline equivalence check started.
  No test file had been read at this point.

Nothing in `data/CMAPSS/test_FD00*` or `RUL_FD00*` was read by any code in this project
between 2026-10-22 and this authorisation (§7 of `SUPPLEMENT_DATA_USE.md`).

## 1. The instruction being acted on

Human message, 2026-10-31T00:58+08:00, verbatim:

> 按照你前面说的四、如果一定要做性能对比，需要什么
> 必须同时满足三条，缺一条都无效：
> 解封官方 test set（协议变更，需要你授权）；
> 统一 cutoff / cap / metric 定义，并在表里写明；
> 报告 U，让人判断差异是否超出 1.23。

This is read as an instruction to satisfy all three conditions for a performance
comparison. Condition 1 requires a protocol change that the task book (§8) forbids by
default, so it is recorded here rather than acted on silently.

## 2. What would change

| item | before | after |
|---|---|---|
| data used | FD001/FD002 **training** engines only | training engines **and** `test_FD002` + `RUL_FD002` |
| data statement | "official test files never used" | must disclose test use, in the abstract's data sentence too |
| §7 reproducibility | re-runnable without test files | test-side numbers become re-runnable only for someone holding the files |
| remaining sealed sets | FD001 test, FD003, FD004 (all) | FD001 test, FD003, FD004 stay sealed |

FD003 and FD004 remain untouched in every scenario. Only `test_FD002` and `RUL_FD002`
would be opened, because FD002 is the only subset with a complete 5-fold training
evaluation in this study.

## 3. Evaluation protocol on the test side (must be fixed before reading)

The training-side protocol is a **matched cutoff**: each engine is scored at 50/70/90% of
its own observed life, one prediction per cutoff. The test files give a single truncated
trajectory per engine plus one true RUL, so the same protocol has to be reconstructed:

```
T_i        = len(trajectory_i) + RUL_true_i        # full lifespan, recovered
cutoff     = round(f * T_i) for f in {0.5, 0.7, 0.9}
if cutoff >= len(trajectory_i): skip that (engine, cutoff)      # cannot observe it
target     = min(RUL_true_i + (len(trajectory_i) - cutoff), 125)  # same cap as training
```

Predictions come from the **already-trained fold-k model whose training set excludes
engine i**? — not possible on the test side: test engines were never in any training set,
so every fold model is a valid predictor. To avoid a new degree of freedom, the paper
would report the **mean over the 5 fold models × 3 seeds**, i.e. 15 predictors per test
engine, and the spread across folds as an additional uncertainty component.

Metrics: RMSE, MAE, NASA score, late rate, severe rate — identical definitions to the
training side (§4.5), with the same cap of 125 and the same `e > 10` severe threshold.

## 4. Commitments that make this defensible

1. **One shot.** The test set is evaluated once, after the models are frozen. No
   hyperparameter, epoch, seed, architecture or cutoff is changed afterwards.
2. **Every number reported**, including any that contradict the training-side ranking.
3. **Disclosed in three places**: the abstract's data sentence, §7, and the supplement's
   data-use note.
4. **No test-driven model selection.** The architecture table (E11) is already frozen by
   its pre-registration; test numbers are appended, not used to choose.
5. **The training-side 5-fold results stay the primary evidence.** Test results become a
   secondary, comparability-oriented table — because the literature reports test numbers,
   not because test numbers are better evidence.

## 5. What it buys, honestly

* A table that a reviewer can put next to published C-MAPSS results **without** the
  "different protocol" objection.
* One more independent check on the paper's thesis: if the five architectures are also
  indistinguishable on the official test set, the resolution claim generalises beyond the
  training-engine protocol.
* **Risk:** test-set numbers are single-trajectory per engine (no cut averaging), so their
  own uncertainty is *larger*, not smaller. If the ranking looks different from the
  training side, the paper must report the disagreement and cannot use test numbers to
  override the 5-fold result.

## 6. What is required to proceed

A single explicit confirmation from the human, e.g. "授权解封 test_FD002". Until then this
file stays in `PREPARED` state and no code path opens the test files.
