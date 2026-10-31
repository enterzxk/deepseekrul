# E18 — FD001 five-fold replication of the time-channel contrasts

Status: **FROZEN 2026-10-31T10:10+08:00**, before any fold-1..4 FD001 run exists.
Authorised by the human instruction of 2026-10-31T10:05 ("批准立即启动 FD001 五折跨子集复现，
优先验证评价分辨率主张"). Registered before computing, per `paper/QUEUE.md` rule 2.

## 1. Question

Every neural result in this paper comes from FD002 (260 engines, six operating conditions).
The paper's central claim is a resolution boundary of roughly 1.5 RMSE. If that boundary is
a property of the benchmark it must appear on a second subset with a different regime count;
if it is a property of FD002 it must be labelled as such.

FD001 (100 engines, one operating condition, 20 engines per fold) is the only second subset
inside the current data boundary. This experiment asks whether the two contrasts that carry
§4.3 replicate there.

## 2. Design

| item | value |
|---|---|
| subset | FD001 (100 engines) |
| folds | 5, engine-level, mutually exclusive, 20 engines each; the permutation is `split_seed 7`, fold `k` = block `k` |
| fold 0 | **reused** from `experiments/e3_fd001/*` (verified identical partition, 2026-10-31) |
| arms | A (sensor+EMA+cycle), BE (sensor+EMA, no time), PE (bounded sinusoidal time) |
| seeds | 2027, 2028, 2029 (unchanged) |
| budget | 80 epochs, patience 20, `d_model 64`, `num_layers 1`, `--ref-mode global` |
| new runs | 3 arms x 3 seeds x 4 folds = **36** |
| loss / metrics | unchanged from `scripts/run_candidate.py` |

No hyperparameter is re-tuned. No fold is dropped. No arm is added or removed.

## 3. Estimator (identical to FD002, so the two are comparable)

* Per-engine error vectors from `matched_cutoff_predictions.csv`, paired on `unit_id`.
* Point estimate: pooled over the 260 engine-fold pairs (FD001: 100), seed 2027 for the
  interval, because every arm has it; the three-seed means are reported descriptively.
* Interval: clustered bootstrap, 100 000 resamples, engines resampled **within fold**,
  RNG 9000. `Δ = RMSE(first) − RMSE(second)`; negative favours the first arm.
* Fold-level values are always reported beside the pooled value: with 20 engines per fold
  the fold spread is the dominant uncertainty here.
* Equivalence: TOST at δ = 1.0 RMSE, same margin as FD002.

## 4. Power, fixed in advance

Paired MDE for FD001 was computed **before** these runs from the fold-0 residuals:

```
MDE(n) = 2.80 · σ / √n,   σ = 7.73  (delta-method RMSE statistic)
MDE(20) = 4.84      MDE(100) = 2.16
```

So at 20 engines per fold **nothing below 4.8 RMSE is detectable in a single fold**, and at
100 pooled engines nothing below 2.2. The FD002 fold-0 estimate of BE − A was +1.67.

## 5. Pre-registered predictions (falsifiable)

| # | prediction | if it fails |
|---|---|---|
| 1 | BE − A pooled over 100 FD001 engines is **positive** (same sign as FD002) | the time channel is not load-bearing on FD001; §4.3 must say the effect is FD002-specific |
| 2 | the 95% CI of BE − A **includes zero** on FD001, because 1.7 < MDE 2.2 | if it excludes zero, the FD001 effect is larger than the FD002 point estimate and §4.3 must report the discrepancy |
| 3 | PE − A pooled is **within ±1.5 RMSE** and its CI includes zero | if PE wins by more than the margin, the bounded encoding replicates and §4.3 gains a positive result |
| 4 | the fold-to-fold SD of Δ is **larger on FD001 than on FD002** (20 vs 52 engines per fold) | if it is smaller, the fold term is not engine sampling and §4.6 must be revised again |

Prediction 2 and 4 are the ones that test the paper's own thesis. They are stated so that
they can fail.

## 6. What counts as a replication

| outcome | label used in the manuscript |
|---|---|
| same sign, CI includes 0 on FD001, CI excludes 0 on FD002 | **direction replicates, resolution does not** — the paper's main point, stated with two subsets |
| same sign, both CIs exclude 0 | effect replicates; FD001 adds power |
| opposite sign | **not replicated**; §4.3 is restricted to FD002 and the abstract is changed |

## 7. Prohibitions

No re-tuning after seeing a fold. No dropping the fold with the worst number. No switching
the seed for the interval. No changing δ. No using FD001 test data (unread, stays unread).
FD003/FD004 remain sealed.

## 8. Artifacts

`experiments/e18/{arm}_s{seed}_f{fold}/`, `results/dsh/e18/FD001_cv.json`,
`scripts/dsh_e18_fd001_analyze.py`, and a comparison table against the FD002 values in
`results/dsh/e16/arch_corrected.json` and `results/dsh/e10/E10*_cv.json`.
