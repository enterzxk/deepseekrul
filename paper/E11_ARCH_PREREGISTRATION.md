# E11 — Pre-registration: do five published architectures separate on C-MAPSS?

Frozen: 2026-10-30T23:50+08:00, before any baseline run was launched.
(The 1-epoch smoke test of `cnn1d` on fold 1 is the only run that predates this file;
it produced no metric used anywhere.)

## 1. The gap this closes

Every contrast in this paper so far is an ablation of **one** model family: A, PE, BE
and PS are the same 41 858-parameter candidate with one input channel added, removed or
scrambled. That is enough to calibrate the *design* (E10, E10b) but it is **not** enough
to support the paper's headline claim that method differences on C-MAPSS are largely
unresolvable. A reviewer is entitled to say: "you showed that your own ablations are
noise; that says nothing about the methods the literature compares."

E11 fixes exactly that. Five standard C-MAPSS architectures already implemented in
`src/baselines.py` are trained under the identical 5-fold protocol:

| variant | architecture | note |
|---|---|---|
| `cnn1d` | 1-D CNN over the sensor window | Li et al. 2018 style |
| `bilstm` | bidirectional LSTM | Zheng et al. 2017 style |
| `gru` | GRU + additive attention | attention-pooling family |
| `tcn` | dilated temporal convolution | Bai et al. 2018 style |
| `tf` | vanilla Transformer encoder | Vaswani et al. 2017 style |

All five receive the **14-sensor window only**, which is how they appear in the
literature; the candidate arms additionally receive the cycle index and the EMA
features. The contrast is therefore "published architecture as published" versus "our
candidate with engineered features" — the comparison the paper is actually about.

## 2. Design

* 5 architectures × 3 seeds {2027, 2028, 2029} × 5 folds {0,1,2,3,4} = **75 runs**.
* Identical everything else: `--subset FD002 --split-seed 7 --fold k --ref-mode global
  --epochs 80 --patience 20`. Baseline `d_model` is 88 as defined in
  `src/baselines.py`; the candidate uses 64. Parameter counts are reported, not matched
  (matching capacity across architectures is not meaningful here, and the paper says so).
* Estimand and interval: **the E10 recipe** — pooled MSE over 260 engines × 3 seeds,
  sqrt, fold-stratified engine bootstrap, B = 100 000, RNG 9000–9004, MC error reported.
  Sign convention Δ = RMSE(first) − RMSE(second).
* Reference arm for the headline table: **A** (the candidate with cycle + EMA), whose
  fold-0..4 predictions already exist from E10. No A run is repeated.

## 3. Decision rule (fixed now)

Primary, descriptive:

| outcome | condition | what the paper may say |
|---|---|---|
| **DISPERSED** | at least one architecture pair has a 95% CI excluding zero **and** \|Δ\| ≥ 1.5 RMSE | architectures are distinguishable at this scale; report which, and the paper's "method differences are unresolvable" claim is **narrowed** to "differences below 1.5 RMSE" |
| **INDISTINGUISHABLE** | no pair reaches \|Δ\| ≥ 1.5 with a CI excluding zero | all five architectures fall inside one resolution cell; the paper may state that at 260 engines the architecture choice does not separate them |

Secondary, pre-specified and reported regardless:

* the spread of the five architectures' pooled RMSE (max − min), to be compared against
  the between-fold SD of a single architecture (1.11, measured in E10b);
* how many of the 10 pairwise contrasts have a CI crossing zero;
* per-fold rankings, to show whether the *ordering* of architectures is stable across
  folds even when the magnitudes are not.

## 4. Prediction, recorded before the runs

Based on the E10b fold spread (SD 1.11) and the ±1.3 resolution measured in E10:

* **the max−min spread across the five architectures will be 1.0–4.0 RMSE**, i.e. of the
  same order as one between-fold SD;
* **at least half of the 10 pairwise contrasts will have a CI crossing zero**;
* **the per-fold ranking will not be stable** — the top architecture on fold 0 will not
  be the top architecture on at least two other folds.

If any of these three predictions fails, that is reported as a failure of the prediction
and it weakens the paper's thesis; it is not re-interpreted.

## 5. Failure handling

* If a baseline crashes or OOMs: fix only the crash, keep the configuration, rerun that
  cell, and record it in `paper/RUNLOG.md`. No architecture is dropped after seeing a
  number; if one cannot be trained at all, that is reported.
* No seed, fold, epoch or metric search. The rule in §3 is applied as written.

## 6. Artefacts

```
experiments/e11/{cnn1d,bilstm,gru,tcn,tf}_s{2027,2028,2029}_f{0..4}/FD002/*/
    metrics.json, matched_cutoff_predictions.csv
results/dsh/e11/E11_arch.json
paper/E11_RESULT.md
```

## 7. Relation to the manuscript

This becomes the primary table of §4.2 ("what a comparison at this scale can resolve"),
replacing the withdrawn 21-pair table. The E10/E10b contrasts remain as the *design
calibration* (8% resolvable, 1% not), and E11 supplies the *method-level* evidence: five
architectures that the literature treats as meaningfully different, measured on the same
260 engines.

## 8. Boundaries

FD002 only; FD003/FD004 sealed; no official test file read. Five architectures is not
"all methods" — it is five representative families, and the paper says five, not all.
Capacity is not matched across architectures, by design, and parameter counts are
reported alongside every number.

## 9. Amendments

### Amendment 1 — 2026-10-31T01:15+08:00 — TCN and Transformer reduced to a single seed

**What changed.** `tcn` and `tf` run **1 seed (2027) × 5 folds = 10 runs**, not 3 seeds × 5
folds = 30. `cnn1d`, `bilstm`, `gru` and `A` keep all 15 runs; nothing already completed
is discarded.

**Why.** Measured cost, not convenience. Under the 3-way concurrency used here, `cnn1d`
runs at ≈0.5 min/epoch and `tcn` at ≈2.0 min/epoch on the same GPU (measured 01:05–01:13,
3 concurrent, GPU at 100% utilisation, 43 GB free). The full 30-run plan therefore needs
≈17 GPU-hours; the reduced 10-run plan needs ≈5.5. The first attempt also lost 30 runs to
an external SIGTERM at 01:07 and had to be relaunched, which cost another hour.

**Why this does not weaken the comparison.** The seed component of the uncertainty was
measured directly in E9: **2.4% of the variance of a paired contrast on FD002**, against
97.6% for engine sampling. A five-fold pooled estimate with one seed and with three seeds
differ only through that 2.4% component. The between-fold term — the dominant one at 260
engines (63–65% of `u_c`, §4.6) — is fully preserved, because all five folds are kept.

**How it will be reported.** Any row in the architecture table computed from a single seed
is marked `†` in the table and in the caption, and the per-architecture seed counts are
listed in the supplement. No claim in the paper will rest on a single-seed difference
smaller than `u_c`.

**What was NOT changed.** The folds, the split seed, the evaluation protocol, the epoch
budget, the decision rule of §3, and the five architectures themselves.



## Amendment 2 (2026-10-31T09:50+08:00) — baseline capacity sensitivity

**Why.** A reviewer will ask whether the five architectures were strawmen: every arm was
trained at `d_model=64, num_layers=1`, i.e. the smallest configuration that fits the
protocol. If a larger baseline closes the gap to A, then the between-band contrasts of
E11 are a tuning artefact rather than a property of the architectures.

**What.** Two architectures are re-run at `d_model=128, num_layers=2` (4x parameters),
seed 2027 only, all 5 folds = 10 runs. Chosen because they are the two that matter:
`tf` because it is the arm A ties with on the official test set, `cnn1d` because the
largest between-band contrast (+6.28) depends on it being weak.

**Decision rule, fixed before the runs.**

| outcome | interpretation |
|---|---|
| tuned `tf` still within ±1.0 of A | the A–tf gap is not a capacity artefact; §4.2.3 stands |
| tuned `tf` beats A by >1.0 | the candidate's advantage over the Transformer is withdrawn; §4.2.3 and §4.3 must say so |
| tuned `cnn1d` improves by >2.0 | the +6.28 contrast is a capacity artefact and the largest between-band claim is withdrawn |

The same three rules apply to `cnn1d`. No re-running to "find a better setting": one
alternative configuration per architecture, pre-declared, reported whatever it shows.

**Cost.** ~1 GPU-hour.

**Amendment 2a (10:05).** The first launch was a **no-op and is void**: the baseline branch
of `run_candidate.py` built every baseline with `d_model=88` hard-coded and never forwarded
`--d-model` or `--num-layers`, so `--d-model 128 --num-layers 2` produced the *identical*
configuration. Seven runs completed before this was noticed; all seven reproduce their E11
counterparts to six decimal places (`e16` comparison, 2026-10-31T10:03), which is recorded
separately as a determinism check rather than as a capacity result. All seven directories
are marked void and are not used.

Two consequences for the manuscript. (i) Every E11 baseline ran at `d_model=88` while the
candidate ran at `d_model=64`, so the baselines had **more** width than the candidate, not
less; the protocol description in §4.2.3 has been corrected accordingly. (ii) None of the
five baseline classes exposes a depth parameter, so the only capacity lever is width.

The code now has an explicit `--baseline-d-model` (default 88, which preserves the E11
runs bit-for-bit) and records `baseline_d_model` and `num_params` in every `metrics.json`.
The sensitivity is re-run at 128 under `experiments/e17b`.
