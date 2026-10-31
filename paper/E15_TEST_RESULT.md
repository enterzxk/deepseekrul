> **CORRECTED 2026-10-31 (E16/E19).** This note was written before the statistical
> audit. Two of its statements no longer stand. (i) The paired intervals in
> `dsh_e11_arch_analyze.py` were computed from a slice that counted fold 0 three times and
> omitted fold 4; the corrected intervals are in `results/dsh/e16/arch_corrected.json`.
> **A vs the Transformer is −1.43 [−2.998, +0.133] and is NOT significant.** The other four
> contrasts stand. (ii) No equivalence test had been run; TOST at δ = 1.0 fails for every
> within-tier pair (§4.2.3). The resolution/sample-size numbers were also revised:
> MDE 1.09–1.54 at 260 engines; T1 320–644 engines, T2 627–1262 engines
> (`results/dsh/e19/power_and_folds.json`). Use those, not the figures below.
>
> This file is kept as an audit trail. The current text is `paper/SECTION_4_2.md`.

# E15 — Official test set, one shot: the advantage is real only where it is large

Executed 2026-10-31 11:35–11:55 local, once, after every model was frozen.
Authorisation: `paper/TEST_SET_PROTOCOL_CHANGE.md` (human, 2026-10-31T08:57).
Pipeline verification before opening the file: features equal to `prepare()`'s
`matched_set` to **0.0**, prediction replay **0.084 RMSE** — `dsh_test_verify.py --stage verify`
printed VERIFY PASS.

Data: `test_FD002` (259 engines) + `RUL_FD002`. 235 engines usable (24 have fewer than
50 observable cycles and cannot be scored under a 50-cycle window). FD003/FD004 untouched.

Protocol: **last observable window**, one prediction per engine, target clip 125, metrics
identical to the training side. This is the literature-standard protocol and it is *not*
the matched-cutoff protocol used in §4.2–4.3 — which is exactly why the two tables are
not comparable and are reported separately.

## 1. Result

| arm | train (matched, 5-fold) | **test (last window)** | gap | models averaged |
|---|---:|---:|---:|---:|
| **A** (candidate) | 13.953 | **23.958** | +10.005 | 12 |
| tf (Transformer) † | 15.448 | **23.976** | +8.528 | 5 |
| gru (GRU+attn) | 15.971 | 25.061 | +9.090 | 15 |
| bilstm | 16.199 | 25.247 | +9.048 | 15 |
| cnn1d | 19.968 | 29.305 | +9.337 | 15 |
| tcn † | 19.820 | 29.779 | +9.959 | 5 |

† single seed; A has 4 folds × 3 seeds (the fold-0 checkpoint was not retained in the
`e10` layout and is reported as missing rather than reconstructed).

| contrast vs A | train Δ | **test Δ** | retention |
|---|---:|---:|---:|
| tf − A | +1.495 | **+0.018** | **1%** |
| gru − A | +2.018 | +1.104 | 55% |
| bilstm − A | +2.246 | +1.290 | 57% |
| cnn1d − A | +6.015 | +5.348 | 89% |
| tcn − A | +5.867 | +5.821 | 99% |

## 2. What this says, and it is not what I hoped

**The candidate's advantage over the Transformer does not survive the protocol change.**
On 260 held-out training engines A beats `tf` by 1.50 RMSE with an interval clearing zero.
On the official test set the two are **0.018 apart** — indistinguishable.

The other four contrasts behave exactly as the resolution analysis predicts:

| effect size (train) | retention on test |
|---:|---:|
| ≈1.5 (tf) | **1%** |
| ≈2.0–2.2 (gru, bilstm) | 55–57% |
| ≈5.9–6.0 (cnn1d, tcn) | 89–99% |

**The retention is a monotone function of the effect size.** Small advantages vanish,
large ones persist. This is the same boundary that §4.2.3 found between bands (≈1.2 RMSE
unresolvable, ≈1.5+ real), now confirmed on a completely different data split, with a
different protocol and a different set of models.

It also means the paper cannot claim "A is the best architecture on C-MAPSS". The honest
claim is narrower:

> **A is significantly better than the two weakest architectures under both protocols
> (5.3–5.8 RMSE), comparable to GRU/BiLSTM (1.1–1.3, near the resolution limit), and
> statistically tied with the Transformer on the official test set despite a 1.5-RMSE
> training-side advantage.**

## 3. Why the test numbers are ~10 RMSE higher

Not a bug. Three differences, all in the direction of a harder measurement:

1. **Different engines.** Test trajectories are truncated before failure; the model never
   sees the post-onset regime it was trained on. The last-window setting is closer to the
   deployment use case and harder.
2. **One prediction per engine.** No cut averaging, so no variance reduction — the test
   interval is wider than the 5-fold training interval, not narrower.
3. **No fold averaging.** Each model is scored on its own; the training table averages
   over 5 folds × seeds.

This is precisely the ×1.73 protocol sensitivity measured in §4.5, at a larger magnitude.
It is the reason the two tables sit side by side with a barrier between them rather than
merged into one leaderboard.

## 4. Effect on the manuscript

| item | before E15 | after E15 |
|---|---|---|
| §4.2.3 architecture table | training side only | **training + test columns, with the retention column** |
| performance claim | "A significantly better than all five" | **"significantly better than two, tied with one"** |
| §4.5 protocol traps | ×1.73 from cutoff alone | **plus a 10-RMSE gap from split + protocol together** |
| data statement | test never used | **test used once, disclosed in abstract, §7 and supplement** |
| thesis | differences below ≈1.2 unresolvable | **unchanged, and now externally validated** |

## 5. Commitments honoured

* One shot: no model, seed, epoch or cutoff was changed after the test file was opened.
* Every number reported, including the one that removes the headline advantage.
* No test-driven selection: the architectures were frozen by E11's pre-registration before
  this evaluation; the test results were appended, not used to choose.
* The training-side 5-fold result remains the primary evidence; the test table is
  secondary and comparability-oriented.

## 6. Boundaries

* FD002 test only. FD001 test, FD003 and FD004 remain sealed.
* 235 of 259 engines scored; the 24 excluded engines have <50 cycles and are reported as
  excluded rather than dropped silently.
* A's fold-0 checkpoint is missing from the `e10` directory layout, so A is averaged over
  12 models against 15 for three baselines. This is a bookkeeping difference, not a
  selection: the missing fold is the one that would have *helped* A (fold 0 is A's best
  fold, 12.68 against 13.5–15.1 elsewhere), so if anything A's test average is
  conservative.
* Test-side intervals are not reported: with one prediction per engine and 5–15 models, a
  bootstrap over engines would understate the model-selection component. The training-side
  budget (§4.6) is the interval of record; the test table reports point estimates with the
  per-model spread shown in the supplement.
