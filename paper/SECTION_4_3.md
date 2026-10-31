# 4.3 The time channel: a large effect that survives, and a small one that does not

Conventions follow `paper/E6_STATS_AUDIT.md`: Δ = RMSE(first) − RMSE(second),
engine-clustered bootstrap, 100 000 resamples, Monte-Carlo error ≤ 0.02.

Sections 4.1 and 4.2 say that the resolution of a comparison is set by engines, not
seeds, and that five folds over 260 engines can resolve effects of order 8% but not of
order 1%. This section supplies the two contrasts that boundary was measured from, on
the same pipeline, with the same estimator and the same interval recipe.

## 4.3.1 The linear probe: removing time costs a lot

A linear probe over 360 engines (all FD001 + FD002 training engines, 14 sensors plus the
cycle index, ridge regression on the last window per engine) gives:

| arm | pooled RMSE | Δ vs cycle |
|---|---:|---|
| cycle index present | 27.93 | — |
| cycle index removed | 34.82 | **+6.89 [+5.61, +8.18]** |
| cycle replaced by a sinusoidal encoding | 27.97 | +0.04 |

The effect is not specific to FD002: on FD002 alone the same ablation costs **+8.86
[+7.29, +10.39]**. A sinusoidal encoding recovers essentially all of it, which is the
first hint that what matters is *having* the time coordinate, not the functional form
used to encode it.

Probe results are not pipeline results — different preprocessing, different model class,
absolute levels not comparable (§6.3). They motivate the experiment below but do not
stand in for it.

## 4.3.2 On the frozen pipeline: +2.69 on one split, +1.13 over five

Arm A feeds `min(c/250, 1)` into the model; arm BE removes that channel and changes
nothing else (−128 parameters). On the single archived 80/20 split (52 held-out engines)
the contrast was **+2.693 [+1.352, +4.096]**, a 21% RMSE penalty.

Repeating the arm on all five folds (E10b, pre-registered, 12 new runs, A taken from
E10) gives:

| | pooled, 260 engines × 3 seeds |
|---|---:|
| RMSE, A | 13.983 |
| RMSE, BE | 15.116 |
| **ΔRMSE (BE − A)** | **+1.133 (+8.1%)** |
| 95% CI | **[+0.245, +2.023]** |

The effect is **reproduced** — the interval clears zero — but it is **2.4× smaller than
the single split reported**, and one fold reverses its sign:

| fold | 0 (archived) | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Δ (BE − A) | +2.677 | **−0.306** | +1.088 | +1.571 | +0.566 |

Between-fold SD is 1.11. Three seeds per fold agree with their own fold (fold 0:
+2.71/+2.59/+2.73), so the spread is not seed noise.

One boundary, stated here rather than buried: the pre-registered interval resamples
engines *within* folds and is therefore conditional on these five folds. Treated as a
random effect across folds, the interval is `1.12 ± 2.78 × 1.11/√5 ≈ [−0.26, +2.50]` and
would not clear zero. Five folds is a small number of folds. The verdict above follows
the pre-registered rule; the random-effect reading is disclosed in §6.5 and does not
change the paper's conclusion (§4.2.2 makes the same point about the fold floor).

## 4.3.3 The contrast that does not survive

The same design, the same 260 engines, applied to a smaller design change:

| contrast | single split (52) | five folds (260) | 95% CI (five folds) | verdict |
|---|---:|---:|---|---|
| BE − A (remove time) | +2.693 | **+1.133 (+8.1%)** | [+0.245, +2.023] | reproduced |
| PE − A (bounded encoding) | −0.908 | **−0.126 (−0.9%)** | [−1.073, +0.795] | **not reproduced** |

The bounded sinusoidal encoder was, on the single split, a 7.7% improvement with three
mutually agreeing seeds. On 260 engines it is a 0.9% non-result, and **PE wins on 51% of
individual engines**. Neither lifespan nor degradation rate explains the sign (tertile
means −0.22 / −0.58 / +0.42; `corr(Δ, lifespan) = +0.015`).

The per-fold estimates make the failure mode explicit: −0.90, **+1.70**, −1.02, −0.48,
−0.03. A reader given fold 1 alone would report an 11% regression with three consistent
seeds.

Both contrasts were overstated on fold 0 — by 2.4× and 7× respectively — and fold 0 is
the easiest fold for the baseline arm (A scores 12.68 there against 13.5–15.1
elsewhere). Two contrasts are not enough to establish that as a rule; the paper reports
the observation and its direction without generalising it.

### 4.3.3b The effect replicates across subsets — with one dissent

FD001 (100 engines, one operating condition), FD003 (100 engines, one condition, fan
degradation) and FD004 (249 engines, six conditions) were run through the identical
5-fold design as FD002 (`paper/E18_RESULT.md`,
`paper/E20_FD0034_CV_PREREGISTRATION.md` incl. Amendment 2; pre-registrations
`59bb5d9b` and `3f0142f9`):

| contrast | FD002 (260) | FD001 (100) | FD004 (249) | FD003 (100) | pooled (709, inverse-variance) |
|---|---:|---:|---:|---:|---:|
| **BE − A** | **+1.133** [+0.245, +2.023] | **+2.052** [+0.505, +3.733] | **+1.497** [+0.124, +2.981] | −1.315 [−2.949, +0.316] | **+0.979** [+0.348, +1.610] |
| PE − A | −1.433 [−2.998, +0.133] | −0.634 [−2.159, +0.809] | −0.530 [−1.363, +0.258] | −0.952 [−2.438, +0.276] | signs consistent, none significant |
| PE − BE | −3.561 | −2.686 | −2.027 | +0.363 | 3 of 4 significant |

The time channel replicates on three of four subsets, spanning one-condition and
six-condition regimes, and the four-subset pooled estimate is +0.979 [+0.348, +1.610]
over 709 engines — but the heterogeneity is high (Cochran Q = 9.90, df = 3, I² = 70%),
because **FD003 reverses sign** (−1.315, n.s.): on that subset removing the time channel
does not hurt and if anything helps. The three-subset pooling reported earlier
(+1.381 [+0.698, +2.065], 609 engines) excluded FD003 and is retained for provenance.
The encoding choice does not replicate anywhere: PE − A is −0.53 to −1.43 on the first
three subsets and never significant, which is what the resolution boundary predicts for
a sub-1.5-RMSE difference. Of E18's four pre-registered predictions, three held and one
failed in the favourable direction (the FD001 CI cleared zero where the registration
expected it not to); the failure is reported, not absorbed. One e20 run (FD003/A/seed
2027/fold 0) was trained with a wrong layer count by a launcher defect, was preserved,
retrained to the registered specification, and the subset re-analysed; the numbers above
are the fixed-architecture ones (Amendment 2).

## 4.3.4 The mechanism is alignment, not capacity

The bounded encoder adds a `Linear(8, 64)` projection in place of `Linear(1, 64)`: +448
parameters (+1.07%). A parameter-matched control is not informative here — seven dead
input columns receive zero gradient, so the extra capacity is never exercised. The
control that does work destroys the *alignment* while holding everything else fixed: the
scrambled-age variant (PS) feeds each window another engine's age, with the same 8→64
projection, the same parameter count, the same periods, and the same marginal
distribution of ages within a batch.

| contrast | FD002 | FD001 |
|---|---:|---:|
| PS − PE (destroy the alignment) | **+3.512 [+2.214, +4.857]** | **+2.144 [+0.366, +4.093]** |
| PS − BE (alignment destroyed vs time removed) | −0.078 [−1.193, +0.995] | −0.470 |

Two readings. First, breaking the correspondence between the encoding and the engine's
own age costs 3.5 RMSE — larger than any module-swap effect measured in this project.
Second, PS is statistically indistinguishable from BE, which is the control's own
validity check: scrambling removed the information and nothing else.

The mechanism sentence the paper can support: **what carries the signal is having a time
coordinate aligned with the engine, not the shape of the encoding.** This is also why the
PE-versus-A question turned out to be a question about a 1% detail rather than about
"does time matter".

## 4.3.5 Subsets have resolutions too

The same BE − A contrast on FD001 (20 held-out engines per split) is **+1.674 [−0.250,
+3.699]**: directionally identical, statistically invisible, because that subset's MDE
at 20 engines is 4.84. This is not a contradiction between subsets; it is the same effect
measured with a coarser instrument. §4.2 gives the general form.

## 4.3.6 Official-test generalisation (E21)

After all training was frozen, the official test sets of all four subsets were evaluated
under one pre-registered protocol (`paper/E21_TEST_ALL_PREREGISTRATION.md`, sha
`1579e0e0`): one prediction per engine (last observable 50-cycle window), each fold's
training scaler applied unchanged, ensemble over 5 folds × 3 seeds × 3 arms, truth = the
official RUL files. The FD003/FD004 test files were kept permission-sealed until that
point and unsealed on 2026-10-31 with sha256 recorded and mtimes untouched
(`results/dsh/e21/unseal_hashes.sha256`). Short trajectories are excluded as in §6.10
(FD001 7/100, FD002 24/259, FD003 3/100, FD004 20/248 — exactly the counts registered).

Ensemble RMSE (95% bootstrap CI over engines):

| subset | A | BE | PE | n |
|---|---:|---:|---:|---:|
| FD001 | 11.340 [9.422, 13.199] | 13.104 [10.884, 15.272] | 10.608 [8.812, 12.392] | 93 |
| FD002 | 23.386 [20.022, 26.594] | 23.329 [20.085, 26.460] | 23.224 [19.734, 26.552] | 235 |
| FD003 | 11.192 [9.297, 13.052] | 12.879 [10.721, 14.970] | 11.484 [9.464, 13.496] | 97 |
| FD004 | 22.404 [19.429, 25.244] | 22.940 [20.023, 25.729] | 22.317 [19.253, 25.232] | 228 |

Paired contrasts (delta-method bootstrap of engine pairs, 100k reps, 3 RNG seeds;
margin δ = 1.0 RMSE for TOST):

| contrast | FD001 | FD002 | FD003 | FD004 | pooled (653, inverse-variance) |
|---|---:|---:|---:|---:|---:|
| BE − A | +1.764 [−0.038, +3.538] | −0.057 [−0.737, +0.643] ᵉ | **+1.687** [+0.311, +3.109] | +0.537 [−0.125, +1.243] | **+0.482** [+0.038, +0.927] |
| PE − A | −0.733 [−1.570, +0.073] | −0.162 [−0.793, +0.403] ᵉ | +0.292 [−0.679, +1.194] | −0.086 [−0.531, +0.352] ᵉ | — |

ᵉ TOST-equivalent at δ = 1.0.

Three observations, each in the direction the measurement framing predicts rather than
against it. First, the time-channel effect survives transfer: the test-side pooled
estimate is +0.482 [+0.038, +0.927] (I² = 58%) against the dev-side +0.979 [+0.348,
+1.610] (I² = 70%) — same sign, half the magnitude, both consistent with an effect of
roughly 0.5–1.0 RMSE measured through instruments whose per-subset resolution is ≈1.5
RMSE. Second, subset-level signs are not stable across protocols: FD003 flips from
−1.32 (dev) to **+1.69 (test, significant)**, and FD002 is TOST-equivalent at test. A
single-subset single-protocol estimate of a sub-1-RMSE effect is not a measurement; only
the pooled pattern is. Third, the encoding benefit does not transfer at all: three of
four PE−A intervals lie inside ±0.75 RMSE and none is significant — the PE advantage
lived inside the training distribution, which is where §4.3.3 already placed it.

## What this section establishes, and what it does not

* **Supported.** Removing the time channel costs **8.1% [+1.8%, +14.5%]** RMSE over 260
  held-out FD002 engines. The time index is load-bearing.
* **Supported.** A 0.9% improvement from a different time encoding is **not** resolvable
  at this scale: it does not replicate across folds and it wins on half the engines.
* **Supported.** The mechanism is alignment: destroying it costs 3.5 RMSE, and the
  scrambled control is indistinguishable from removing time entirely.
* **Supported.** A single 80/20 split overstated both contrasts (2.4× and 7×), with three
  mutually agreeing seeds each time.
* **Not supported.** That PE is worse than A — the pooled interval crosses zero in both
  directions.
* **Not supported.** That a 5% improvement could be verified on this benchmark at this
  scale: the smallest resolvable effect measured here is 8–9%, and anything below it sits
  inside the fold-to-fold spread.
