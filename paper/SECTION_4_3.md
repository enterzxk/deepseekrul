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
