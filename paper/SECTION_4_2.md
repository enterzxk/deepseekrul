# 4.2 Resolution: what a comparison at this scale can and cannot detect

Section 4.1 established that the held-out engines dominate the variance of any
contrast. This section turns that into a number a reader can use before running an
experiment, and then shows — with a direct experiment — that the naive version of
that number is too optimistic.

## 4.2.1 The minimum detectable effect

For a paired contrast on `n` held-out engines, with per-engine differences `d_i`
having standard deviation `s`, the smallest effect detectable at 80% power and a
two-sided 5% level is

```
MDE(n) = (z_0.975 + z_0.80) · s / √n = 2.80 · s / √n
```

`scripts/mde.py` computes this from the archived per-engine residuals. On FD002:

| contrast | s (per engine) | MDE at 52 | at 100 | at 260 |
|---|---:|---:|---:|---:|
| A vs BE (time removed) | 8.42 | **3.27** | 2.36 | **1.46** |
| A vs PE (bounded encoding) | 6.69 | 2.60 | 1.87 | 1.16 |

On FD001 (100 engines, 20 per split) the same calculation gives MDE 2.38 at 20
engines and 1.84 at 100.

Two consequences, both used later:

* **A 5% improvement is not verifiable at this scale.** At the FD002 baseline of
  ≈13 RMSE, 5% is 0.65 — below the 52-engine MDE of 2.6–3.3, and at the edge of
  the 260-engine MDE of 1.2–1.5. This is why the project's original acceptance
  criterion (≥5% over the strongest baseline, CI excluding zero) could not be met
  by *any* method on this benchmark, independently of method quality.
* **Engine count is the only cheap lever.** Seeds do not enter the formula as a
  variance term (their share is 2.4%, §4.1); the standard error is set by how many
  distinct engines are held out.

## 4.2.2 The naive projection fails: a direct test

The formula predicts that holding out all 260 FD002 engines instead of 52 should
shrink the interval by √5 ≈ 2.24, from roughly ±1.3 to roughly ±0.6. That
prediction was tested directly by 5-fold cross-validation
(`paper/E10_CV_PREREGISTRATION.md`, `paper/E10_RESULT.md`): every engine held out
exactly once, 3 seeds, 24 new runs.

| | single split (52 engines) | five folds (260 engines) |
|---|---:|---:|
| ΔRMSE, PE − A | −0.908 (−7.7%) | **−0.126 (−0.9%)** |
| 95% CI | [−2.03, +0.29] | **[−1.07, +0.80]** |
| CI half-width | 1.16 | 0.93 |
| projected half-width from 1/√n | — | 0.52 |
| engines where PE wins | — | 51% |

The interval did shrink, but by **20%**, not by 55%. The reason is visible in the
per-fold estimates:

| fold | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Δ (PE − A) | −0.904 | **+1.699** | −1.023 | −0.477 | −0.031 |

The between-fold standard deviation is **1.10 RMSE**. A five-fold interval cannot
be narrower than what that heterogeneity allows: `1.10/√5 = 0.49` per standard
error, which is exactly the observed 0.475. In other words:

> **The resolution of a k-fold comparison is bounded below by the between-fold
> heterogeneity of the effect, not by the number of engines.** Adding folds helps
> until `s_between/√k` meets `s_within/√(k·n_fold)`; after that, more engines buy
> nothing.

This is a stronger statement than "more engines are better", and it is the reason
§4.2.4 recommends reporting fold-level estimates alongside any pooled one.

## 4.2.3 Two contrasts, one design: where the boundary sits

The calibration is only meaningful if the same design is applied to effects of
different size on the same engines. That is what E10 and E10b provide: identical
pipeline, identical folds, identical seeds, identical interval recipe, different
design change.

| contrast | single split (52) | five folds (260) | 95% CI (five folds) | verdict |
|---|---:|---:|---|---|
| **BE − A** (remove the time channel) | +2.693 (+21%) | **+1.133 (+8.1%)** | [+0.245, +2.023] | **reproduced** |
| **PE − A** (bounded vs linear time) | −0.908 (−7.7%) | **−0.126 (−0.9%)** | [−1.073, +0.795] | **not reproduced** |

So the smallest effect this design resolves on FD002 is of order **1.1–1.3 RMSE
(8–9%)**, in agreement with the independent MDE calculation of §4.2.1 (1.16–1.46 at
260 engines). Below that, an interval computed on 260 engines still crosses zero —
not because the estimator is imprecise, but because the fold-to-fold heterogeneity
of the *effect itself* (SD ≈ 1.1) is as large as the effect.

The single-split column is the more uncomfortable half of the table. Both contrasts
were **overstated** on fold 0 — 8.1% reported as 21%, 0.9% reported as 7.7% — and in
both cases the fold that produced the number is the easiest fold for the baseline
arm (A scores 12.68 on fold 0 against 13.5–15.1 elsewhere). Two contrasts cannot
establish that this bias has a consistent direction; the paper reports it as an
observation and does not generalise it. What can be said is weaker and sufficient:
**a single 80/20 split is not a noisy estimate of the population effect, it is an
estimate of one particular holdout**, and its error is comparable to the effects the
literature routinely reports.

## 4.2.4 What this does to a published claim

Consider the claim "design change X improves RMSE by 7.7%". On fold 0 it is true:
12.679 → 11.775. On fold 1 the same change is **−11%** (13.494 → 15.192). The two
folds are both valid 80/20 splits of the same 260 engines, both trained with the
same protocol and the same three seeds, and within each fold all three seeds agree
with that fold's answer.

A reader who saw only fold 0 would report a 7.7% improvement with three consistent
seeds. A reader who saw only fold 1 would report an 11% regression. The pooled
answer is −0.9% with an interval crossing zero. **No amount of seed replication
distinguishes these cases**; only holding out different engines does.

The same comparison run in the opposite direction — on the *large* effect — gives a
number that survives: 8.1% with an interval clearing zero. The design is therefore
not broken; it is calibrated. It can support an 8% claim and cannot support a 1%
one.

## 4.2.5 Buying resolution: three levers, ranked by cost

| lever | cost | effect on the interval | evidence |
|---|---|---|---|
| **More held-out engines (k-fold)** | k × training | ×0.8 at k = 5, floored by between-fold heterogeneity | §4.2.2, §4.2.3 |
| **More cuts per engine** | zero training, one extra forward pass | **×0.66–0.75** at fixed engine count | §4.5, `scripts/eval_resolution.py` |
| More seeds | k × training | ×0.97 going from 3 to 10 seeds | §4.1 |

The second lever is the only one that is free, and it is the one the literature
uses least. Scoring each held-out engine at two cut points instead of one narrows
the interval by a quarter to a third at identical engine count:

| contrast | single cut | two cuts | ratio |
|---|---:|---:|---:|
| FD002 A vs BE | 3.362 | 2.528 | **0.75** |
| FD002 A vs PE | 3.508 | 2.303 | **0.66** |
| FD001 A vs BE | 4.635 | 3.282 | **0.71** |

Caveat, stated in the same breath: these two-cut numbers come from an exploratory
re-analysis (`paper/E7_RESOLUTION_NOTE.md`), not a pre-registered experiment. The
direction is mechanically necessary — more observations per engine, same engines —
but the magnitudes are empirical and subset-dependent. On FD001 the four-cut
variant is *worse* than two cuts, because requiring four valid cuts drops the
shortest-lived engines from the pool (60 → 42 engines). More cuts is not
monotonically better; it trades within-engine correlation for between-engine
coverage.

## 4.2.6 A multiplicity claim that had to be withdrawn

An earlier version of this section stated that "across 21 paired comparisons, none
survives Bonferroni correction". That table is withdrawn: the seven model cells it
compared were produced by three experiment scripts that no longer exist, and **no
per-engine predictions were stored**, so the table cannot be re-derived from any
artefact (`paper/E6_STATS_AUDIT.md`; `paper/CLAIMS.md` C3).

What can be said instead is what this section now contains: a per-contrast MDE, an
explicit interval for each contrast that has per-engine predictions, and the
demonstration that a 0.9-RMSE effect is inside the fold-to-fold noise of this
benchmark. The multiplicity statement would have been the paper's strongest
rhetorical claim and it is the one claim that turned out to rest on nothing — which
is itself an instance of the failure mode the paper documents.

## What this section establishes, and what it does not

* **Supported.** At 52 FD002 engines the MDE is 2.6–3.3 RMSE; at 260 it is 1.2–1.5.
  A 5% improvement (≈0.65) is below both.
* **Supported.** Five-fold cross-validation over 260 engines shrinks the interval
  by only 20%, not the 55% that 1/√n predicts, because between-fold heterogeneity
  (SD 1.10) sets the floor.
* **Supported.** The smallest effect this design resolves on FD002 is 1.1–1.3 RMSE
  (8–9%): an 8.1% effect reproduces (BE − A) while a 0.9% effect does not (PE − A),
  on the same folds, seeds and estimator.
* **Supported.** Two cuts per engine at fixed engine count narrow the interval to
  0.66–0.75 of the single-cut width.
* **Not supported.** That more cuts are always better (FD001 four-cut is worse), or
  that these ratios transfer to other datasets.
* **Not supported.** Any claim resting on the withdrawn 21-comparison table.
