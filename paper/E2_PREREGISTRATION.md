# E2 — Pre-registration: can a monotone health index make the lifespan estimator usable?

Status: **frozen before any run** (2026-10-29). Any change after the first result
is an amendment and must be logged in §7 with a timestamp.

## 1. Question

R3.1 established that the deployable lifespan scale fails because the prefix
estimator is unusable: `corr(T̂,T) = 0.196`, `median T̂/T = 1.90`, and 154/154
prefix windows produced no admissible estimate at cutoffs 60/100/140
(`REVIEW_20261022_R3.md` §H.2, §J.6). The stated cause is that the observable
health index is not monotone, so any prefix extrapolation is dominated by
operating-condition noise.

**Hypothesis (falsifiable):** replacing the 14-sensor mean with an
operating-condition-corrected, sign-aligned FPCA1 health index raises
`corr(T̂,T)` on held-out engines to **≥ 0.5**, which is the gate fixed in §J.7
for allowing `T̂` anywhere near a candidate.

## 2. A degeneracy that must be stated up front

The task book's descriptive criterion is "held-out median Spearman ≥ 0.3".
Once a **per-engine monotone projection** is applied, within-engine Spearman
between the index and cycle is 1 **by construction**, so that number carries no
information about quality. We therefore:

* report the monotonicity score of the **unconstrained** index as the
  descriptive quantity the book asks for (H0/H1);
* use `corr(T̂,T) ≥ 0.5` on held-out engines as the **operative decision gate**,
  because it is the quantity the downstream feature actually needs and it was
  fixed before this experiment existed.

This is a clarification of an ambiguous criterion, not a relaxation: it is
strictly harder to pass than a monotonicity score that the constraint forces to 1.

## 3. Data and splits

* FD002 **training engines only** (260 units). No test file is read.
* Splits: seeds 2027/2028/2029, using the same permutation rule as
  `scripts/dsh_r3_four_arms_v2.py` (`rng.permutation`, `nval = max(8, n//5)`),
  so the holdout engines are identical to R3.1 and the numbers are comparable.
* Fit scope: cluster statistics, PCA loadings, failure level `L`, and any
  calibration map are fitted on the **training engines only**.

## 4. Health-index variants (all share cluster-z preprocessing)

`z`: per-row standardisation by KMeans-6 cluster of `op_1..op_3`, statistics
fitted on training engines (identical to R3).

| id | construction |
|---|---|
| **H0** | `PCA1(z)` fitted on training rows, sign aligned so `corr(h, cycle) > 0` on training |
| **H1** | H0 + per-engine EMA (α = 0.3), the same filter family the frozen pipeline uses |
| **H2** | H1 + per-engine isotonic projection (non-decreasing) |
| **H3** | H1 + cross-engine percentile calibration against training engines at matched normalised age |

H0 is the literature-standard baseline; H1–H3 are the three cheapest monotone
repairs. No variant may use information after the cutoff.

## 5. Estimator and metrics

Estimator (unchanged from R3 so that any difference is attributable to `h`):

```
T̂ = c + (L_s − h(c)) / slope(1..c),  clipped to [1.2c, 400]
L_s = mean end-of-life h over training engines in the same degradation-rate tertile
```

Reported per variant, on the 52 held-out engines × 3 seeds:

1. monotonicity score of `h` (mean fraction of non-negative first differences);
2. median within-engine Spearman(`h`, cycle) — descriptive only, see §2;
3. **`corr(T̂, T)`** (primary);
4. `median T̂/T` and its IQR;
5. fraction of windows that fall back to the pooled constant.

## 6. Decision rule (fixed now)

* **PASS** if the best variant reaches `corr(T̂,T) ≥ 0.5` **and** fallback ≤ 5%.
  Then `T̂` may be re-tested on the neural arms (a separate pre-registration).
* **FAIL** otherwise. The failure is reported as the falsification evidence for
  §4.4 and the lifespan-scale route is closed for this health-index family —
  no further tuning of `L`, cutoffs, or smoothing after seeing the result.
* Variant selection uses **training-set** `corr(T̂,T)` only; the holdout is
  evaluated once, for all four variants, and all four are reported.

## 7. Amendments

**Amendment 1 — 2026-10-29, before the first successful run.**
The estimator's rate class (`slow/mid/fast`, used to pick `L_s`) was written as
"over training engines in the same degradation-rate tertile" without saying how a
**held-out** engine is classified. Using its full-trajectory rate would be
future information, so the class is now the engine's **prefix slope at the same
cutoff**, with tertile thresholds taken from training engines at that cutoff.
This is the deployable version and is strictly harder (noisier slope → more
misclassification). No result had been produced when this was changed: the first
attempt aborted on a `KeyError` before writing any metrics.

**Amendment 2 — 2026-10-29, same time.** H3 is **not implemented** in this run:
the cross-engine percentile calibration needs a per-cycle training reference that
does not exist yet, and inventing one after seeing H0–H2 would be an
unpreregistered design choice. H3 is deferred and reported as "not run".
