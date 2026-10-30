# E2 — Result: a monotone health index does **not** rescue the lifespan estimator

Run: 2026-10-29, CPU only, FD002 training engines (208/52 split, seeds 2027–2029),
cutoffs 60/100/140, 1383 (engine, cutoff) observations.
Artifacts: `results/dsh/e2/E2_health.json`, `E2_health.csv`;
script `scripts/dsh_e2_health_index.py`; design `paper/E2_PREREGISTRATION.md`
(incl. Amendments 1–2, both logged before this run produced a number).

## Verdict: **FAIL** (pre-registered gate `corr(T̂,T) ≥ 0.5`)

| variant | monotonicity | median Spearman(h, cycle) | corr(T̂,T) all | corr(T̂,T) no-fallback | median T̂/T | fallback |
|---|---:|---:|---:|---:|---:|---:|
| H0  cluster-z → PCA1 | 0.505 | 0.377 | 0.084 | 0.319 | 1.92 | 3.7% |
| H1  + EMA(0.3) | 0.518 | 0.674 | 0.157 | 0.346 | 1.92 | 3.9% |
| H2  + isotonic | **1.000** | **0.976** | 0.280 | **0.347** | 1.92 | 0.9% |

Per-seed `corr(T̂,T)` without fallbacks: H2 = 0.325 / 0.348 / 0.364.
H3 was not run (Amendment 2).

## What this means

1. **The repair does what it says and changes nothing that matters.** Forcing
   monotonicity takes the score from 0.505 to 1.000 and Spearman from 0.377 to
   0.976, yet `corr(T̂,T)` moves only 0.319 → 0.347. Monotonicity of the index is
   *not* the binding constraint on the estimator.
2. **The 1.9× bias is structural.** Every variant keeps `median T̂/T = 1.92`
   (R3.1 measured 1.90 on a different code path), i.e. the gap between the
   end-of-life level `L` and the prefix slope, not the noise in `h`.
3. **The double gate is what caught it — and that is the transferable lesson.**
   Read on its own, the index-level criterion ("held-out median Spearman ≥ 0.3")
   is satisfied by H0 at 0.377; the downstream gate (`corr(T̂,T) ≥ 0.5`) fails at
   0.319. So a paper reporting only index-level quality would have declared
   success on a feature that cannot be used. The lesson for §4.4 is therefore
   **not** "our criterion failed" but "an index-level score must always be
   reported next to a downstream-usability score, because the index constraint
   can be satisfied while the quantity of interest is not".

## Consequence for the paper

* The lifespan-scale route is **closed for this health-index family**: no further
  tuning of `L`, cutoffs, or smoothing after seeing this result (§6 of the
  pre-registration). `T̂` stays out of any candidate (the §J.7 gate is not met).
* §4.4 now has three layers: oracle headroom (+8.23, significant) → deployable
  estimator fails (+0.08 vs cycle) → the obvious repair fails too (this run).
  That is a stronger, and honest, story than "we fixed short-lived engines".
* C7/C8 in `paper/CLAIMS.md` are unchanged; C13 is added for this result.

## Threats to validity

* Probe preprocessing (no `OperatingConditionScaler`); absolute levels are not
  comparable with the frozen pipeline.
* Only FD002, only three cutoffs, only three seeds; 52 held-out engines.
* `L` is still a per-stratum constant. A per-engine end-of-life model is a
  *different* hypothesis and was not pre-registered here — it must not be
  introduced as a rescue of this one.
