# Gate 0 Report — Causal Lifetime Prior Value (Stage A verdict)

> Frozen criteria: `gates.json` (gate0.criteria), pre-registered before any
> arm was trained. Verdict script: `scripts/summarize_gate0.py` (engine-
> clustered paired bootstrap, 5,000 reps, pooled across the 5 outer folds).
> Date: 2026-11-03. Stage A = seed 101. Stage B (seeds 202/303) runs now and
> must confirm before numbers are frozen for the paper.

## Verdict: **GO_TO_GATE1** (4/4 PASS)

| id | comparison (FD002 unless noted) | ΔRMSE | 95% CI | rule | verdict |
|----|--------------------------------|-------|--------|------|---------|
| G0.1 | M1 vs M0 | **−0.279** | [−0.498, −0.053] | upper < 0 | **PASS** |
| G0.2 | M1_shuf vs M1 | **+0.312** | [+0.114, +0.517] | lower > 0 | **PASS** |
| G0.3 | M1_const vs M1 | **+0.279** | [+0.072, +0.485] | lower > 0 | **PASS** |
| G0.4 | M1 vs M0 (FD001 guardrail) | −0.103 | [−0.468, +0.258] | upper < +0.5 | **PASS** |

Pooled val RMSE (seed 101): FD002 — M0 12.559, M1 12.280, M1p 12.356,
M1_shuf 12.592, M1_const 12.560. FD001 — M0 9.915, M1 9.811.

## Mechanism reading

1. **The strictly causal prior helps**: concat of (μ_T, σ_T) improves the
   PE backbone by 0.28 RMSE on FD002 with CI excluding zero — with a prior
   that provably consumes no future information (staircase cutoff ≤ t, four
   leakage unit tests passing, nested cross-fitting).
2. **The gain is individualization, not capacity**: permuting μ_T across
   engines within the fold (M1_shuf) *removes* the benefit and overshoots
   M0 (+0.31 vs M1, CI > 0). A constant median lifetime (M1_const) does the
   same (+0.28 vs M1, CI > 0). Both controls have identical architecture and
   parameter count as M1 — the information content of the per-engine pairing
   is the only difference.
3. **No guardrail violation**: FD001 (single condition, weaker prior,
   corr@c100 ≈ 0.59–0.71) shows a small non-significant improvement and the
   CI upper bound is far inside the +0.5 harm threshold.
4. **M1p (tau encoding) ≈ M1** at Stage A (12.356 vs 12.280 pooled): the
   clock encoding is not yet better than plain concat; Gate 1's H-comparisons
   will decide whether the clock formulation earns its place (and module B/C
   will test internal conditioning).

## Supporting diagnostics

- **Prior quality by cutoff** (outer-val, FD002): corr(μ_T, T) rises from
  ≈0.27 at c=50 to ≈0.93–0.96 at c≥150; MAE falls from ≈35 to ≈9–11 cycles.
  The prior is a late-life-sharpening signal — consistent with the mechanism
  that gains concentrate on mid/late-life windows.
- **Prior-implied RUL** (clip(μ_T − t, 0, 125)) on the same val windows:
  FD002 17.59 vs PE 11.97 — and vs **H1 (HGB direct RUL) ≈ 31.3** (FD001 H1
  ≈ 33.1 vs PE 9.44). The *same* causal features that predict RUL directly
  at RMSE ≈ 31 predict the total lifetime at corr ≈ 0.8: lifetime identity
  is the more learnable quantity, and the clock converts it into RUL-relevant
  structure far better than direct regression.
- Flagged windows (t < min cutoff, median fallback): none in evaluation
  windows (window ends start at t=50 = min cutoff); the fallback exists for
  protocol completeness.

## Leakage audit (blocking, all green)

| test | property | result |
|------|----------|--------|
| T1 | future sensor perturbation ⇒ μ_T(t) unchanged | PASS |
| T2 | outer-val label change ⇒ outer-train μ_T/σ_T unchanged (same outer fold) | PASS |
| T3 | outer-val label change ⇒ that engine's own μ_T unchanged | PASS |
| T4 | ∀ windows: consumed cutoff ≤ window end (no future interpolation) | PASS |

(See `tests/test_prior_causality.py`; run on the real FD002 pipeline with
controlled synthetic perturbations.)

## Stage A scope & honesty notes

- Stage A verdict is seed-101 only; Stage B (seeds 202/303, all 5 arms ×
  FD002, M0/M1/M1p × FD001) is running and must reproduce all four passes
  before any number is quoted in the paper.
- Development-val protocol (dev engines run to failure; windows include
  end-of-life RUL=0). Official test remains sealed.
- H1 uses the prior estimator's hyperparameters without RUL-specific tuning
  (frozen choice, no outer-val tuning); a train-RMSE underfitting check is
  queued to preempt "sandbagged baseline" concerns.

## Next (Gate 1, per frozen plan)

1. H2/H3 arms (nested inner-CV weight selection; OOF H1 feature).
2. Module B/C single zero-init experiments vs M1p with shuffled-tau control.
3. Stage B confirmation → freeze Gate 0 numbers.
