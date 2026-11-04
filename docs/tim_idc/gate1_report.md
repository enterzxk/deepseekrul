# Gate 1 Report: Consumption Mechanism

Date: 2026-11-04. Verdict file (script-generated, read-only aggregation):
`experiments/ida/gate1/gate1_verdict.json`. Criteria: `gates.json` → `gate1`
(G1.1/G1.2/G1.3, frozen 2026-11-03). Statistics identical to Gate 0
(engine-clustered pooled bootstrap, 5000 reps, FD002, 259 engines, seed 101).

## Verdict: **GO_TO_MODULE_BC** — M1 and M1p both beat H1, H2, H3

| Criterion | Pair | RMSE (PE / H) | ΔRMSE | 95% CI | Verdict |
|---|---|---|---:|---|---|
| G1.1 | M1 vs H1 (HGB direct RUL) | 12.280 / 29.449 | **-17.152** | [-18.746, -15.587] | PASS |
| G1.1 | M1p vs H1 | 12.217 / 29.449 | **-17.217** | [-18.805, -15.636] | PASS |
| G1.2 | M1 vs H2 (M0/HGB blend) | 12.280 / 12.505 | **-0.226** | [-0.442, -0.003] | PASS |
| G1.2 | M1p vs H2 | 12.217 / 12.505 | **-0.290** | [-0.575, -0.004] | PASS |
| G1.3 | M1 vs H3 (PE + H1 feature) | 12.280 / 12.595 | **-0.313** | [-0.543, -0.091] | PASS |
| G1.3 | M1p vs H3 | 12.217 / 12.595 | **-0.378** | [-0.656, -0.098] | PASS |

Decision rule satisfied twice over: both M1 and M1p beat all three H arms.

## Reading

1. **G1.1 is a magnitude statement, not a marginal one**: HGB direct RUL
   regression on the same causal features is catastrophically worse
   (29.45 vs 12.28). The PE backbone is doing irreducible work; this also
   quantifies why "just predict RUL with the prior's features" was never a
   viable alternative (consistent with the Gate 0 diagnostic ladder:
   31.3 vs 12.0).
2. **G1.2**: the inner-CV weight selection collapsed to w≈0.95–1.0 (PE
   dominant; `h2_weights.json`) — the blend itself already rejected HGB, and
   even the residual HGB share hurts. Caveat for the manuscript: M1 vs H2
   CI upper is -0.003, i.e. the exclusion of zero is real but thin; report
   the exact interval, do not round it into a stronger claim.
3. **G1.3**: injecting H1's RUL guess as an input feature (H3) is worse than
   consuming the lifetime prior through the clock (M1/M1p) — H1's prediction
   acts as a noise feature. Lifetime *identity* + tau conditioning is the
   right consumption channel.
4. M1p (tau sinusoidal encoding, +640 params) edges out M1 (12.217 vs
   12.280) — consistent with the clock design hypothesis; the margin is
   small and should be presented as point-estimate preference, not a
   confirmed ranking.

## Disposition

- Per the frozen pipeline, proceed to **Module B / Module C**: one small
  zero-init experiment each (<10% extra params), compared against M1p, with
  a shuffled-tau control; keep only if CI excludes 0 (gates.json →
  `modules_BC`).
- Gate 0 multi-seed context: seeds 101/202/42 all pass G0.1 robustly; seed
  303 and seed 42 show the attribution controls (G0.2/G0.3) and the FD001
  guardrail (G0.4) losing per-seed significance while keeping sign — carried
  into the manuscript per `gate0_multiseed.md`.
