# Gate 0 Multi-Seed Confirmation (Stage B)

Date: 2026-11-03. Purpose: confirm the frozen Gate 0 verdict (seed 101,
`experiments/ida/gate0/gate0_verdict.json`) on two additional training seeds
(202, 303) with identical arms, folds, and frozen criteria. Aggregation is the
same `scripts/summarize_gate0.py` (engine-clustered bootstrap, 5000 reps,
read-only over `val_predictions.csv`); verdict files:
`experiments/ida/gate0/gate0_verdict_s{101,202,303}.json`.

## Results (all 5 outer folds per seed, FD002 n=259 engines, FD001 n=80)

| Criterion | Rule | seed 101 | seed 202 | seed 303 |
|---|---|---|---|---|
| G0.1 M1 vs M0 (FD002) | upper<0 | **-0.279** [-0.498,-0.053] PASS | **-0.347** [-0.607,-0.078] PASS | **-0.448** [-0.681,-0.216] PASS |
| G0.2 M1_shuf vs M1 (FD002) | lower>0 | **+0.312** [+0.114,+0.517] PASS | **+0.310** [+0.058,+0.558] PASS | +0.149 [-0.105,+0.387] FAIL |
| G0.3 M1_const vs M1 (FD002) | lower>0 | **+0.279** [+0.072,+0.485] PASS | **+0.272** [+0.022,+0.510] PASS | +0.188 [-0.042,+0.422] FAIL |
| G0.4 M1 vs M0 (FD001 guard) | upper<+0.5 | -0.103 [-0.468,+0.258] PASS | +0.060 [-0.282,+0.402] PASS | +0.255 [-0.276,+0.787] FAIL |
| Verdict | — | GO_TO_GATE1 | GO_TO_GATE1 | STOP |

## Honest reading

1. **The core claim is robust**: G0.1 (prior injection helps M0) passes on all
   three seeds with a consistent or growing effect (-0.279 / -0.347 / -0.448).
   All 12 sign checks across seeds and criteria point in the predicted
   direction.
2. **The attribution controls lose significance on seed 303**: shuffled
   (+0.149) and constant (+0.188) controls still *hurt* relative to M1
   (direction correct 3/3 seeds), but their 95% CIs cross zero. The
   individualization attribution stands on seeds 101/202 and is
   directionally supported, not universally significant, on 303.
3. **The FD001 guardrail is inconclusive on seed 303**: point estimate
   +0.255 with CI upper +0.787 exceeds the frozen +0.5 no-harm bound, so on
   this seed we cannot exclude a moderate FD001 degradation. FD001 was never
   a target condition for the prior (age-dominated lifetimes); this is a
   robustness limitation, not a sign reversal.

## Disposition (gating discipline)

- The frozen, pre-registered verdict is the seed-101 one: **GO_TO_GATE1**.
  Stage B was a confirmation layer; criteria are NOT re-interpreted,
  re-metriced, or relaxed post hoc.
- Gate 1 continues (H1/H2/H3 arms running). A multi-seed confirmation issue
  in Stage B does not by itself halt the pipeline, because the frozen main
  verdict did not fail.
- Manuscript presentation rules (binding):
  - Report all three seeds; never quote only 101.
  - G0.1 may be stated as "significant on 3/3 seeds, effect -0.28 to -0.45".
  - G0.2/G0.3 must be stated as "direction-consistent on 3/3 seeds,
    individually significant on 2/3" — no stronger claim.
  - G0.4 must carry the seed-303 caveat explicitly.
  - Any pooled-across-seeds analysis is descriptive robustness only and must
    be labeled as post hoc; it cannot resurrect a failed per-seed criterion.
- If Gate 1 outcomes depend on the attribution claim, re-examine seed-303
  variance sources (fold-level diagnostics) BEFORE any further claim, and
  document in a new file (never edit this one).
