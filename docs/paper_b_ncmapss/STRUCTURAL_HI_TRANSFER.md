# Structural HI transfer to N-CMAPSS

Status: **design note, nothing measured**. Written before any DS02 result so the
expectation cannot be fitted to the outcome afterwards.

## Scope: the current round tests no HI

The frozen N0–N3 ladder has **no HI arm**. N3 reads its switches from
`FREEZE_ABLATION_SEED101.json` → cell `A8_mssa_L70`, and that cell is
`ms_mode=full, sa_mode=basic, hi_source=none` (verified against
`paper_b/experiments/grids/ablation8.json`; the freeze manifest's own note says
"final model remains MS+SA").

Consequences, stated before any number exists:

- This round produces **no cross-dataset HI result**. Nothing in it may be
  reported as evidence for or against the HI head, in either direction.
- The test designed below is **pre-registered for N4/N5**, which D043 defers
  until the C-MAPSS mechanism is frozen.
- Any table or sentence pairing "DS02" with "HI gain" is wrong until an N4/N5
  run exists. This matters because the registry records `freeze_source`: a later
  reader could otherwise infer from "N3 = frozen DSPT" that HI was in scope.

The rest of this file is therefore a **deferred pre-registration**, not a plan
for the current round.

## What "structural HI" means here

In the C-MAPSS line, `hi` is an auxiliary supervision head: the model is asked to
predict a monotone health indicator alongside RUL. The audit
(`docs/paper_b_baselines/BASELINE_PROTOCOL_AUDIT.md` §7) found its independent
effect to be **stably near zero** on C-MAPSS — −0.03 capped, −0.05 raw on FD001,
with the sign never flipping but the magnitude never clearing seed noise.

That is the prior this section must test, not assume.

## Why DS02 could differ

1. **Real degradation trajectories.** C-MAPSS is simulated with a fixed fault
   progression; DS02 carries multi-flight, variable-condition histories. A
   monotonicity prior may be more informative when the trajectory is noisier.
2. **The cycle axis is weaker.** On C-MAPSS the cycle feature carries ~70% of
   the gap over a vanilla model. If DS02's cycle axis is less informative, the
   headroom for a structural prior is larger.
3. **Flights are natural segments.** A health indicator defined per flight is a
   more meaningful object than one defined per truncated window.

## What is forbidden

- Defining the HI target from the complete lifetime in a way that makes it a
  monotone transform of the label. That would be label leakage dressed as
  auxiliary supervision.
- Selecting the HI formulation after seeing DS02 test results.
- Reporting a HI gain without a paired, per-fold comparison against the identical
  model without the head.

## Test to run (after the data dictionary is fixed)

The same ablation as C-MAPSS, on the frozen N3 configuration: N3-with-HI versus
N3-without-HI, identical seeds, identical folds, paired per fold.

Report: paired delta per fold, mean, dispersion, and whether the sign is
consistent. If the sign flips across folds, the answer is "no detectable
effect" — the same conclusion C-MAPSS reached, and a legitimate one.

## Falsification criteria

- If N3-with-HI fails to beat N3-without-HI in at least 5 of 6 folds, the
  transfer claim for the HI head is rejected.
- If the gain appears only under one label convention (capped vs raw), it is a
  convention artefact and is reported as such.
- If the HI target turns out to be a monotone function of the label, the whole
  experiment is void.
