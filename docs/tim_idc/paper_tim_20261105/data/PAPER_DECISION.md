# Paper decision after the one-shot test

## Trigger

The goal says: if the official test shows the FULL-vs-M0 gain is insufficient, shrink to a measurement/empirical paper. The full-budget official test satisfies that condition:
- FULL-M0 significant only on FD003;
- FD001 ns (CI upper +0.010), FD002 ns (+0.348, worse), FD004 ns (-0.524, CI crosses 0);
- on FD002, FULL is significantly worse than OPS (+0.590).

## Decision

The current evidence does **not** support a "new method with better results" paper. It supports a rigorous measurement/empirical paper:

**Working title:** *When does a causal lifetime prior help RUL prediction? A budget-controlled study on C-MAPSS*

**Claim:** A causal, engine-specific lifetime prior derived from sensor histories is informative for lifetime estimation, but its end-to-end RUL benefit is budget- and subset-dependent; under a full training budget the sensor prior does not reliably beat the PE baseline, and on the multi-condition FD002 the age/condition prior is better.

**Contributions:**
1. Causal, nested-cross-fitted lifetime prior with staircase cutoffs and leakage tests.
2. Sensor-information measurement analysis: at fixed ages, sensor change/slope features carry lifetime information beyond age/condition (8/8 engine-paired CIs below zero).
3. Budget-controlled end-to-end evaluation: lightweight vs full-budget, four subsets, three seeds, one-shot official test; the benefit is not robust.
4. A negative result with a clear mechanism: more sensor features are not uniformly better; the age/condition prior is more robust on FD002 under the full budget.

**What not to claim:**
- SOTA or robust improvement over PE;
- uncertainty calibration (Gate 2 Spearman 0.034, sigma ~1e-14);
- conformal reliability as a main contribution (intervals 33-83 cycles wide);
- CET as a competitive baseline (adapted CET is 16-22; official CET-MSA is a different protocol).

## Remaining work

1. Consolidate the official-test tables and the measurement tables into the paper's results section.
2. Write the method (causal prior, staircase, nested cross-fitting, injection), measurement analysis, budget-controlled results, and limitations.
3. Figures: prior construction, sensor-vs-OPS measurement, budget interaction, official-test forest/CI plot.
4. Baselines: label adapted CET vs official CET-MSA; keep the historical E22a row separate.
5. Audit appendix: gates, one-shot protocol, sealed predictions, hashes.
6. Internal review and submission.

## Alternative

If the requirement is a "new method with clearly better results", the current line cannot deliver it. The alternatives are (a) reset and develop a new method, missing the 11-07/08 deadline, or (b) change venue/target to a measurement/PHM venue that values the negative/mechanism result. This is a user decision, not a technical one.
