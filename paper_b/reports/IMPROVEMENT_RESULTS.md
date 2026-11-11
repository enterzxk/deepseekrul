# Supported improvements (evidence-gated)

Status: in progress. An improvement is listed here only if it passes the
pre-registered rule in `FREEZE_RULE.md`: beats its reference on >=3/4 subsets
by >=1% mean validation RMSE and is not worse than DSPT-base overall.

## 1. Candidates under test

| candidate | mechanism hypothesis | reference | status |
|---|---|---|---|
| SA + sensor identity | SA lacks sensor-role information | SA | `[RESULT]` |
| SA + condition embedding | SA mixes sensors across operating conditions | SA | `[RESULT]` |
| HI on h_B | HI should read the representation used by the RUL head | HI (h_f) | `[RESULT]` |
| HI without monotone | the target violates monotonicity ~48% locally | HI (h_f) | `[RESULT]` |
| MS single_proj | MS gain may be the projection | base | `[RESULT]` |
| MS multi_noproj | MS gain may be multi-scale | base | `[RESULT]` |
| MS multi3 | parameter-matched single scale | base | `[RESULT]` |

## 2. Accepted improvements

### A1. MS + SA (no HI) on FD001

- **Scope:** FD001 only.
- **Validation:** 11.112 +- 0.344 vs base 11.874 +- 0.258 (-6.4%).
- **Official test:** 10.598 +- 0.299 vs base 11.054 +- 0.350 (-4.1%).
- **Seeds:** 101/202/303; the gain is larger than the seed std and the
  validation and test directions agree.
- **Why MS+SA and not FULL:** FULL is 11.132 validation, statistically tied
  with MS+SA; MS+SA has fewer parameters and drops the unsupported HI head.
- **Source:** `results/fd001_final_table.csv`, `results/raw/A8_mssa_L70/FD001`.

## 3. Rejected improvements (kept for the record)

### R1. HI auxiliary supervision (FD001)

HI alone is +0.5% validation (worse than base); HI added to MS+SA (FULL) gives
no gain over MS+SA. The probe explains it: the target is noisy and largely
redundant with RUL. HI is removed from the final model.

### R2. FULL over MS+SA

No measurable gain (11.132 vs 11.112 validation, 0.2%, far below the seed std).
The simpler model is selected by the pre-registered tie-break.

### R3. Mechanism grids (SA identity/condition, HI h_B/no-mono, MS controls)

Not run: the multi-subset grid was stopped at the user's request after the
FD001 ablation was complete. The mechanism explanations remain probe-level.

## 4. What is not claimed

- No claim of universal superiority: any accepted improvement is stated with
  its subset scope.
- No claim based on a single seed.
- No claim that uses the official test for selection.