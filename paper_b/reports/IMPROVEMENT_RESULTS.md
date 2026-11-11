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

`[RESULT]`

## 3. Rejected improvements (kept for the record)

`[RESULT]`

## 4. What is not claimed

- No claim of universal superiority: any accepted improvement is stated with
  its subset scope.
- No claim based on a single seed.
- No claim that uses the official test for selection.
