# E17 — Baseline capacity sensitivity

Question a reviewer will ask: were the five published architectures strawmen? Every E11 arm
ran at `d_model = 88`; the candidate ran at 64 with 41,858 parameters.

Design: `paper/E11_ARCH_PREREGISTRATION.md` Amendment 2, decision rules frozen before the
runs. The two architectures that matter are re-run at `d_model = 128`, one seed (2027), all
five folds, everything else identical.

| arm | d = 88 | d = 128 | change | parameters |
|---|---:|---:|---:|---:|
| tf (Transformer) | **15.459** | 16.029 | **+0.57 (worse)** | 144,146 → 301,826 |
| cnn1d | 19.754 | 19.636 | −0.12 (better) | 85,801 → 176,001 |

Pooled over 260 engines, seed 2027, same folds.

**Verdict against the pre-registered rules.**

* "tuned `tf` beats A by >1.0 → withdraw the candidate's advantage": not triggered. The gap
  *widens* to 2.00, in the candidate's favour.
* "tuned `cnn1d` improves by >2.0 → withdraw the +5.73 contrast": not triggered. It moved
  0.12, which is one eighth of the resolution limit.

So the between-tier contrasts are not a capacity artefact. Doubling the width of the two
baselines that matter moves them by −0.12 and +0.57, against a resolution limit of 1.3–1.5
at this engine count: **the baselines were not under-parameterised, and the candidate won
with a quarter to a third of their parameters.**

Two limits on this conclusion. Only width could be varied: none of the five baseline
classes exposes a depth parameter, so "capacity" here means width alone. And one alternative
configuration per architecture is not a hyperparameter search, so the claim is "not rescued
by the obvious capacity change", not "cannot be tuned to win".

## Ancillary finding: a determinism check

The first launch of this experiment was a no-op (§Amendment 2a). Its seven completed runs
reproduced their E11 counterparts to six decimal places in `best_val_rmse`, which is a
stronger repeatability check than §7's single-configuration replay: **seven independent
process invocations, two days apart, same numbers to 1e-6.**

Artifacts: `experiments/e17b/*`, `results/dsh/e16/e17b_capacity.json`.
