# CIDO-RUL ledger

Branch: `cido-rul` (forked from `ecs-rul`; ECS code and results retained as comparators).

| item | file | sha256 (first 16) | frozen at |
|---|---|---|---|
| Architecture novelty audit | `paper/CIDO_NOVELTY_AUDIT.md` | 754b951ac2ed00dc | 2026-11-01T14:55+08:00 |
| Method design (v2, post-audit) | `paper/CIDO_METHOD_DESIGN.md` | 6a80c3757af90d6e | 2026-11-01T14:58+08:00 |
| F2 gate result | `paper/CIDO_F2_RESULT.md` | 3c74bc7596ed3080 | 2026-11-01T15:30+08:00 |
| F1 gate result | `paper/CIDO_F1_RESULT.md` | a5ad8e643b0f536f | 2026-11-01T16:20+08:00 |

## Gate status

| gate | what it decides | state |
|---|---|---|
| **F2** (Prop. 2, FPT under filtered drift) | whether there is a mathematical claim at all | **PASS**, amended: coefficient 2 confirmed, two asymptotes located, gap band `τ_c/τ_f ∈ [0.1, 3]` identified |
| **F1** (Prop. 1, h/b identifiability) | whether the bias-state operator has an identifiable role | **FAILED** — `paper/CIDO_F1_RESULT.md`. The rank claim is true instantaneously (affine min eigenvalue −7e-11 vs nonlinear 1.8e4) but the trajectory system is identified by the *dynamics* asymmetry, not by the observation nonlinearity; with a valid test the affine model is twice as accurate. Bias-state operator demoted to engineering component; **no architecture-level claim remains** |
| **H1–H6** | mechanism and calibration | **not run** — needs trained models |

## What the audit changed

Two of the three originally proposed operators are **withdrawn as novel claims**:
the shared/specific innovation split (factor analysis / fault geometry: Apley & Shi 2001,
Dunia 1996, Lock 2013) and the evidence-conditioned transition with rate-as-state plus
first-passage RUL (Si et al. 2013 MSSP is structurally identical). What survives:

1. the `h`-vs-bias identifiability argument, available only with a nonlinear observation
   model (Prop. 1);
2. the first-passage law under a filtered drift (Prop. 2, now verified);
3. per-channel **bias states with their own dynamics**, filtered jointly with `h`, which
   FDI (discrete hypotheses), factor analysis (i.i.d. specific term) and deep SSMs (no
   per-channel model) each leave unattended.

## Code state

* `src/candidates/model.py` — `CandidateModel.encode()` extracted from `forward`.
  Verified bit-exact: 41,858 parameters, `max|Δ| = 0.0` on prediction / trajectory /
  embedding against the pre-refactor module, and `encode()` + head reproduces `forward()`.
* `scripts/run_candidate.py` — `build_windows(..., return_meta=True)` returns
  `(unit, end_cycle, observed_length)` per window; default arity unchanged, verified
  against the pre-change output (`x` identical, windows ordered within unit).
  This is the contract a recurrence needs; without it the engine axis does not exist.

## Next

The architecture claim is now **NO-GO**: three candidate operators have been removed
(shared/specific split by the literature audit, evidence-conditioned transition by the same
audit, bias states by F1). What remains is Prop. 2 and the gap-band finding, which support a
*methodological* contribution (how to compute the RUL distribution when the rate is driven by
the observations) rather than a new network operator. Decision on which path to take is
recorded in `paper/CIDO_F1_RESULT.md` §7.
