# ROUNDS — manual-round log (goal-mode substitute)

Mode: `manual` (this runtime exposes no `create_goal`; see `paper/STATUS.json.goal_mode`).
One row per round. A round counts only if it ends with STATUS.json + CLAIMS.md
updated and the five-step contract satisfied.

| round | utc | action | prereg_sha | command | artifacts | gate | verdict | next |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-29T13:20Z | E3 pre-registration (document only) | `c8f6d653a8a08fbd25eda09bf318300c0301a05977c06de878cc796b0051c8d6` | none (no code touched) | `paper/E3_PREREGISTRATION.md` | — | submitted for human review | on approval: backup + `use_pe` patch + equality smoke |

| 2 | 2026-10-29T15:05Z | E3 implementation + 9 runs | `c8f6d653a8a08fbd25eda09bf318300c0301a05977c06de878cc796b0051c8d6` | `python scripts/run_candidate.py --subset FD002 --variant {A,BE,PE} --seed {2027,2028,2029} --split-seed 7 --ref-mode global --epochs 80 --patience 20 --d-model 64 --num-layers 1` | `experiments/e3/*`, `results/dsh/e3/E3_contrasts.json`, `paper/E3_RESULT.md` | non-inferior PASS (upper +0.291) | PE not better (CI crosses 0); BE-A +2.65 significant | E4 from-scratch reproduction |

| 3 | 2026-10-29T15:20Z | T0 bootstrap recheck + corrections | `c8f6d653` (E3 prereg, unchanged) | `python scripts/dsh_e3_bootstrap_stability.py --big 100000 --seeds 9000 9001 9002 9003 9004` | `results/dsh/e3/bootstrap_stability.json`, `paper/E3_AMENDMENT_1.md`, corrected `paper/E3_RESULT.md`, `paper/CLAIMS.md` C14 | **FAIL** (converged upper bound +0.317 > +0.30) | non-inferiority withdrawn; point estimate kept | T1 + T2 |

| 4 | 2026-10-29T16:20Z | T1 + T2 (E4) | `c8f6d653` (T1) / `dc83ea8b` (E4) | `run_candidate.py ... --patience 80` (BEfix x3); clean-dir A_s2027 + `dsh_e4_replay.py` | `experiments/e3fix/*`, `experiments/e4/*`, `results/dsh/e4/replay.json`, `paper/E4_RESULT.md`, `paper/T1_BE_SENSITIVITY.md` | E4 PASS (delta 0.0000); T1 no change | reproducibility established for A; +2.65 stands | E6 audit |

## Circuit breakers in force

* `paper/STATUS.json` sha unchanged at the end of a round → stop.
* Any gate FAIL → that line is closed; move to the next item.
* The same human blocker for 3 consecutive rounds → mark `blocked`.
* Any urge to change a claim to fit a result → stop and change the claim's
  status first.

## Notes

* Round 1 found a scope error in the instruction: the frozen model is
  `src/candidates/model.py`, not `src/model.py` (`run_candidate.py:49`).
  The pre-registration is written against the former.
