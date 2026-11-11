# Archive index (Paper B)

Immutable reference: git tag **`archive/paper-b-pre-cleanup-95e729e`** (commit
`95e729e53b5f80004c1038030807a5f9172533d6`).

## Status legend

| status | meaning |
|---|---|
| ACTIVE | current, protocol-fixed code and evidence |
| SUPPORTED | improvement with a controlled, replicated gain |
| ARCHIVED | historical exploration, kept for traceability |
| INVALID | affected by a protocol defect; never cite as performance evidence |

## What moved where

| old path | new path | status |
|---|---|---|
| `docs/paper_b_code/run_dspt.py` | `paper_b/archive/legacy_2026_11/code/run_dspt.py` | INVALID (repo copy is non-runnable; server copy is the P0 working tree) |
| `docs/paper_b_code/train_vanilla.py` | `paper_b/archive/legacy_2026_11/code/train_vanilla.py` | ARCHIVED |
| `docs/paper_b_code/eval_vanilla.py` | `paper_b/archive/legacy_2026_11/code/eval_vanilla.py` | ARCHIVED |
| `docs/paper_b_code/PHASE1_AUDIT.md` | `paper_b/archive/legacy_2026_11/code/PHASE1_AUDIT.md` | ARCHIVED (historical audit, partly superseded) |
| `docs/paper_b_code/P0_FIXES.md` | `paper_b/archive/legacy_2026_11/code/P0_FIXES.md` | ARCHIVED (claims code fixes that were never committed) |
| `docs/paper_b_code/ABLATION_DEFINITIONS.md` | `paper_b/archive/legacy_2026_11/code/ABLATION_DEFINITIONS.md` | ARCHIVED |
| `docs/paper_b_results/*` | `paper_b/archive/legacy_2026_11/results/*` | INVALID as paper evidence (see below) |
| `docs/paper_b_results/RESULTS_SUMMARY.md` | `paper_b/archive/legacy_2026_11/results/RESULTS_SUMMARY.md` | ARCHIVED; OOF numbers INVALID |
| `docs/paper_b_results/VANILLA_ABLATION_REPORT.md` | `paper_b/archive/legacy_2026_11/results/VANILLA_ABLATION_REPORT.md` | ARCHIVED; cross-model comparison INVALID |

## Historical experiment grids

| grid | location on lab3090 | cells | status | reason |
|---|---|---:|---|---|
| OOF 5-fold | `experiments/OOF*` (61 dirs) | 60 | **INVALID** | truncation points sampled from the official test RUL distribution (`RESULTS_SUMMARY.md` §0); test information leaked into validation |
| DSPT 32-cell ablation | `experiments/ABL_*` (32 dirs) + `experiments/ABLTEST` | 32 | **INVALID** as a table | single seed/fold; test predictions produced by an evaluator that cannot load the checkpoints; validation early stopping on window-level RMSE |
| Vanilla 96-cell ablation | `experiments/VB_*` (96 dirs) | 96 | **ARCHIVED** | 3 seeds, but window-level early stopping; its cross-model claim vs DSPT is confounded (different evaluators and feature sets) |

## Numbers that may be quoted only with a caveat

| number | source | caveat |
|---|---|---|
| DSPT FD001 base test 10.6036 / FULL 12.6984 | `results/abl_test2.csv` | single seed 101, fold 0; evaluator provenance broken; test observed many times |
| DSPT FD002/3/4 base 11.40/11.95/12.36 | `results/abl_test2.csv` | same |
| Vanilla FD001 BASE 15.91 ± 0.69 | `results/vb_test.csv` | L=70, 3 seeds; window-level early stopping; not comparable to DSPT numbers above |
| OOF FD001 10.118 ± 0.960 | `results/oof_summary.csv` | **do not cite**: truncation leakage |

## Restore commands

```bash
# whole pre-cleanup tree
git worktree add /tmp/paperb_pre archive/paper-b-pre-cleanup-95e729e

# one file
git show archive/paper-b-pre-cleanup-95e729e:docs/paper_b_code/run_dspt.py > /tmp/run_dspt_legacy.py
```

Server artefacts (checkpoints, logs, raw predictions) remain under
`lab3090:/home/user/rulb/experiments/` and `/home/user/rulb/logs/`.
