# E4 — Pre-registration: from-scratch reproduction (G3 core evidence)

Frozen 2026-10-29 23:40 UTC+8, before any E4 run. This file's sha256 is passed to
every E4 run via `--prereg-sha` and recorded in `metrics.json`.

## Question

Does the current code (with the additive `use_pe` branch, default off) reproduce a
run that was produced by the pre-patch code, when started from an empty working
directory?

## Design

| | |
|---|---|
| reference | `experiments/e3/A_s2027/FD002/A_s2027/metrics.json` (RMSE 13.086 matched, val 12.543) |
| replay | load `best_model.pt`, rebuild `A`, re-evaluate the matched-cutoff set; expect max abs pred diff ≤ **1e-3** |
| from scratch | clean dir `/home/yndx/zxk/e4_clean` (fresh `src/`, `scripts/`, `data/` symlink), same command, same seed |
| acceptance | \|ΔRMSE_matched\| ≤ **0.15** vs the reference, and \|Δval\| ≤ 0.15 |
| diagnostics on failure | report max\|Δpred\|, per-epoch loss trace distance, and whether the divergence is in data order or in optimisation; **do not tune to fit** |

## Why both parts

The replay isolates *code + weights* (deterministic given the saved state). The
from-scratch run additionally exercises data ordering, dataloader workers, and CUDA
non-determinism, which the replay cannot see.

## Rule on failure

If \|ΔRMSE\| > 0.15 the paper must report the run-to-run spread explicitly and G3 is
**not** satisfied by this run alone; a second seed is then needed before any
reproducibility claim is written. No hyperparameter may be changed to close the gap.
