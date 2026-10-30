# T1 — BE fixed-budget sensitivity: no inflation from early stopping

Question (raised in review of E3): BE stopped at 32–44 epochs because of
`--patience 20`; could the "+2.65 RMSE from removing time" be an artefact of BE
receiving less optimisation?

Design: rerun BE with `--patience 80` (never triggers within 80 epochs), seeds
2027–2029, everything else identical. Preregistered as a sensitivity check; no
change to the main conclusion either way.

| run | best epoch | epochs run | val RMSE | matched RMSE |
|---|---:|---:|---:|---:|
| BE (patience 20, E3) | 15 / 12 / 24 | 35 / 32 / 44 | 13.809 / 13.712 / 14.081 | 14.214 / 16.449 / 15.325 |
| **BEfix (patience 80)** | **15 / 12 / 24** | **80 / 80 / 80** | **13.809 / 13.712 / 14.081** | **14.214 / 16.449 / 15.325** |

Identical to four decimal places. The best checkpoint is reached early and the
extra epochs only add over-fitting (val rises to 17.0–17.7 by epoch 60); because
the run restores the best state before the pseudo-test, patience cannot change any
reported number.

**Verdict: the E3 contrast stands.** BE − A = +2.65 [+1.32, +4.05] is not an
artefact of early stopping, and §4.3 may keep "load-bearing" rather than
"budget-sensitive".

Artifacts: `experiments/e3fix/BEfix_s*/FD002/BEfix_s*/metrics.json`
(each carries `prereg_sha = c8f6d653`, `started_at`, `finished_at`).

Side effect worth recording: this also shows the training trajectory is
deterministic in the epoch count — 35 vs 80 epochs produced the same best
checkpoint, which is what E4 then confirms at the whole-run level.
