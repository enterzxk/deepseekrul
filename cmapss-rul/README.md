# C-MAPSS Remaining Useful Life Prediction

Current milestone: **TIM track — Stage 1 complete, Stage 2 in progress.**

> **Invalidation notice.** A real ordering bug was found in
> `CRGraphTransformer._smooth_terminal` (tail window was old->new while the
> weights and the `-k` offset are defined new->old), so **every P9/P10 number
> that used the synergy head is void** and must be re-run. Baselines are not
> affected. Fixed in this commit with regression tests
> (`tests/test_audit_fixes.py`). The same commit corrects the Soft-NASA
> surrogate, which invalidates the ABC/ABCE tournament cells.

Stage-1 literature/protocol table: [`docs/LITERATURE_BENCHMARK.md`](docs/LITERATURE_BENCHMARK.md).
Audit + Stage-2 report: [`TIM_AUDIT_AND_STAGE2_REPORT.md`](TIM_AUDIT_AND_STAGE2_REPORT.md).

Previous milestone: **P10 complete.** Native 14x14 sensor-space graph filter
(no pooling, exact identity at init) + adaptive tail slope. It fixes the FD002
graph sign (-0.266 -> +0.257) and passes the FD002 80/20 gate, but the plain
Transformer still wins at 5 seeds (13.289 / 12.707 vs ours 13.753 / 13.709);
P9 remains the FD001 record (13.209 / 12.868 ensemble). See
[`P10_FINAL_VICTORY_REPORT.md`](P10_FINAL_VICTORY_REPORT.md).

Previous milestone: **P9 complete. Sequential CR-GCN -> ConvTransformer ->
synergized monotonic head, trained on 100% of engines for five seeds
(2027-2031). FD001 13.209 single / 12.868 ensemble (best to date, beats the
Transformer); FD002 13.490 / 12.904, still behind the Transformer by 0.51.
See [`P9_DIRECTOR_REPORT.md`](P9_DIRECTOR_REPORT.md); earlier phases in
[`P8_SOTA_REPORT.md`](P8_SOTA_REPORT.md) and
[`P8_EXECUTION_LOG.md`](P8_EXECUTION_LOG.md).**

## Layout

~~~text
src/dataset.py                 C-MAPSS, clip125, windows, metrics, FD002 scaler
src/model.py                   STG-MambaNet (identity-prior graph, SSM/LSTM)
src/loss.py                    PhysicsGuidedLoss (MSE + asymmetric + monotonic)
scripts/run_ablation.py        validation-selected, test-once runner
scripts/tune_validation.py     validation-only P1/P2 tuner (never reads test)
scripts/run_val_tuning.py      P1 grid launcher
scripts/run_p2_tuning.py       P2 loss grid launcher
scripts/lock_hparams.py        writes results/locked_hparams.json
scripts/run_p3_locked.py       five-seed locked FD001 confirmatory runner
scripts/run_fd002.py           FD002 condition-aware external validity
scripts/summarize_ablations.py clip125 main + raw appendix aggregation
scripts/generate_final_figures.py  Phase-5 figures
scripts/generate_final_tables.py   results/final_tables.tex
scripts/generate_morning_report.py MORNING_REPORT.md
archive_root_v0/               archived root-level legacy implementation
docs/FINAL_PROTOCOL.md         frozen protocol
docs/STAGE3_REPORT.md          pilot/historical report (superseded)
MORNING_REPORT.md              final overnight report
results/final_FD001_5seeds/    final five-seed metrics
results/final_FD002_3seeds/    final three-seed FD002 metrics
figures/final/                 final PDF/PNG figures
~~~

## Frozen protocol

- Main label protocol: **clip125** for train/validation/test; raw test metrics
  are stored in the appendix only.
- Standard 14 sensors; training-only normalization.
- FD001: window 50, stride 1, d_model 64, dropout 0.15, weight_decay 1e-4,
  batch 256, lr 1e-3, 80 epochs, patience 20.
- Graph: A = (1-alpha) I + alpha Softmax(ReLU(E1 E2^T)), alpha init 0.1.
- Loss: MSE + Asym(1.0) + Mono(0.05), late weight 2.0.
- Parameter-matched LSTM: 142,806 vs Full 144,265 (1.01%).
- P3 seeds: 101, 202, 303, 404, 505.
- Test suite: 34 collected, OK with 6 skipped while official test files are
  sealed. Unseal explicitly only for authorized evaluation.

## Final FD001 result (clip125, mean +/- std, 5 seeds)

| Variant | RMSE | NASA Score |
|---|---:|---:|
| w/o A-GCN | 14.104 +/- 1.200 | 369.635 +/- 102.242 |
| Identity-Graph (A=I) | 14.689 +/- 0.683 | 392.977 +/- 86.252 |
| w/o SSM (matched LSTM) | 14.405 +/- 0.329 | 359.656 +/- 22.626 |
| w/o Physics Loss | 14.123 +/- 0.868 | 357.238 +/- 56.818 |
| w/o Mono Loss | 14.133 +/- 0.538 | 324.621 +/- 40.663 |
| Full Model | 13.423 +/- 0.487 | 327.119 +/- 59.184 |

Full has the best RMSE and MAE, but not the best NASA score (w/o Mono is
slightly better). A-GCN, SSM, and the physics objective all have positive
paired deltas on FD001; L_Mono improves RMSE but hurts the late-prediction
fraction. FD002 reverses the temporal result: the matched LSTM is better than
Full (14.363/1001.130 vs 14.955/1456.569). Do not claim a universal module win.

## P7 surgical update

P7 removes the delta features, switches the main temporal backbone to
BiLSTM-Attn with zero-init condition FiLM, and scans hollow-graph alpha.
FD001 (5 seeds, alpha=0.15): Full 14.155 / 357.740; w/o Mono Head
13.587 / 313.152; Baseline BiLSTM 14.959 / 362.637. FD002 (1 seed):
Full 14.120 / 799.204; Transformer 12.679 / 785.018. The FD001 <=12.5 gate
is still not met and the FD002 Transformer is stronger. See
P7_FINAL_VERDICT.md.

## Quick start

~~~bash
cd /home/yndx/zxk/bearing-rul-pi
python -m unittest discover -s tests -v
python scripts/generate_final_figures.py --out figures/final
python scripts/generate_final_tables.py
python scripts/generate_morning_report.py
~~~

## Integrity rule

No ablation number is hand-written or selected to fit an expected story.
All final numbers come from metrics.json / test_predictions.csv generated by
the frozen runner. Negative results are reported as negative.
