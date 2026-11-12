# Research handoff — cmapss-rul-conformal

- Stage: **S9** Paper Evidence Consolidation
- Goal: 把 C-MAPSS RUL 保形区间方法做到可投 IEEE TIM：方法成立、证据完整、论文成稿。
- Checkpoint: `checkpoints\2026-11-11T09-08-07-075Z-S9.json` (state revision 22)
- Updated: 2026-11-11T08:39:55.464Z
- Note: Paper B takeover 2026-11-11: legacy archived (tag archive/paper-b-pre-cleanup-95e729e), paper_b/ protocol-fixed package committed (7f26387, b24158f, 09076b8). 199-run grid running on lab3090 2x3090 (ablation8 96 + vanilla 24 + nocycle 12 + ms 18 + sa 24 + hi 18), attribution 7 runs on a40. Legacy checkpoints reproduce archived predictions (corr>=0.9935). Mechanism probes done (HI Spearman -0.79..-0.92, ~48% local violations; sensor hetero 10-19x on FD002/FD004; ACF ~25 cycles). Freeze + test pending ablation completion. Authoritative status: paper_b/reports/CURRENT_STATUS.md

## Active task
- paperb-ablation8-20261111: Paper B protocol-fixed DSPT 8-config ablation (96 runs, 3 seeds x 4 subsets, L=70)

## Next actions
- 10:00 前已交付 docs/VERDICT_20261110.md（饱和地图 + 两个审计更正 + 三个带成本选项）
- C12 A4 已独立复算一致：均值 +0.246%、中位 +0.930%、11/20 变差、FD004 −3.02%
- C13 口径已澄清：46.82% 是相对改善的 seed 方差份额；绝对 Winkler 仅 3.92%（变异主要来自子集）
- 台账 C1/C3/C4/C5/C6/C7/C8/C12/C13 已全部独立复算通过；C11 已更正（泄漏→6/6、−16.30%、2/6 可达 0.90）
- 待用户 10:00 在 A（测量协议）/B（3–6 个月方法程序）/C（改期刊或改贡献类型）中选一；未选前不启动训练
- 若选 B：第一步是找新问题而不是新方法；先过无 GPU 新颖性门，再用 L2 批准的 20 GPU-h 先导

## Running jobs
- paperb-ablation8-20261111: Paper B protocol-fixed DSPT 8-config ablation (96 runs, 3 seeds x 4 subsets, L=70) (running)

## Resume instructions
权威源是 docs/ 不是本文件。恢复顺序：读 AGENTS.md → docs/research_state.md → docs/decision_log.md → docs/results_summary.md → docs/manuscripts/SUBMISSION_CHECKLIST.md，核对产物确实在磁盘上，再继续。research/agent-state/state.json 只提供任务队列与幂等账本，不复制 docs 里的数字（避免双份真相）。

