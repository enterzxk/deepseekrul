# Research handoff — cmapss-rul-conformal

- Stage: **S9** Paper Evidence Consolidation
- Goal: 把 C-MAPSS RUL 保形区间方法做到可投 IEEE TIM：方法成立、证据完整、论文成稿。
- Checkpoint: `checkpoints\2026-11-09T04-05-19-461Z-S9.json` (state revision 12)
- Updated: 2026-11-09T04:01:25.884Z
- Note: goal round 1：论文级证据脚本 paper_evidence.py（13/13 断言，读 seeded 权威表）；CLAIM_LEDGER（C1–C15，含允许/禁止措辞）；OUTLINE（IJPHM 稿骨架）；NEIGHBOR_MATRIX（含核实等级）。关键修正：MC-Dropout 权威表为 mcd80_cells_seeded.csv（D023），首轮表未设种子。下一步：全文矩阵补齐、IJPHM 当前要求、图脚本、正文。

## Active task
- none

## Next actions
- MC-Dropout 80 格已完成并收口（D023）；写稿时遵守读法约束：均值 +9.5% 可说、中位 −1.0% 不可说成“多数格更好”、FD004 −1.8% 在噪声内、对齐口径只作诊断
- 按 D022 贡献边界重写稿件：测量/评估定位；尺度异质性可复现（方向翻转 0/80），可承载 E5 类贡献
- 核验 IJPHM 官方范围/文章类型/原创性要求，更新 docs/VENUE_FIT.md
- 补最近邻全文差异矩阵（CRULP closed access；Yan 审计包已读包内文档）
- 正文严格分开名义水平与覆盖对齐，标注 E3 单 seed、E7 事后、FEMTO 范围

## Running jobs
- none

## Resume instructions
权威源是 docs/ 不是本文件。恢复顺序：读 AGENTS.md → docs/research_state.md → docs/decision_log.md → docs/results_summary.md → docs/manuscripts/SUBMISSION_CHECKLIST.md，核对产物确实在磁盘上，再继续。research/agent-state/state.json 只提供任务队列与幂等账本，不复制 docs 里的数字（避免双份真相）。

