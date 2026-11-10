# Research handoff — cmapss-rul-conformal

- Stage: **S9** Paper Evidence Consolidation
- Goal: 把 C-MAPSS RUL 保形区间方法做到可投 IEEE TIM：方法成立、证据完整、论文成稿。
- Checkpoint: `checkpoints\2026-11-09T15-47-51-676Z-S9.json` (state revision 19)
- Updated: 2026-11-09T15:45:03.260Z
- Note: 夜间研究收口：C1/C4/C5/测量协议四线查新全部被占（饱和地图）；发现并修正两个审计问题（FEMTO 自校准泄漏、D032 截断前提未复现）；C13 口径澄清（46.82% 是相对改善的 seed 份额，绝对 Winkler 仅 3.92%）；C12 A4 独立复算一致（+0.246%）。交付 docs/VERDICT_20261110.md。L2 结论：当前资产不足以支撑 TIM 级正向方法贡献。

## Active task
- none

## Next actions
- 【重要】C11 FEMTO 台账数字不可用：freeze_audit.py G1-G2 把 test 行当校准集（自校准泄漏）；独立重算为 6/6、−16.30%（α 固定 0.1），但 6 个轴承只有 2 个前沿能到 0.90；见 docs/AUDIT_FEMTO_C11.md 与 docs/tables/femto_check.csv
- 【重要】D032 的「末窗口虚高覆盖、test-like 后 PICP=0.80」未被复现：删末窗口 0.885→0.886，test-like 均值 0.887；且每引擎 1 窗口时逐机覆盖退化为 0/1；见 docs/MEASUREMENT_CHECK_truncation.md
- 保形 RUL 空间已系统查新为饱和（C1/C4/C5/选项1 均被占）：见 docs/SATURATION_MAP_conformal_RUL.md；按 L2 Q3=没有，不得硬造方法贡献
- 10:00 前交付：饱和地图 + 两个审计更正 + 三个带成本的选项（见 docs/VERDICT_20261110.md）
- 若用户选择测量协议路线，需先把 D032 的说法改成「引擎级覆盖依赖窗口数与聚合方式」，并重新查新
- 本地 experiments/ 已从 a40 恢复（80 preds + 80 mcdropout_seeded + femto 6 轴承）

## Running jobs
- none

## Resume instructions
权威源是 docs/ 不是本文件。恢复顺序：读 AGENTS.md → docs/research_state.md → docs/decision_log.md → docs/results_summary.md → docs/manuscripts/SUBMISSION_CHECKLIST.md，核对产物确实在磁盘上，再继续。research/agent-state/state.json 只提供任务队列与幂等账本，不复制 docs 里的数字（避免双份真相）。

