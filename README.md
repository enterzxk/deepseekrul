# C-MAPSS RUL 区间预测研究仓库

研究工程仓库：方法代码、逐 run 实验证据、决策记录与稿件都在这里。**所有历史提交保留**，
被删除的文件可以从 git 历史恢复（见文末「仓库整理记录」）。

## 先读这四份（按顺序）

| # | 文件 | 作用 |
|---|---|---|
| 1 | [`docs/research_state.md`](docs/research_state.md) | 当前目标、已完成步骤、待执行任务、恢复点 |
| 2 | [`docs/decision_log.md`](docs/decision_log.md) | 已作出的决策（不得重开已决事项） |
| 3 | [`docs/results_summary.md`](docs/results_summary.md) | 权威实验记录（含已撤回结论及原因） |
| 4 | [`docs/manuscripts/SUBMISSION_CHECKLIST.md`](docs/manuscripts/SUBMISSION_CHECKLIST.md) | 稿件完成度与缺口 |

## 目录

```
src/          方法代码（lifetime prior / punet / cet / ecs / twoscale / candidates）
paper_b/      论文 B 的代码与逐 run 证据
  results/      每个 run 的 metrics.json 与 val/test 逐点预测（论文数字的来源）
  experiments/  配置、网格、冻结协议
  hc/           训练与评测入口
  archive/      接管前的历史代码与结果（不作为论文证据）
scripts/      复算与出表脚本
docs/         状态、决策、稿件、表、图、wiki
  paper_b_edc/  论文 B 的 24h 归因冲刺记录
  paper_b_hc/   HC 线的规格、审计与 gate 记录
  manuscripts/  稿件（ijphm_v2 等）与投稿清单
  tables/       论文表格（由 scripts 生成）
  figures/      论文图
experiments/  实验产物（ida 等）
tests/        因果性/泄漏测试
research/     agent 机读状态（state.json / decisions.jsonl）
```

## 证据规则

- 论文里的每个数字都应能从 `paper_b/results/` 或 `docs/tables/` 复算；口径以
  `docs/results_summary.md` 为准。
- 不提交训练权重（`*.pt` / `*.pth` / `*.ckpt`）与大型原始数据；权重可从代码与种子重训。
- 结论被推翻只追加「已撤回 + 原因」，不删旧记录、不改写历史。

## 分支

- `main` —— 当前主干（与 `paper-b-contribution-attribution` 同一提交）。
- `paper-b-contribution-attribution` —— 论文 B 贡献归因线。
- `punet-interval` / `ncmapss-ds02` / `codex/paper-b-ncmapss-response-20261113` /
  `codex/paper-b-open-source-reproduction` / `paper-b-span` —— 在用或在审的旁支。

## 仓库整理记录

**2026-11-14**：默认分支前移到当前主干；删除 3 个已合并/废弃分支
（`rul-calibration`、`tim-idc-rul`、`rul-filter-probe`）；`codex/paper-b-open-source-reproduction`
上的 257 个 `checkpoint.pt`（112 MB）从树中移除（权重可重训，仓库策略禁止入库）；
删除会话临时目录与 LaTeX 构建日志。被删内容均在 git 历史中，旧分支另存于本地
`deepseekrul-backup-20261114.bundle`。
