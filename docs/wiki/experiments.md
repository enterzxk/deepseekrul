# 实验登记

> 每次**正式**实验登记一行：配置、判定门、结果摘要、证据路径。数字以 docs/results_summary.md 为准。
> 不登记"跑跑看"；没有判定门的运行不算正式实验。

## 登记表

| ID | 日期 | 目的/假设 | 配置（数据/折/seed） | 判定门 | 结果摘要 | 证据路径 | 状态 |
|---|---|---|---|---|---|---|---|
| E001 | 2026-10~11 | 主方法 vs 基线（H001 主检验） | C-MAPSS FD001-004，5 折，2 seed（+2 追加） | 覆盖对齐 Winkler 配对胜率 | 29/40 胜，中位 −8.4%；名义水平 19/20 胜 | experiments/punet_matrix/（79/80 格） | 完成，缺 1 格 |
| E002 | 2026-11 | 跨域泛化 | FEMTO 6 轴承 | 同口径 Winkler | 5/6 胜，−12.2% | experiments/femto/ | 完成 |
| E003 | 2026-11-08 | 宽度缩放诊断（D001 决定 C） | 现有 checkpoint，a ∈ {0,0.5,1,2,4} | 存在 a≠1 显著更优 | a=1 最优（47.5），放大更差；路线关闭 | docs/tables/width_scaling.csv | 完成，假设证伪 |
| E004 | 2026-11 | 置换消融（A4） | 20 折置换标签 | 与真标签差异显著 | 均值 +0.2%，无信号 | docs/tables/ | 完成 |
| E005 | 2026-11 | 覆盖诊断（引擎级/阶段级） | 20 折 | 截面 PICP 达标但引擎级失效 | 引擎级最小覆盖 0.50–0.71 | docs/tables/engine_*.csv | 完成 |

## E-MCD80 · MC-Dropout 补到 80 格（D009 第 ③ 项 / D019 A2）

- **日期**：2026-11-09 01:43 起（a40 后台）。
- **目的**：把"dropout 采样标准差作为每样本条件尺度"这一操作化补到完整 80 格，
  支撑内部报告中的限定结论；**不作为对 UACQR/CRULP 机制的评价**（D019 B3）。
- **配置**：`scripts/run_mcd80_backfill.sh`。Stage A：40 格已有 checkpoint（seeds 42/7）
  直接跑 `run_mcdropout.py`（T=30）；Stage B：40 格（seeds 2024/13）先
  `run_e2e_cqr.py --heads point --save-ckpt` 重训再跑。产物 `experiments/punet_mcd80/`，
  不覆盖冻结的 `experiments/punet_matrix/`。
- **判定门（启动前预注册）**：名义水平为主口径（δ=0.1，可部署；D015 已删覆盖对齐）；
  折级 n=20；胜≥60% 折且均值≤−5% 且 CI 排除 0 → 撤回"无整体收益"；输≥70% 折 → 限定反例成立；
  否则只作子集条件陈述。**完整性门**：重训点模型的 `preds_point.csv` 必须复现冻结版
  （max|y_lo|≤1e-3），否则配对无效、升级 L2。
- **结果（80/80 格，seeded 权威口径）**：名义水平下尺度单独效应（同中心对照 `const_mc`）
  43/80 胜、均值 +9.5%、中位 −1.0%；vs CQR 8/80；实际覆盖 0.881 vs 0.899（偏低 1.8pp）；
  逐子集 FD001 +18.8%、FD002 −7.1%、FD003 +28.1%、FD004 −1.8%；
  折级 11/20 胜、均值 +9.5%、中位 −1.4%、bootstrap [+1.3%, +19.0%]（描述性）。
- **判定门**：第三档（异质）→ 只作子集条件陈述。
- **可复现性**：首轮未设种子（重跑 picp 差 ≤7.6e-3）；已加 `R.set_seed(seed)` 并整批重跑，
  抽检差全 0.0。两遍独立抽样的噪声底线：主指标 |Δ| 中位 0.63pp / p90 3.20pp，
  **方向翻转 0/80**。
- **完整性**：重训 40 格点模型与冻结 `preds_point.csv` 逐位一致（max|Δ|=0.000e+00）。
- **协议回归检查**：重算 `const`/`cqr` 与 `docs/tables/cells_all.csv` 逐位一致（max|Δ|=7.1e-15）。
- **证据路径**：`docs/tables/mcd80_cells_seeded.csv`（权威）、`docs/tables/mcd80_cells.csv`（首轮）、
  `scripts/analyze_mcd80.py`、`scripts/compare_mcd_draws.py`、`scripts/verify_mcd80.py`、
  a40 `experiments/punet_mcd80_seeded/`。
- **状态**：完成（80/80）。

## 待登记

- 补跑 FD004 f4 s13 后登记为 E001 的补齐行，不改 E001 原行。


## 规范

1. 一行一实验；同一实验追加 seed 用"配置"列说明，不新开行除非判定门不同。
2. 结果摘要必须与 results_summary.md 一致；不一致先修权威源。
3. 失败/证伪实验同样登记，状态写"完成，假设证伪"。


## E-H004 · 因果前缀 + 引擎-阶段覆盖正则（FD001 单子集试跑）

- **配置**：冻结中心（baseline 的 (lo+hi)/2），只训半宽头（MLP 49→64→32→1，softplus）；
  特征 = 42 维窗口传感器统计 + cutoff + mu_T + sigma_T + centre + 3 维中心轨迹滚动统计；
  覆盖匹配在校准发动机上用二分法求加性 q（目标 0.90）；评估用未参与拟合的 40% 发动机。
- **判定门**：见 hypotheses.md §H004（4 条）。
- **结果（FD001，seed 42，5 折均值）**

| 变体 | 覆盖 | 阶段短缺 | p10 | 宽度 | Winkler(α=.1) |
|---|---|---|---|---|---|
| CQR（基线） | 0.869 | 0.107 | 0.734 | 29.4 | **41.95** |
| eng_pinball | 0.891 | 0.085 | 0.751 | 32.2 | 49.54 |
| grouped_noprefix | 0.891 | 0.085 | 0.751 | 33.1 | 50.59 |
| **h004（前缀+分组）** | 0.888 | 0.090 | **0.770** | 30.5 | 45.84 |
| prefix_only | 0.894 | 0.085 | **0.782** | **29.9** | 46.33 |
| uniform（宽度匹配） | 0.873 | 0.108 | 0.748 | 30.5 | 43.86 |

- **观察**
  1. 所有训练头都降低阶段短缺（0.084–0.090 vs 0.107），且**优于宽度匹配的均匀放宽**（0.108）——不是单纯变宽
  2. 但 Winkler 全部劣化 9–21%，**违反判定门第 3 条（≤2%）**
  3. 分组惩罚项没有增量：eng_pinball 与 grouped_noprefix 相同（0.085）
  4. 前缀信息主要改善 p10（0.782 vs 0.751）并降低宽度（29.9 vs 32.2）
- **状态**：**未达判定门**。且 eval 覆盖未对齐（CQR 0.869 vs 头 0.891），需按主协议的
  "沿覆盖前沿对齐"重做比较，才能判定第 1、2 条。
- **下一步**：把比较改成覆盖对齐（扫加性 q 得 (coverage, shortfall) 前沿），再跑 4 子集 × 4 seeds。

## Paper B public-source reproduction matrix — 2026-11-11

Registered objective: full upstream methods under author and DSPT-matched
protocols; exact engine split/seeds/cap/NASA rules and zero window-search
budget in `docs/paper_b_baselines/PROTOCOL.md`. Complete matrix, commands,
per-run config hashes/checkpoints/predictions and evaluation gates:
`paper_b/experiments/open_source_reproduction_20261111/`.
Source/training/final metric audits are indexed in
`docs/paper_b_baselines/INDEPENDENT_AUDIT.md`; authoritative numeric tables
are generated in OPEN_SOURCE_REPRODUCTION_REPORT.md, not duplicated here.

## PB-OS-MATCHED-20261112 · complete

Purpose: rerun complete public methods under currentDSPT allprefix/engine-mean
selection, commonFP32/cap125/split andofficialendpoint rules. Registered
models/seeds/windows andvalidation-onlyauthor-ceiling amendment in
paper_b/baselines/open_source_matched_20261112/{PROTOCOL,AUTHOR_LIMIT_POLICY}.json.
Evidence/decision: PB-OS-006–010;94physical/168logical, independentauditPASS.
Primaryauthor-ceiling and40sensitivity separate; extra-dataTwoPhase separate.
Numeric source/report: paper_b/experiments/open_source_matched_20261112/ and
docs/paper_b_baselines/MATCHED_COMPARISON_REPORT_20261112.md.
No changes toPaperA/DSPT/oldresults.
