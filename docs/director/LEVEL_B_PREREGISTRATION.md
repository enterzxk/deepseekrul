# Level B 预登记（冻结，不得用于调参）

日期：2026-10-21
状态：**预登记，未执行**。FD003/FD004 官方测试与标签保持 `chmod 000` 封存。

## 1. 冻结项

| 项 | 冻结值 |
|---|---|
| 开发划分种子 | `split_seed = 7`（引擎级 80/20） |
| Level B 划分 | `split_seed = 2027`（与开发划分不同的引擎集合，预先生成、不得更改） |
| 模型种子 | 2029, 2030, 2031 |
| 评价协议 | Protocol C（绝对网格 60/100/140/180/220 + 存活规则）+ matched-cutoff（`cutoff_seed=2026`） |
| 指标 | RMSE、MAE、NASA、late fraction、severe-late(>10)、逐发动机 signed error |
| 主统计 | 发动机级配对 Bootstrap（20000 次），报告 95% CI |
| 主表口径 | pooled RMSE（全部 cutoff × 种子合并）；macro 均值单列 |
| 门禁 | ① matched-cutoff RMSE 优于 tf；② FD001 [0,25) severe-late ≤ tf 的 1.1 倍；③ 单调性 N/A 规则不变 |

## 2. 执行约束

- Level B 只在候选架构冻结后执行**一次**；结果不用于任何后续调参。
- 若未通过，必须记录失败并重新开启新一轮探索，不得反复宣称同一集合"仍然独立"。
- FD003/FD004 仅在 Director 明确批准后解封。

## 3. 脚本哈希

评价脚本 `scripts/run_candidate.py`、`scripts/error_analysis.py`、
`scripts/audit_stats.py` 的 SHA256 需在解封前记录于此（下一轮补）。
