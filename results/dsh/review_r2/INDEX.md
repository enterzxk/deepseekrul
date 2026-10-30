# results/dsh/review_r2 — index

| 文件 | 内容 | 来源/命令 |
|---|---|---|
| `A1_training_audit.json` | 训练 run 审计：逐目录 run 数、epochs、时间窗、并发证据、未知项 | `scripts/dsh_review_r2_audit_runs.py` … `_audit5.py` |
| `A1_runs.csv` | 逐 run 明细（143 行）：run_id、结束时间、种子、epochs、代码哈希（全为 unknown）、依据 | `scripts/dsh_review_r2_runs_csv.py`, `_epochs.py` |
| `A2_test_access_trace.json` | 官方测试读取追溯（脚本 + 产物 mtime） | `scripts/dsh_review_r2_trace_test_access.py`, `_trace_artifacts.py` |
| `A3_corrections.json` | C1–C7 撤回与更正的机器可读版 | 手工整理，对应 REVIEW_R2 §2 |
| `PREREG_R2_C.md` + `.sha256` | C 实验预登记（含运行前 Amendment 1） | 写入后即哈希，实验随后运行 |
| `C_predictions.csv` | 逐 (数据集, 发动机, 截断, 视图, 模型, 折) 预测 | `scripts/dsh_review_r2_views.py`（30 次外层拟合） |
| `C_metrics.csv` | 视图×模型 的 pooled/macro RMSE、NASA、late、severe | `scripts/dsh_review_r2_metrics.py` |
| `C_paired_ci.json` | V2−V1 配对 bootstrap 95% 区间 | 同上 |

退出码：`dsh_review_r2_views.py` 正常结束（产物 300KB，30 次拟合断言通过）；`dsh_review_r2_metrics.py` 正常结束。

| `PREREG_R2_D.md` + `.sha256` | 路线 1 前提检验的预登记 | 写入后哈希 |
| `D_precondition.json` | 该检验结果，**已标 VOID**（代数近恒等 + E 未构造） | `scripts/dsh_review_r2_precondition.py`（10 次拟合） |
