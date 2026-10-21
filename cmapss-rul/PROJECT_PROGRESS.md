# C-MAPSS RUL 项目进度

更新时间：2026-10-21

## 当前状态

- **P0 完成**：clip125 协议、archive_root_v0、src 单一生产库、identity prior、trajectory/L_Mono、参数匹配 LSTM、34 tests OK（6 个真实数据测试在 test 封存后 skip）。
- **P1 完成**：9 组架构配置 x 3 开发种子，validation-only；锁定 w50 / dropout 0.15 / wd 1e-4，val RMSE 11.917 ± 0.534。
- **P2 完成**：9 组损失配置 x 3 开发种子，validation-only；Full loss = Asym 1.0 + Mono 0.05；Asym 1.0 为最佳单项。
- **P3 完成**：FD001，5 个新种子 101/202/303/404/505，6 变体，30 次运行。
- **P4 完成**：FD002，3 种子 101/202/303，4 变体，工况感知 KMeans 归一化。
- **P5 完成**：figures/final 四图、results/final_tables.tex、MORNING_REPORT.md。
- **P6 完成（未达 Q2 门禁）**：结构单调头 mono=0.000；hollow graph 反而使 P6 Full 降到 15.168/429.336，BiLSTM 14.145/332.595 更强；详见 Q2_UPGRADE_REPORT.md。
- **P7 完成（部分门禁通过）**：去掉差分、BiLSTM-Attn 主干、alpha=0.15；FD001 Full 14.155/357.740，w/o Mono 13.587/313.152；FD002 Full 14.120/799.204（1 seed）。FD001 未达 <=12.5，且 FD002 Transformer 12.679 更强。详见 P7_FINAL_VERDICT.md。

## P3 最终结果（FD001, clip125, 5 seeds）

| Variant | RMSE | NASA | MAE | Late | Severe | Mono | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| w/o A-GCN | 14.104 ± 1.200 | 369.635 ± 102.242 | 10.352 | 0.530 | 24.60 | 0.465 | 143,816 |
| Identity-Graph (A=I) | 14.689 ± 0.683 | 392.977 ± 86.252 | 10.583 | 0.504 | 21.80 | 0.463 | 143,816 |
| w/o SSM (matched LSTM) | 14.405 ± 0.329 | 359.656 ± 22.626 | 10.379 | 0.468 | 18.60 | 0.313 | 142,806 |
| w/o Physics Loss | 14.123 ± 0.868 | 357.238 ± 56.818 | 10.194 | 0.510 | 21.60 | 0.451 | 144,265 |
| w/o Mono Loss | 14.133 ± 0.538 | 324.621 ± 40.663 | 10.246 | 0.450 | 17.80 | 0.446 | 144,265 |
| Full Model | 13.423 ± 0.487 | 327.119 ± 59.184 | 9.719 | 0.522 | 18.60 | 0.453 | 144,265 |

Raw appendix：Full 14.456 ± 0.468 / 354.360 ± 55.360。

## P4 FD002 结果（clip125, 3 seeds）

| Variant | RMSE | NASA | Late | Severe | Mono |
|---|---:|---:|---:|---:|---:|
| w/o A-GCN | 16.126 ± 1.037 | 1385.647 ± 88.676 | 0.463 | 52.33 | 0.494 |
| w/o SSM (matched LSTM) | 14.363 ± 1.059 | 1001.130 ± 248.916 | 0.422 | 43.33 | 0.447 |
| w/o Physics Loss | 15.512 ± 0.784 | 1513.909 ± 215.770 | 0.528 | 59.00 | 0.491 |
| Full Model | 14.955 ± 0.715 | 1456.569 ± 248.479 | 0.474 | 53.33 | 0.480 |

## 结论

- FD001：Full 的 RMSE/MAE 最佳；A-GCN、SSM、物理损失均有正向 paired delta。
- FD001：Full 的 NASA 不是最佳（w/o Mono 324.621 vs 327.119）；L_Mono 提升 RMSE，但 late fraction 从 0.450 升到 0.522，安全侧不占优。
- FD002：匹配 LSTM 明显优于 Full（14.363/1001 vs 14.955/1457）；SSM 的多工况泛化优势不成立。
- FD002：A-GCN 仍有效（w/o A-GCN 16.126 vs Full 14.955）。
- FD002 raw 误差很大（Full 26.568 RMSE），不要将 raw 指标与 clip125 混表。
- identity-prior 图可能接近单位矩阵，论文必须报告 alpha 与 off-diagonal 质量，不能空泛宣称强动态耦合。

## 关键文件

- MORNING_REPORT.md
- docs/FINAL_PROTOCOL.md
- results/final_tables.tex
- results/final_FD001_5seeds/
- results/final_FD002_3seeds/
- figures/final/
- logs/overnight_progress.log
- results/checkpoint_status.json

## 下一项重要任务

1. 若写论文：以 A-GCN 为主证据，SSM 仅 FD001 支持、FD002 反转；Mono 仅精度收益、不是安全收益。
2. 重做 monotonic 机制（独立 trajectory head / 更强约束），否则不要宣称安全单调性。
3. 重新评估 FD002 工况归一化与模型容量，raw 误差仍高。
4. 不要再基于已观察的 test 分数改模型；如需新实验，用新的 held-out seeds/数据。
