# 早安简报：Overnight Marathon Pipeline

生成时间：2026-10-21

## 结论先行

今晚流水线没有完成 7 个 Stage 的全部实验。原因是 Stage 2 的 5-fold 验证门禁未通过，且共享 GPU 被其他项目持续占满（约 75 秒/epoch），原定 5×5×30 轮超参数网格预计需要约 15 小时。按照任务书“门禁必须通过、测试集只能最后解封”的规则，Stage 3–5 的正式测试与跨数据集结论不能继续，官方 FD001–FD004 测试集仍保持不可读。

没有编造任何测试集数字、基线结果、鲁棒性结果或论文结论。

## 已完成

- 发动机级 5-Fold 划分（100 台训练发动机，每折 80/20）
- Target Scaling：标签除以 125，推理回乘 125
- MinMax 与 Z-score scaler
- FD002/4 的 6 工况 KMeans 簇内 Z-score 路径与单元测试
- Lite_STG_Mamba：A-GCN + 多尺度双向选择性 SSM + 通道门控
- 一阶差分输入（14 raw + 14 delta = 28 通道）
- Dropout、SmoothL1、复合验证指标、退化分箱重采样
- 4 个外部基线 + TCN 的代码

## 真实的 FD001 5-Fold Validation

| Fold | RMSE | NASA Score | MAE |
|---|---:|---:|---:|
| 0 | 12.788 | 8304.347 | 9.115 |
| 1 | 14.080 | 7984.492 | 9.529 |
| 2 | 15.444 | 16658.906 | 10.702 |
| 3 | 15.390 | 11820.870 | 11.396 |
| 4 | 14.119 | 8183.320 | 9.891 |
| Mean ± std | 14.364 ± 0.984 | 10590.387 ± 3351.168 | 10.127 ± 0.822 |

单折最好结果为 Fold 0 的 12.618，但 5-Fold 平均为 14.364，未达到 Stage 2 的 12.8 门禁。

## 未完成项

- Stage 2 的 5×5 γ×τ 网格：启动后因共享 GPU 预计超过 15 小时，已终止。
- Stage 3–5：未运行基线矩阵、完整消融、测试集评估、不确定性、鲁棒性和少样本迁移。
- Stage 6：只生成了真实可得的 fig7 和 5-fold RMSE 补充图。
- 工作区中另有来源不明的旧图 fig1/2/3/4，未纳入本报告，也不作为今晚证据。

## 关键文件

- src/dataset.py
- src/model.py
- src/loss.py
- src/baselines.py
- scripts/train_dev_fold.py
- scripts/run_5fold_dev.py
- scripts/run_hyperparam_grid_5x5.py
- scripts/run_benchmark_matrix.py
- scripts/final_blind_eval.py
- scripts/statistical_tests.py
- results/FINAL_PAPER_REPORT.md
- paper/main.tex
- figures/fig7_learned_sensor_graph_heatmap.png
- figures/supp_fold_validation_rmse.png

## 下一步

1. 在 GPU 空出后完成 Stage 2 的 5×5 网格并重跑 5-Fold。
2. 分析 Fold 2/3 的困难发动机和 0.98 的折间方差。
3. 在验证集上试验每台发动机早期基线归一化。
4. 若 12.8 仍不可达，应重新定义论文主张，而不是继续堆模块。
5. 配置冻结后才允许一次性解封官方测试集。

## 完整性声明

所有数字来自磁盘上的 metrics.json 或 CSV。缺失阶段明确标记为未运行。
