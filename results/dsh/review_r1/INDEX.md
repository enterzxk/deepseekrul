# results/dsh/review_r1 产物索引

共 37 个文件。由 Round 156 生成索引；文件内容即服务器上的原始产物。

**规范指标文件**：`metrics_clipped_round146.csv`（裁剪到 [0,125] 的标准口径）。
`full_metrics_table.csv` 为旧口径（未裁剪），保留供追溯。

## 子目录

* `rul_archive/predictions/` — 51 个逐窗口预测 CSV（三个切分 × 各单元 × 各种子）
* `rul_archive/weights/` — 32 个对应的 `best_model.pt` 权重
* 详见 `rul_archive/README.md`（含命名规则与正确复算方法）

## 文件

| 文件 | 内容 |
|---|---|
| `advantage_vs_rul.json` | 逐发动机优势与 RUL 的关系 |
| `band_significance.json` | 按 RUL 分段的显著性检验 |
| `bias_variance.csv` | 按发动机的偏差/方差分解明细 |
| `c_interaction.json` | C 假设：状态相关交互检验（p=0.625，被否定） |
| `c_interaction_null.json` | C 假设的零假设分布 |
| `c_response_shift.csv` | 工况响应随退化变化的原始量 |
| `c_response_shift_summary.json` | 上述的汇总 |
| `clustered_bootstrap_vs_ref.csv` | 引擎聚类 bootstrap 对照 |
| `conformal_coverage.csv` | 共形区间覆盖率（80.3%/91.7%） |
| `conformal_onesided.csv` | 单侧共形区间 |
| `cp_post_onset.csv` | CP 标注在起点之后的样本 |
| `cp_vs_baseline_common_truth.csv` | CP 与基线在共同真值上的对比 |
| `crossfit_c.json` | 发动机级 cross-fitting 结果 |
| `cutoff_sensitivity.csv` | 截断敏感性（Protocol C） |
| `ensemble_metrics.csv` | 跨单元集成的指标 |
| `error_concentration.csv` | 误差质量按 RUL 区间的分布（91% 在 60–126） |
| `full_metrics_table.csv` | 完整指标表（旧口径，未裁剪） |
| `gap_check.json` | 交付与目标的差距检查 |
| `metrics_clipped_round146.csv` | **标准口径**：裁剪到 [0,125] 后的全单元指标 |
| `near_failure_metrics.csv` | 近失效段指标 |
| `per_engine_advantage.csv` | 逐发动机优势 |
| `per_seed_metrics.csv` | 逐种子指标（含跨种子 SD） |
| `perf_design.csv` | 功效设计 |
| `pooled_comparisons.json` | 124 台池化的成对比较 |
| `pooled_engine_rmse.csv` | 池化下逐发动机 RMSE |
| `pooled_engines.json` | 池化发动机清单 |
| `prediction_collapse.csv` | 预测塌缩现象 |
| `prediction_corr.csv` | 预测间相关（中位 0.982） |
| `reference_diagnostic.csv` | 参考诊断 |
| `reference_diagnostic.json` | 参考诊断汇总 |
| `reference_strata.json` | 参考分层 |
| `split_variance.csv` | 切分方差（跨验证集 SD 中位 1.83） |
| `stage_bias.csv` | 按退化阶段的偏差 |
| `stage_calibration.csv` | 按退化阶段的校准 |
| `three_split_recheck.csv` | 三切分复核 |
| `training_audit.csv` | 训练审计 |
| `variance_decomposition.json` | 方差分解（方法 2.6%、验证集 31.5%、交互+种子 65.9%） |