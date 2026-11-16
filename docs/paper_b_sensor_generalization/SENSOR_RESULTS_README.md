# 结果表说明（`SENSOR_BASELINE_RESULTS.csv` / `SENSOR_AUGMENTATION_RESULTS.csv`）

两张表都由 `scripts/export_results_tables.py` 从**逐机预测**重算生成，不是抄汇总 JSON。
只有 `best_ep` / `elapsed_s` / `code_fingerprint` 取自元数据。
认格一律走 `scripts/_cells.py`（按 `config.json` 的 `split_seed` 过滤 + 唯一性硬报错）。

## 1. 行数与构成

| 表 | 行数 | 构成 |
|---|---:|---|
| `SENSOR_BASELINE_RESULTS.csv` | 32 | E0 发现 12（split 17）+ 确认 20（split 29） |
| `SENSOR_AUGMENTATION_RESULTS.csv` | 108 | E1a/E1b/E2a/E2b 发现 48 + E1a/E1b/E2a 确认 60 |

增强表**逐格一行**；臂级统计（合并 Δ、CI、方向计数、seed sd、去 FD001）在同一臂的
每一行上重复，方便按 `arm` 过滤直接读。

## 2. 两个口径，不可混用（**最容易踩的一条**）

| 列 | 来源 | 口径 |
|---|---|---|
| `val_engine_mean_roots_rmse`、`clean_mean_roots`、`delta_rmse` | `val_pred.csv`（clean 评测路径） | 逐发动机 RMSE 再对发动机取平均 |
| `noise_beta{0.25,0.5,1.0}_mean_roots`、`..._degradation` | `val_noise_predictions.csv`（**噪声压力评测路径**，列 `y_pred_evaluated`） | 同一口径，但**走的是另一条评测路径** |

**噪声列的绝对值不能与 clean 列直接相减**。`audit_noise_pressure.py` 报的 β=0 是
11.057，而基线表 FD001 的 clean 是 9.6484，差约 1.4 个周期——那是**评测路径不同**，
不是矛盾。可比的是同一路径内的 Δ 与比值。

## 3. 配对差分的聚合（与判定时完全一致）

```
每个 (subset, seed)：d_u = RMSE_u(arm) − RMSE_u(E0)，在共同发动机上
每个 subset：把各 seed 的 d 向量逐元素取平均
合并：四个子集等权平均
CI：10000 次发动机级 bootstrap
```

**先把 seed 平均、再在发动机上重采样**——把三个 seed 当作同一台发动机的重复测量。
若改成把 seed 拼起来重采样（单位数 ×3），区间会**构造性地变窄**；本导出器早先犯过
这个错并已改正（E1a 发现阶段的宽度从 0.500 变回 0.548）。

## 4. bootstrap 种子：按阶段取判定当时那一个

| 阶段 | 种子 | 出处 |
|---|---|---|
| 发现（split 17） | `20261115` | `audit_e1e2.py:112` |
| 确认（split 29） | `20261116` | `audit_confirm.py:125` |

论文只报这一套。`arm_ci_seed` 列记录用的是哪个种子。

另外每臂用**另一个种子**再跑一遍，把端点差记在 `arm_mc_endpoint_drift_max`
（以及 `arm_ci_lo_alt_seed` / `arm_ci_hi_alt_seed`）。**这是量 Monte-Carlo 误差，
不是换种子挑结果**；`arm_ci_lo` / `arm_ci_hi` 永远是判定当时那一套。

实测 MC 端点漂移：最大 **0.0153**（E1b 确认），多数在 0.003–0.008。
所以「CI 端点对随机流有约 0.01 量级敏感性」这句可以写进复现性说明。

> **一处需要纠正的旧说法**：曾把「E2a 发现阶段上界 −0.039 被 0.024 的抖动吃掉 62%」
> 当成 MC 脆弱性证据。那个 0.024 实际是**两种聚合方式之差**（pooled-seed vs seed 平均），
> 不是 MC 误差。按正确聚合、实测 MC 漂移只有 **0.0041**，占其到 0 距离（0.0406）的
> **约 10%**。结论不变（E2a 确认阶段 NOT REPLICATED，本就不是候选），但理由要换。

## 5. 复现

```bash
python scripts/export_results_tables.py \
    --runs experiments/paper_b_sensor_generalization/runs \
    --out-dir docs/paper_b_sensor_generalization --boot 10000
```
