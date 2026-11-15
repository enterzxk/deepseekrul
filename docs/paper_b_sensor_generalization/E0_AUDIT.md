# E0 独立复算（军师）

对象：`split_seed=17` 的 Strong DSPT 12 格（4 子集 × 3 seed），run 根
`/home/user/pb_sensor/experiments/paper_b_sensor_generalization/runs/`。
方法：从每格 `val_pred.csv` 重算，不读 `metrics.json` 的聚合字段；脚本 `scripts/audit_e0.py`。

---

## 0. 结论

**通过。** 12/12 格齐全，两个口径的最大差都远小于 1e-4；划分按发动机分组、截点为
`round(f·n)` 的五个值，均已核。**但 `metrics.json` 的四个字段里只有一个是真的发动机级**，
主指标必须用 `val_engine_mean_roots_rmse`，见 §2。

| 检查 | 结果 |
|---|---|
| 12 格齐全、无缺失 | **12/12** |
| 每台验证机截点数 | **全部 5 个** |
| 验证机数 | FD001 20 / FD002 52 / FD003 20 / FD004 50 |
| `mean_u RMSE_u` vs `val_engine_mean_roots_rmse` | 最大差 **4.09e-7** |
| `sqrt(mean_u MSE_u)` vs `val_engine_equal_rmse` | 最大差 **4.23e-7** |
| pooled vs `rmse` / `val_pooled_rmse` / `val_engine_rmse` | 最大差 **4.23e-7** |
| `sqrt(mean_u MSE_u)` vs pooled | 最大差 **3.55e-15**（平衡设计下恒等） |

误差来自 float32 预测 + float64 重算的舍入，不是口径错误。

## 1. 12 格结果（主口径 = `mean_u RMSE_u`）

| 子集 | s101 | s202 | s303 | 均值 | seed sd |
|---|---:|---:|---:|---:|---:|
| FD001 | 8.5740 | 7.8177 | 8.7442 | **8.3786** | 0.4932 |
| FD002 | 10.0713 | 9.8386 | 10.0539 | **9.9879** | 0.1296 |
| FD003 | 8.6900 | 8.6236 | 9.0217 | **8.7784** | 0.2133 |
| FD004 | 11.0771 | 11.3679 | 11.9006 | **11.4485** | 0.4176 |

四子集等权均值 **9.6484**。与 zxk 报的表逐位一致。

## 2. 四个字段的实现口径（这是本轮最容易踩的坑）

| 字段 | 实际是什么 | 能否作主指标 |
|---|---|---|
| `val_engine_mean_roots_rmse` | `mean_u sqrt(mean_c (p−y)²)` | **是**（与前几轮一致、更保守） |
| `val_engine_equal_rmse` | `sqrt(mean_u mean_c (p−y)²)`，**数值等于 pooled** | 否，名字误导 |
| `rmse` | pooled | 否（辅助） |
| `val_pooled_rmse` | pooled | 否（辅助） |
| `val_engine_rmse` | pooled（本轮新发现的别名） | 否 |

在「每台机恰好 5 个截点」的平衡设计下，`sqrt(mean_u MSE_u)` 与 pooled **恒等**
（实测最大差 3.55e-15），所以 `val_engine_equal_rmse` 不是独立口径，只是 pooled 的副本。
FD004/s303 上 `val_engine_equal_rmse` = 13.9896，而真正的发动机级 `mean_roots` = 11.9006，
差 **2.09 周期**——用错字段会把结论整体抬高。

## 3. 划分与截点

- `data.py::split_units`：`rng = default_rng(split_seed + 20261020)`，对**排序后的发动机 ID**
  做 `permutation`，取前 20% 作验证机；`_prepare_split` 用 `raw.unit.isin(train/val_units)` 过滤。
  **按发动机分组，不按窗口随机分。**
- `data.py::make_val_windows`：`c = int(min(n, max(1, round(f*n))))`，f ∈ `(0.4,0.55,0.7,0.85,0.95)`；
  `train_one` 显式传 `cuts`，`cuts=None` 会直接报错。
- 12 格的 `config.json` 全部 `split_seed=17`、`fp32=True`（逐格核过，不是只看文档）。

## 4. 复现

```bash
python3 scripts/audit_e0.py /home/user/pb_sensor/experiments/paper_b_sensor_generalization/runs
```

脚本只读 `val_pred.csv` 与 `metrics.json` 的字段值，不 import 训练/分析代码。
