# `fixed40` 30 格对账锚点 manifest

来源：`paper_b/experiments/open_source_matched_20261112/physical_runs.csv` @ `d7bd9c17`，
逐格 `config.json` / `validation_metrics.json` / `validation_predictions.csv` 从同一提交读出。
逐格数据：`FIXED40_ANCHORS.csv`（30 行）。提取命令见本文件末尾。

## 1. 30 格共有的协议参数（全 30/30 一致）

| 键 | 值 | 说明 |
|---|---|---|
| `epochs` | **40** | 上限，不是「跑满 40」 |
| `patience` | **12** | **含 `ffve`**——不是上游的 10 |
| `min_delta` | `1e-4` | 早停判据 |
| `checkpoint_selection` | `val_engine_mean_rmse` | 选轮口径 |
| `precision` | `float32` | 30/30 |
| `val_frac` / `split_seed` | `0.2` / `7` | 按发动机划分 |
| `rul_cap` | `125` | |
| `training_windows` | `all_cycle_left_padded` | |
| `validation_predictions.csv` | **30/30 在盘上** | 逐机对账的参照物 |

逐格不同的量（必须从 `config.json` 抄，不能从 `configuration()` 默认值猜）。
**按 `(模型, 子集)` 分组，不能按模型分组**——`lr` 有两条按子集变：

| 模型 | 子集 | `window` | `batch` | `lr` | `weight_decay` |
|---|---|---:|---:|---:|---:|
| aidgn | FD001 | 50 | 128 | 0.001 | 0.0 |
| aidgn | FD004 | 50 | 128 | 0.001 | 0.0 |
| two_phase | FD001 | 45 | 256 | 0.001 | 0.0 |
| two_phase | FD004 | 45 | 256 | **0.0001** | 0.0 |
| dual_mixer | FD001 | 30 | 204 | 0.001 | 0.0 |
| dual_mixer | FD004 | 30 | 204 | 0.001 | 0.0 |
| ffve | FD001 | 30 | 128 | **0.005** | **0.01** |
| ffve | FD004 | 30 | 128 | **0.001** | **0.01** |
| fcstgnn | FD001 | 30 | 100 | 0.001 | 0.0 |
| fcstgnn | FD004 | **50** | 100 | 0.001 | 0.0 |

`tf32=false` 30/30。

> **v1 勘误（2026-11-22，zxk 报、本管理员复核）**：本文第一版按模型汇总，写成了
> `two_phase lr=0.001 / ffve lr=0.001 / fcstgnn wd=0.01`，三处全错——`two_phase` FD004 的 `lr`
> 是 0.0001（差 10 倍）、`ffve` FD001 是 0.005（差 5 倍）、`wd=0.01` 属于 **ffve** 不是 fcstgnn。
> 错因是我把 30 格的聚合计数按顺序派给了模型，没有做分组明细——正是「裸聚合」。
> 逐格值见 `FIXED40_ANCHORS.csv` 的 `batch`/`lr`/`weight_decay` 三列。

## 2. 逐格 `epochs_run` / `best_epoch`

| 模型 | 子集 | seed | epochs_run | best_epoch | budget_truncated | val_engine_mean_rmse |
|---|---|---:|---:|---:|---|---:|
| aidgn | FD001 | 101 | 27 | 15 | false | 11.1004 |
| aidgn | FD001 | 202 | 16 | 4 | false | 10.6822 |
| aidgn | FD001 | 303 | 16 | 4 | false | 11.3565 |
| aidgn | FD004 | 101 | 32 | 20 | false | 13.1161 |
| aidgn | FD004 | 202 | 25 | 13 | false | 12.8844 |
| aidgn | FD004 | 303 | **40** | 35 | **true** | 12.5346 |
| two_phase | FD001 | 101 | 25 | 13 | false | 12.6753 |
| two_phase | FD001 | 202 | 25 | 13 | false | 11.3438 |
| two_phase | FD001 | 303 | 24 | 12 | false | 11.5971 |
| two_phase | FD004 | 101 | **40** | **40** | **true** | **23.8786** |
| two_phase | FD004 | 202 | **40** | **40** | **true** | **24.1526** |
| two_phase | FD004 | 303 | **40** | **40** | **true** | **23.7897** |
| dual_mixer | FD001 | 101 | **40** | 32 | **true** | 11.8591 |
| dual_mixer | FD001 | 202 | 21 | 9 | false | 12.6784 |
| dual_mixer | FD001 | 303 | 24 | 12 | false | 12.7642 |
| dual_mixer | FD004 | 101 | 18 | 6 | false | 14.5052 |
| dual_mixer | FD004 | 202 | 22 | 10 | false | 14.6131 |
| dual_mixer | FD004 | 303 | 22 | 10 | false | 14.7928 |
| ffve | FD001 | 101 | 27 | 15 | false | 10.8461 |
| ffve | FD001 | 202 | 25 | 13 | false | 11.3072 |
| ffve | FD001 | 303 | 28 | 16 | false | 11.5249 |
| ffve | FD004 | 101 | 26 | 14 | false | 12.9948 |
| ffve | FD004 | 202 | 27 | 15 | false | 13.2417 |
| ffve | FD004 | 303 | 26 | 14 | false | 12.8112 |
| fcstgnn | FD001 | 101 | 16 | 4 | false | 12.8544 |
| fcstgnn | FD001 | 202 | 23 | 11 | false | 12.5696 |
| fcstgnn | FD001 | 303 | 22 | 10 | false | 13.6032 |
| fcstgnn | FD004 | 101 | 19 | 7 | false | 14.0816 |
| fcstgnn | FD004 | 202 | 18 | 6 | false | 13.9811 |
| fcstgnn | FD004 | 303 | 18 | 6 | false | 13.7269 |

`validation_metrics.json` 与台账 `validation_engine_mean_rmse` 逐格比对：**0 处不一致**。

## 3. 三条会影响 A 段怎么跑的结论

1. **`fixed40` 不是「跑满 40 轮」。** 25/30 早停（`epochs_run` 16–32、`budget_truncated=false`），
   只有 5 格打满 40。A 段必须复现 `epochs=40 + patience=12 + min_delta=1e-4 + 选轮=val_engine_mean_rmse`，
   否则每格停在不同 epoch，`|Δ| ≤ 0.2` 比的是「停得早不早」而不是环境。
2. **`ffve` 的 patience 是 12，不是上游的 10。** 而且它 25–28 轮就停（best 13–16），
   在冻结协议里**已经收敛**，不是「40 轮才刚起步」。用上游 `configuration()` 的 100000/10 反而复现不了锚点。
3. **脆的锚点不是早停那 25 格，是打满 40 的 5 格。** 早停可复现（同一 patience/min_delta/选轮）；
   打满 40 的那几格是**在还在改善时被预算切断**的，环境一偏就动得多。
   最脆的是 `two_phase/FD004` 三格：`best_epoch = 40 = 最后一轮`，`val_engine_mean_rmse` 23.79–24.15，
   约为它自己 FD001 的两倍——**没有收敛**。对账时这三格应单独看，`|Δ|` 超 0.2 不自动等于「环境不一致」。

## 4. A 段复现清单（逐格从 `config.json` 抄）

```
epochs=40, patience=12, min_delta=1e-4, precision=float32,
val_frac=0.2, split_seed=7, rul_cap=125,
training_windows=all_cycle_left_padded,
checkpoint_selection=val_engine_mean_rmse,
window/batch/lr/weight_decay 见 §1 表
```

**第一道对账信号不是 RMSE，是 `epochs_run` / `best_epoch`**：A 段某格停在与锚点不同的 epoch，
说明数据或环境已经偏了，此时再比 RMSE 只是在这个偏差上再加一层噪声。

## 5. 提取命令

```bash
git show d7bd9c17:paper_b/experiments/open_source_matched_20261112/physical_runs.csv
git show d7bd9c17:paper_b/experiments/open_source_matched_20261112/physical_runs/fixed40/<model>/<subset>/s<seed>/config.json
git show d7bd9c17:.../validation_metrics.json
git cat-file -e d7bd9c17:.../validation_predictions.csv
```

---

## 6. 冻结曲线：`argmin` 与「水平」是两个统计量，不能互相替代

逐 epoch 曲线在 `FIXED40_CURVES.csv`（772 行，30 格 × 1..`epochs_run`，列
`model,subset,seed,epoch,val_engine_mean_rmse`），从 `history.json` 直接取。

**索引已核**：`history[best_epoch - 1]["val_engine_mean_rmse"]` 与
`validation_metrics.json::val_engine_mean_rmse` **30/30 逐位相等**（差 0）。`epoch` 字段 1-based。

### 6.1 两个统计量差多少（`均值 − argmin`，恒为正）

| 方法 | 自身全程均值 − argmin | 1..16 均值 − argmin |
|---|---|---|
| aidgn | +0.50 … +0.92 | +0.61 … +1.34 |
| dual_mixer | +1.19 … +2.07 | +1.02 … +2.83 |
| fcstgnn | +0.41 … +2.06 | +0.41 … +2.34 |
| ffve | +0.63 … +1.41 | +0.95 … +2.05 |
| **two_phase** | **+12.74 … +29.16** | **+19.49 … +46.27** |

**所以：拿 A 段的曲线均值去比冻结的 `validation_engine_mean_rmse`，每格都会凭空多出这段差。**
对 aidgn 是 +0.5~0.9（已经 2.5~4.6 倍于 0.2 的阈值），对 `two_phase` 是 +13~+29——30 格会全判
ABORT，而且看起来像环境炸了。**这不是边缘误差，是比错了统计量。**

### 6.2 窗口必须两边一样长

`two_phase/FD004` 的曲线仍在陡降（`best_epoch=40=末轮`），所以「均值」对窗口长度极敏感：

| 格 | 1..39 均值 | 1..40 均值 | 差一个 epoch 的位移 |
|---|---:|---:|---:|
| two_phase/FD004/s101 | 53.375 | 52.637 | **+0.737** |
| two_phase/FD004/s202 | 54.060 | 53.312 | **+0.748** |
| aidgn/FD004/s303 | 13.463 | 13.453 | +0.010 |

**+0.74 是 0.2 阈值的 3.7 倍。** 所以「水平」只能定义在**两边公共的 epoch 窗口**上
（A 段与锚点各自跑到第 n 轮，就都取 1..min(n, n_anchor)）；用「各自全程均值」互比，
`two_phase` 会因为轮数差 1 就判 ABORT。

### 6.3 谁用这张表

- **`paper_c/gate1.py`**：A 段对账的第 4 条（水平 `|Δ| ≤ 0.2`）与第 5 条（逐机秩）用公共窗口。
- **台账**：只存 `validation_engine_mean_rmse`（argmin），**不对它设门**；B 段闸门是
  `gate1.py` 的 `verdict` + 判据脚本哈希（`SCHEMA.md` §6 第 11 项）。
- 任何人要手算水平，从这里取锚点曲线，**不要**用 `FIXED40_ANCHORS.csv` 的 `val_engine` 当水平。
