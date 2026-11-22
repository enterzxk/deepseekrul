# Paper C Phase 2：N-CMAPSS 转向与 Gate 2-N

> 日期：2026-11-22。数据：`/var/tmp/ncmapss_offload_20261120/extracted/` 的
> DS02-006、DS05、DS06、DS07（各 6 台发动机）。**test 未读。**
> 预注册 Gate 2-N 未修改。

## 1. 数据与协议

- 每个文件 6 台发动机，63–100 个 flight cycles；按 `(unit, cycle)` 聚合传感器均值。
- 输入：14 个 `X_s` 传感器 + 4 个连续工况 `W=[alt, Mach, TRA, T2]`，共 18 维；无 `T` 健康参数。
- 5-fold unit-level CV（6 台 → 每折 1–2 台验证）；train-only z-score；L=50；causal left padding。
- cuts：0.25/0.40/0.55/0.70/0.85/0.95；指标 engine-mean RMSE。
- 说明：每折验证发动机只有 1–2 台，方差很大；这是 N-CMAPSS 数据规模限制，不是选择性报告。

## 2. B_min 选型（GRU vs TCN × L30/50，5-fold）

| subset | GRU L30 | GRU L50 | TCN L30 | TCN L50 |
|---|---:|---:|---:|---:|
| DS02 | 22.851 | 22.808 | 22.623 | **22.587** |
| DS05 | 22.802 | 22.820 | **22.702** | 22.657 |
| DS06 | 21.977 | 21.972 | 21.820 | **21.710** |
| DS07 | 22.872 | 22.841 | 22.531 | **22.364** |

- **B_min = TCN L50**（51,201 params），4/4 子集最好或接近最好。
- 与 C-MAPSS 不同，N-CMAPSS 上 TCN 略优于 GRU；数据更小、工况更连续。

## 3. M1-CADF（修正 + per-condition）

| subset | B_min | M1_win | M1_eng | M1_static |
|---|---:|---:|---:|---:|
| DS02 | 22.587 | 22.817 | 22.834 | 22.814 |
| DS05 | 22.657 | 23.349 | 23.088 | 23.376 |
| DS06 | 21.710 | 22.438 | 22.192 | 22.503 |
| DS07 | 22.364 | 22.760 | 22.787 | 22.797 |

- 全部 M1 臂在 4/4 子集都**差于** B_min；最好的 M1_eng 也只有 DS06 22.192（仍差 0.482）。
- 结论：per-condition 基线在 N-CMAPSS 上也没有带来增益；M1 砍掉。

## 4. M2-Lite（per-timestep W FiLM + last-step）

| subset | B_min | M2-Lite |
|---|---:|---:|
| DS02 | 22.587 | 23.356 |
| DS05 | 22.657 | 23.422 |
| DS06 | 21.710 | 22.577 |
| DS07 | 22.364 | 23.315 |

- 4/4 子集都差于 B_min；FiLM 条件化没有带来增益。
- 结论：M2-Lite 砍掉。

## 5. Gate 2-N 判定

| 条件 | 实测 | 结果 |
|---|---|---|
| 点预测在 ≥3/4 子集比 B_min 好 ≥5% | 0/4 | **FAIL** |
| 同时胜 B_min-Cap | 未测（已低于 B_min） | **FAIL** |
| engine-level win-rate ≥65% | 未达 | **FAIL** |
| EOL 0.95 不退化 | M1/M2 均退化 | **FAIL** |

- **0/4 子集通过；Gate 2-N FAIL。**
- 按计划：允许最多 1 次 N-CMAPSS v1.1 修订；但 M1/M2 在 4/4 子集都差于 B_min，
  没有“接近过门”的臂可救；不再消耗一次修订。
- **Terminal Stop（Path 3B）**：停止训练，生成 `FINAL_AUDIT_BUNDLE.md`；
  test 保持封存。

## 6. 解释

- N-CMAPSS 每折只有 1–2 台验证发动机，5-fold 方差大；B_min 的 TCN 已经能利用连续 W，
  M1 的“个体基线”与 W 条件化信息重叠；M2-Lite 的 FiLM 在小样本上过拟合。
- 这不是“模块没想法”，而是：在极简强基线 + 小样本 5-fold 下，三个模块都没有可证伪的
  净收益。继续加模块会重蹈 Paper B 的覆辙。

## 7. 产物

- `paper_c/ncmapss_data.py`、`ncmapss_train.py`、`ncmapss_m1.py`、`ncmapss_m2lite.py`
- `paper_c/run_ncmapss_phase0.py`、`run_ncmapss_m1.py`、`run_ncmapss_m2lite.py`
- `results/paper_c/p2_nc_selection.json`、`p2_nc_m1.json`、`p2_nc_m2lite.json`
