# Paper C Phase 0：干净底座与 B_min 冻结

> 日期：2026-11-22。依据：`clean_restart_plan.md` §3 Phase 0。
> **test 未读取；所有 scaler/KMeans 只在训练发动机上拟合。**

## 1. 磁盘

- `/home` 当前 **92G 可用（98%）**，已满足计划要求的 **≥30G**。
- 未删除 `paper_b` 的任何结果/checkpoint；Phase 0 不需要通过删旧数据来腾空间。
- 后续只写 `paper_c/`、`results/paper_c/`、`configs/`，单次训练 checkpoint 很小。

## 2. 干净工程

| 文件 | 内容 |
|---|---|
| `paper_c/data.py` | 14 传感器、5-fold engine-level、每折 train-only scaler/KMeans、左 padding + valid_mask、6 cuts |
| `paper_c/models.py` | GRU-64（2 层）与 4-block causal TCN-64；无 cycle 输入 |
| `paper_c/metrics.py` | pooled/engine-mean RMSE、MAE、NASA、slope error、paired bootstrap、PICP/MPIW |
| `paper_c/train.py` | AdamW + CosineAnnealing + grad clip 1.0 + FP32；早停 |
| `paper_c/run_phase0.py` | FD001/FD004 × GRU/TCN × L30/50 × 5 folds |

- 数据：`/home/user/rulb/data/CMAPSS`；只读 `train_FD*.txt`。
- 输入：14 个传感器 `[2,3,4,7,8,9,11,12,13,14,15,17,20,21]`。
- 多工况：FD002/004 在每折训练集上 fit `KMeans(k=6)`，再按簇 fit scaler。
- cuts：`0.25, 0.40, 0.55, 0.70, 0.85, 0.95`。

## 3. 硬断言

### A0.1 零泄漏

- 扫描 `paper_c/*.py`（排除 audit/repro 自身）：**无** `test_FD*` / `RUL_FD*` 引用。
- 5 折 train/val 发动机集合交集为空。
- `Scaler.fit` 只接收当前折 `tr_df`；`make_val_windows` 只用该 scaler 的 `transform`。
- 结果：**PASS**（`results/paper_c/p0_audit.json`）。

### A0.2 可复现

- 同折、同 seed、GRU L50 fold0，5 epochs，`deterministic=True`，连续两次。
- 两次 state_dict 最大绝对差：**0.0**。
- 结果：**PASS**（`results/paper_c/p0_repro.json`）。

## 4. B_min 选型（5-fold engine-level，engine-mean RMSE）

| model | L | FD001 | FD004 | mean |
|---|---:|---:|---:|---:|
| **GRU** | **50** | **10.630 ± 0.912** | **12.142 ± 0.548** | **11.386** |
| GRU | 30 | 12.246 ± 0.830 | 13.388 ± 0.468 | 12.817 |
| TCN | 30 | 13.072 ± 1.048 | 14.293 ± 0.515 | 13.682 |
| TCN | 50 | 13.269 ± 1.336 | 14.432 ± 0.726 | 13.851 |

- **B_min = GRU, L=50, hidden=64, 2 layers, 51,073 params**。
- FD001 10.630 已优于计划目标 ≤11.2；FD004 12.142 优于 ≤13.5。
- TCN 明显弱于 GRU（13–14），可能是实现/调参问题；按计划选方差更低、训练更稳的 GRU，
  不把 TCN 的弱结果当作“基线阉割”——TCN 结构与 GRU 同等对待，均跑满 5 折。
- **B_min-Cap = GRU, hidden=64, 3 layers, 76,033 params**，用于 Phase 1 容量对照。

## 5. 冻结

`configs/backbone_freeze.json` 记录：

- `backbone=gru`, `L=50`, `use_cycle=false`；
- `backbone_config={hidden:64, layers:2, n_params:51073}`；
- `cap={model:gru, hidden:64, layers:3, n_params:76033}`；
- sensors、cuts、5 folds、split_seed=2026；
- 完整 selection 表与 selected 指标。

## 6. 下一步（Phase 1）

1. 实现 M1（CADF）、M2（CMST）、M3（CD-RCL）。
2. 在 FD001 & FD004 上跑单模块快筛（5-fold，每折 1 seed）：
   - `B_min`、`B_min-Cap`、`B_min+M1`、`M1_static`、`B_min+M2`、`M2_gate_only`、
     `M2_pool_only`、`B_min+M3`、`M3_tobit_only`、`M3_rate_only`、`M3_vanilla_nll`。
3. 按预注册 Gate 1 判定；未过门模块最多一次 v1.1 修订，仍不过则砍掉。
4. 至少 2 个模块过 Gate 1 才进入 Phase 2；否则停止并汇报。

**Phase 0 状态：PASS。**
