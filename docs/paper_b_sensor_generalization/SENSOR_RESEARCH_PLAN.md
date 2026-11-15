# SENSOR_RESEARCH_PLAN（冻结件）

> study_id: `paper_b_sensor_generalization`
> 分支：`research/paper-b-sensor-generalization`，从 `paper-b-contribution-attribution` 的 HEAD `0e7893e8` 切出。
> 本文件在**任何本轮训练启动之前**冻结；之后只允许追加「是否满足」，不允许改阈值、口径或臂定义。

## 0. 一句话

在冻结的强 DSPT 上，用**训练期传感器增强**（整通道 mask / 残差尺度噪声）检验「传感器冗余与噪声」是否还有可复现的独立增益；增强类最好只能到 ENGINEERING-ONLY，GO 留给 E4/E5 的结构类候选。

## 1. 边界（不可越）

- 只用 lab3090 的两张 RTX 3090；A40 不启动新实验。
- 全部 FP32；**不得**用历史 BF16 结果充当 FP32 对照。
- Cycle 沿用冻结的 `pre_norm`；本轮**不改** `min(cycle/250, 1)` 截断逻辑。
- 官方 TEST 不参与任何设计、选轮、选臂或阈值；只在冻结后一次性评价、只作描述。
- 不把相关滑动窗口当独立设备；统计单位是发动机。
- 记录全部实验，不只保存获胜配置；失败格如实登记。

## 2. 代码身份

| 项 | 值 |
|---|---|
| 分支 | `research/paper-b-sensor-generalization` |
| 切出点 | `0e7893e8`（`paper-b-contribution-attribution` HEAD） |
| 训练入口 | `scripts/run_sensor_aug.py` |
| 增强实现 | `paper_b/sensor_aug/{augmentation,noise_stats,evaluation}.py` |
| 改动文件 | `paper_b/models/dspt.py`（`sensor_keep`：注意力 key-mask、pooling 重归一化、encode 清洗）、`paper_b/training/{data,train,metrics}.py`（`cond_seq`、增强钩子、engine-equal 指标） |
| 代码指纹 | `code_fingerprint()` = **8 个文件** sha256[:12]：7 个模型/训练文件 + `scripts/run_sensor_aug.py`（臂的 BASE 定义写在这里，不纳入指纹就会「改了协议但指纹不变」）。另存 7 文件 legacy 值供跨 study 对照。manifest/freeze 记两个值，每格 `metadata.json` 记 `git_head`。 |
| 本轮指纹 | `h8=740809bed8c9`，`h7=3f3a3e68594e`（旧 study 的 `h7` 是 `90f20d35953d`；差异来自 `train.py` 的 `fp32` 默认值与入口断言，不改模型数学） |

`data.py` 新增 `cond_seq`（每窗每时间步的工况簇）**只**供噪声增强按步取尺度；回归网络仍只吃窗口末端工况。

## 3. 数据与划分

- 数据：`data/CMAPSS`，FD001–FD004，14 传感器，`span=70`，`rul_cap=125`。
- 划分：`split_units(units, split_seed, fold=0, val_frac=0.2)`，**按发动机分组**。
- **发现阶段 `split_seed=17`**；**确认阶段 `split_seed=29` 预先保留，不得提前用于任何模型选择**。
- 模型种子：`101 / 202 / 303`。
- 验证截点：`round(f·n)`，`f ∈ {0.40, 0.55, 0.70, 0.85, 0.95}`，`n` 为该机自身寿命。这是协议常数，不读官方测试 RUL。
- 官方 TEST：冻结前不打开。

## 4. 模型（冻结的强 DSPT）

```
DSPT(num_sensors=14, d_model=64, num_layers=2, dropout=0.2, max_span=70,
     ms_mode='full', sa_mode='basic', hi_source='hf', hi_mono='correct',
     use_cycle=True, use_pad_mask=True, n_cond=6, mode='dual')
```

训练：AdamW，`lr=1e-3`，`wd=1e-4`，`batch=512`，`epochs≤40`，`patience=12`，按验证 engine-level RMSE 选轮，FP32。

## 5. 实验臂（E0–E2；E3 已删除）

| 臂 | 定义 | 参数 | 与旧 study 的映射 |
|---|---|---|---|
| **E0** | 无增强（基线） | — | G0 |
| **E1a** | 训练期整通道 mask，p=0.10 | 同一传感器整窗全时间步一起遮；raw/偏差/EMA 三通道一起；显式 availability mask；至少留 1 路 | G1 |
| **E1b** | 训练期整通道 mask，p=0.20 | 同上 | G2 |
| **E2a** | 训练期残差尺度噪声，α=0.25 | 同机、周期差 1、同工况、两端有限的高频残差，稳健尺度 σ；`s=min(ασ, 0.20·r)`；加在 x 上、特征重建之前 | G3 |
| **E2b** | 同上，α=0.50 | | G4 |

**E3（mask + noise 组合）删除**：两个单通道在旧 study 均未达预注册门槛，组合没有剩余假设。`run_sensor_aug.py` 在 `task_config` 里硬拒绝 `mask_prob>0 and noise_alpha>0`，防止被重新打开。

**重要映射更正**：旧报告里的「噪声通道 G2 +8.73%、FD001 +15.3%」**不是噪声注入臂**，而是 **mask p=0.20（G2）在噪声压力评测下**的鲁棒性收益。旧 study 的噪声注入臂是 G3/G4，其噪声通道收益为 **−1.47% / +2.62%**。本轮按代码与 `stage1_table.md` 为准，不按旧报告的口头标签。

## 6. 旧结论的引用方式（不重复已完成的工作）

| 旧结论（split_seed=7, FD001/FD004, BF16 训练 + FP32 评价） | 本轮处理 |
|---|---|
| mask 无 clean 增益（G1 −0.14%、G2 −0.39% 宏观） | 直接引用，不重跑 clean 结论 |
| mask 在噪声压力下有信号（G2 +8.73%，FD001 +15.3%，FD004 +2.2% CI 跨 0） | **只在 2 个子集上出现过**，四子集未验证 → E1a/E1b 必须在 split 17 上补 4 子集 |
| 噪声注入臂无增益（G3 −1.47%、G4 +2.62%） | 按本轮 spec 复核一次（E2a/E2b），不假设结论可迁移 |
| 缺失通道 +1.7%~+2.0%，远低于 10% | 引用，不作为候选 |

**为什么仍然跑 E1**：唯一还站着的信号来自 mask 在噪声压力下的鲁棒性，而它只测过 FD001/FD004。把它当作「已判死」会把唯一有信号的臂排除掉。

## 7. 判据（冻结）

主指标：**engine-equal RMSE** = `sqrt(mean_u MSE_u)`，其中 `MSE_u` 是发动机 u 全部验证截点的均方误差。另报 MAE、NASA mean、逐机误差、seed sd。

配对：同 `(subset, seed)` 的 `ΔRMSE = arm − E0`（负 = 更好）。95% CI 用**发动机级** bootstrap ≥10000 次（重采样发动机 ID，保留该机全部截点）。

### 7.1 三档

**GO**（仅结构类 E4/E5 可及）四条同时满足：
1. 四子集合并 `ΔRMSE ≤ −0.3 cycle`；
2. 配对 95% CI 不含 0；
3. ≥3/4 子集方向为负；
4. 在 `split_seed=29` 的确认集上复现（5 seed，同预算）。

**ENGINEERING-ONLY**：clean 通道不显著，但某个鲁棒性通道（缺失 k=1/2/3、噪声 β=0.25/0.5/1.0）稳定改善，且 clean 恶化不超过 1%。**E1/E2 即使有效也只能落这一档**——训练期增强不包装成架构创新。

**STOP**：没有任何臂在 discovery 上达到 GO 或 ENGINEERING-ONLY；或确认集不复现。

### 7.2 护栏

- NASA mean 比值 > 1.10 的臂不得进入候选（即使 RMSE 更好）。
- 单子集显著不算数；不得用「最好的 seed」代表一个臂。
- 不得因为某格难看而剔除；只有查实技术缺陷才作废，作废要重跑并写明原因。

## 8. run_id 与产物

```
run_id = {arm}-{subset}-s{seed}-{config_hash8}
experiments/paper_b_sensor_generalization/runs/{run_id}/
    config.json        # 完整配置 + code_fingerprint + git HEAD
    status.json        # RUNNING / COMPLETE / FAILED
    best.pt
    val_pred.csv       # 逐发动机、逐截点：unit, cut, y_true, y_pred
    metrics.json       # engine-equal RMSE/MAE/NASA + pooled + seed 信息
    train.log
```

- 断点续跑：`runs/` 已 COMPLETE 的格不再启动；claim 用 `O_EXCL`。
- 每格训练前写 `config.json`；训练后写 `metrics.json`；失败写 `FAILED` 与 traceback。

## 9. 执行顺序

1. **E0**：4 子集 × 3 seed = **12 格**（本阶段交付）。
2. **E1a/E1b + E2a/E2b**：4 × 3 × 4 = **48 格**（下一阶段，需 E0 先落地做配对基线）。
3. 汇总：engine-level 重算 → 配对 ΔRMSE + CI → 按 §7 判定。
4. 若出现候选：在 `split_seed=29` 上确认（5 seed），再一次性评价 TEST（描述性）。

## 10. 不做的事

- 不重跑 E3 组合。
- 不改 `min(cycle/250,1)`、不改 Cycle 注入位置、不改窗口长度。
- 不新增 Attention / 不改骨干（E4/E5 属于后续阶段，且必须先过文献员的新颖性判断）。
- 不用官方 TEST 做任何选择。
