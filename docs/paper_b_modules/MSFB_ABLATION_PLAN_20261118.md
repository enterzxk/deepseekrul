# MSFB 算不算模块？消融怎么做

> 日期：2026-11-18。状态：3-seed 部分 test 已出（FD004 还差 2 seed），容量对照运行中。
> 协议：与 `MATCHED_COMPARISON_REPORT_20261112.md` 相同（FP32、split_seed=7、fold=0、
> seeds 101/202/303、40 epoch、patience 12、cap125、engine-level RMSE、官方 test 每机末窗一次）。

## 0. 先回答“算不算模块”

**结构上算**：MSFB 是一个可分离的传感器特征处理块——8 个因果特征族 + feature-type
attention + 零初始化融合；它相对 DSPT 是“只加一个变量”（额外特征路径），step 0 等于
DSPT base。它不是“又加一层 CNN/SA”（DSPT 已有 conv3/5/7 与 sensor_attn）。

**但按本项目的“模块”标准，目前还不够**：模块必须通过**同协议 3-seed 全子集 + 容量对照
+ 逐族消融**，证明增益不是容量、不是 seed、不是子集选择。当前 3-seed test：

| 子集 | MSFB 3-seed test（mean ± sd） | DSPT-V0-Cycle-FP32 | Δ（MSFB−DSPT） | 判定 |
|---|---:|---:|---:|---|
| FD001 | **10.04 ± 0.35**（10.048/9.688/10.389） | 10.60 ± 0.44 | **−0.56** | 正 |
| FD002 | **11.14 ± 0.36**（11.556/10.885/10.993） | 11.03 ± 0.05 | **+0.11** | 轻微变差 |
| FD003 | **11.49 ± 0.71**（10.838/11.379/12.240） | 10.68 ± 0.13 | **+0.81** | 明确变差 |
| FD004 | 11.85（仅 s101；s202/s303 运行中） | 12.38 ± 0.76 | −0.53（1 seed） | 待定 |

**结论**：MSFB 目前是**子集相关的候选模块**，不是通用模块。FD001/FD004 有正信号，
FD002/FD003 变差；在容量对照和逐族消融之前，不能写“MSFB 有效”。如果容量对照追平
FD001 的增益，它就不是模块，只是容量；如果逐族消融能找到 FD003 的伤害源，才可能修成
一个条件性模块。

## 1. 消融矩阵

所有臂使用**同一协议**、同一 data split、同一 seed 集；test 只做一次性描述性比较，
不用 test 选臂。

| # | 臂 | 改动 | 目的 | 预期/判定 | 状态 |
|---|---|---|---|---|---|
| A0 | DSPT-V0 | 原 DSPT（full/basic/hf/cycle） | 基线 | 10.60/11.03/10.68/12.38 | 已有（报告） |
| A1 | **MSFB** | +8 特征族 + feature-type attention | 主候选 | FD001/FD004 应更好；FD002/FD003 不应显著变差 | 3-seed 部分完成 |
| A2 | **Zero-feature control** | 同参数量，额外 8 通道置零 | 隔离容量 | 若 A2≈A1，A1 的增益是容量 → 不是模块 | FD001/FD003 s101 运行中 |
| A3 | **Base-repeat control** | 额外 8 通道重复 base 特征 | 隔离“新信息” | 若 A3≈A1，增益不是新传感器特征 | 待跑 |
| A4 | **Uniform attention** | feature-type attention 换成均匀平均 | 注意力是否必要 | 若 A4≈A1，attention 不是贡献 | 待跑 |
| A5 | **Concat+linear** | 8 族直接 concat + 单层 Linear | 融合方式对照 | 若 A5≥A1，attention 多余 | 待跑 |
| A6 | **Leave-one-family-out**（8 臂） | 去掉 dx / ddx / slope / cusum / ema-diff / contrast 之一 | 找贡献/伤害源 | 至少一族去掉后 FD001 变差；FD003 的伤害应能定位 | 待跑 |
| A7 | **Single-family-only**（6 臂） | 只用一族 | 找最小充分族 | 若单族≈全族，特征库叙事过宽 | 待跑 |
| A8 | **MSFB w/o MS** | `ms_mode=off` | 与已有 CNN 的交互 | 若 MSFB 增益消失，说明它依赖 CNN | 待跑 |
| A9 | **MSFB w/o SA** | `sa_mode=off` | 与 sensor attention 的交互 | 若 MSFB 增益消失，说明它依赖 SA | 待跑 |
| A10 | **Random-init fusion** | 去掉零初始化 | 初始化对照 | 若 A10≈A1，零初始化不是原因 | 待跑 |
| A11 | **Extra-only** | 去掉 base 路径，只用 8 族 | base 是否必要 | 应显著变差；否则 base 冗余 | 待跑 |
| A12 | **Sensor shuffle** | 训练/测试时打乱传感器通道 | 是否用传感器结构 | 若 A12≈A1，学的是容量/工况，不是传感器 | 待跑 |
| A13 | **Feature shuffle** | 在时间/传感器间打乱 8 族特征 | 是否用因果结构 | 若 A13≈A1，特征族无信息 | 待跑 |
| A14 | **Future-inclusive features**（诊断，不进主表） | 用未来帧算 slope/EMA | 因果性检查 | 应更好；若主模型接近它，说明主模型已在用未来信息（不应发生） | 待跑 |

## 2. 统计与判定

- 每个臂 3 seed × 4 子集；报 mean ± sd、逐 seed、配对差（MSFB−DSPT）。
- 主判定：**FD001 的 3-seed 配对差 95% CI 排除 0**，且 FD002/FD003 不显著变差；
  否则只能写“FD001 条件性”。
- 容量门：A2（zero-feature）与 A1 的差 < 0.10 → 判定增益来自容量，MSFB 不算模块。
- 特征门：至少一个特征族 leave-one-out 使 FD001 变差 ≥0.10，且该族在 A7 单族中有效；
  否则“传感器特征库”叙事不成立。
- 交互门：A8/A9 若与 A1 差 <0.10，说明 MSFB 的增益不依赖已有 CNN/SA（可写“独立特征模块”）；
  若差很大，说明它是 CNN/SA 的补充，不能单独归因。
- test 纪律：A0–A14 的 test 只做一次描述性比较；不根据 test 改特征/结构。

## 3. 现在最该先跑的三个

1. **A2 zero-feature control**（FD001 + FD003 s101）——直接回答“是不是容量”。
2. **A6 leave-one-family-out** 在 FD001 与 FD003 上——找 FD001 的贡献族和 FD003 的伤害族。
3. **A8/A9**（MSFB w/o MS / w/o SA）——回答“它和已有 CNN/SA 是什么关系”。

如果 A2 追平 A1，停止 MSFB 主线；如果 A6 找到伤害族并能去掉，MSFB 可修成条件性模块；
如果 A8/A9 显示 MSFB 依赖 CNN/SA，论文应写成“传感器特征 + CNN/SA 的联合模块”，不能
单独声称特征库的独立贡献。

## 4. 当前运行

- 3090 `experiments/msfb_grid`：FD001/FD002/FD003 各 3 seed + FD004 s101 已完成；
  FD004 s202/s303 运行中。
- 3090 `experiments/msfb_ctrl`：A2 zero-feature control，FD001/FD003 s101 运行中（cuda:1）。
- 完成后：CPU 跑 test（GPU0 已满），汇总 mean/sd/配对差，再决定 A6/A8/A9。
