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
| FD004 | **12.30 ± 0.65**（11.846/13.052/12.006） | 12.38 ± 0.76 | **−0.08** | 中性 |

**结论**：MSFB 目前是**子集相关的候选模块**，不是通用模块。FD001 明确正（−0.56），
FD004 中性（−0.08），FD002 轻微变差（+0.11），FD003 明确变差（+0.81）。在容量对照和
逐族消融之前，不能写“MSFB 有效”。如果容量对照追平 FD001 的增益，它就不是模块，
只是容量；如果逐族消融能找到 FD003 的伤害源，才可能修成一个条件性模块。

### A2 zero-feature capacity control 结果（2026-11-18，s101）

| 子集 | A1 MSFB test | A2 zero-feature test | DSPT-V0 test | 读法 |
|---|---:|---:|---:|---|
| FD001 | **10.0482** | 10.5639 | 10.60 | A2≈V0；A1 比 A2 低 **0.516** → FD001 增益是**特征**，不是容量 |
| FD003 | **10.8375** | 10.6243 | 10.68 | A2≈V0；A1 比 A2 高 **0.213** → FD003 伤害也是**特征**，不是容量 |

**这是目前最关键的一刀**：容量对照没有解释 FD001 的增益，也没有解释 FD003 的伤害。
所以 MSFB **在 FD001 上是一个真正的特征模块**；但特征库的符号是**子集相关**的：
FD001 +0.52，FD003 −0.21。下一步不是继续加特征，而是 **A6 leave-one-family-out**
找 FD003 的伤害族、FD001 的贡献族；如果去掉某族后 FD001 仍正、FD003 回到中性，
MSFB 才能成为条件性模块。

### A6 leave-one-family-out 结果（s101，test RMSE）

| 变体 | FD001 | FD003 |
|---|---:|---:|
| MSFB full | **10.0482** | **10.8375** |
| A2 zero-feature control | 10.5639 | 10.6243 |
| drop `slope` | 10.0616 | 11.3568 |
| drop `ema_diff` | 10.0769 | — |
| drop `cusum` | **11.5374** | 10.8834 |
| drop `contrast` | — | 11.0599 |

**FD001：CUSUM 是贡献族。** 去掉 `cusum` 后从 10.0482 掉到 11.5374（比零特征对照
10.5639 还差）；去掉 `slope`/`ema_diff` 几乎不变。说明 FD001 的 +0.52 主要来自
**累积偏差/change-point 特征**，不是整个 8 族特征库。

**FD003：没有单一伤害族。** 去掉 `contrast`（11.0599）、`cusum`（10.8834）、
`slope`（11.3568）**都更差**，全量 MSFB（10.8375）反而最好；零特征对照（10.6243）
最好。说明 FD003 的 −0.21 不是某一族造成的，而是**额外特征路径整体在双故障模式上
过拟合/不适配**。

**下一步：CUSUM-only**。如果 CUSUM-only 在 FD001 保持 ~10.05、FD003 回到 ~10.6，
MSFB 可以收窄成“CUSUM 传感器特征模块”；如果 FD003 仍然 ~10.8，伤害来自额外路径/
融合本身，只能把 MSFB 限定为 FD001 条件性模块，或按 regime 门控。

### CUSUM-only 结果（s101，test RMSE）

| 变体 | FD001 | FD003 |
|---|---:|---:|
| MSFB full | **10.0482** | **10.8375** |
| A2 zero-feature control | 10.5639 | 10.6243 |
| **CUSUM-only** | **10.0878** | **11.1647** |

- **FD001：CUSUM-only 几乎复现全量 MSFB**（10.088 vs 10.048；control 10.564，DSPT-V0 10.60）。
  说明 FD001 的 +0.51 基本来自**单个可解释的 CUSUM 特征**；其余 7 族不是必需的。
- **FD003：CUSUM-only 仍然伤害**（11.165 vs control 10.624；full 10.838）。
  去掉其他族后反而更差，说明 FD003 的伤害**不是某一族**，而是“真实特征进入额外路径”
  这件事本身在双故障模式上过拟合；其他族只是部分补偿。
- 结论：MSFB 是**FD001 条件性模块**，不是通用模块。要让它在 FD003 安全，需要
  **regime/fault-mode 门控**（在 FD003 关闭额外路径）或只写 FD001 专用；不能靠
  去掉某一族来修。

### CUSUM 变体（s101，test RMSE）

| 变体 | FD001 | FD003 | 读法 |
|---|---:|---:|---|
| raw CUSUM-only | **10.0878** | 11.1647 | FD001 最好，但 FD003 伤害 |
| MAD-normalized | 10.7809 | 10.8573 | 两个都不好，不是 fix |
| **detrend（去发动机早期趋势）** | **10.3472** | **10.6803** | **FD003 回到 DSPT 水平；FD001 仍有 +0.25** |
| zero-feature control | 10.5639 | 10.6243 | 容量对照 |
| DSPT-V0 | 10.60 | 10.68 | 基线 |

**detrend CUSUM 是目前唯一在 FD001/FD003 都非负的变体**：
- FD001 10.347（比 DSPT-V0 低 0.25，比 raw CUSUM 低 0.26 的增益少）；
- FD003 10.680（与 DSPT-V0 持平，比 raw CUSUM 的 11.165 好 0.48）。

机制解释：raw CUSUM 把“发动机自身正常退化趋势”也算进累积量，双故障模式下这个趋势
不同，额外路径会过拟合；detrend 先减掉发动机早期线性趋势，只保留**偏离自身趋势的
累积偏差**，因此在 FD003 上不再伤害。代价是 FD001 的增益从 +0.51 降到 +0.25。

**下一步**：detrend CUSUM-only 的 3-seed × 4 子集全网格；如果 FD001 +0.25 保持、
FD002/FD003/FD004 非负，它就是一个**安全的条件性/通用传感器特征模块**；如果
FD001 的 +0.25 在 3 seed 下消失，则 raw 与 detrend 都只能写 FD001 条件性。

### detrend CUSUM 3-seed × 4 子集（官方 test）

| 子集 | detrend 3-seed | DSPT-V0 | Δ（detrend−DSPT） | 判定 |
|---|---:|---:|---:|---|
| FD001 | 10.601 ± 0.493（10.347/10.277/11.179） | 10.60 ± 0.44 | **0.00** | 中性；s101 的 +0.25 是 seed 噪声 |
| FD002 | **10.676 ± 0.043**（10.662/10.642/10.724） | 11.03 ± 0.05 | **−0.35** | **稳定正** |
| FD003 | 11.151 ± 0.536（10.680/11.034/11.738） | 10.68 ± 0.13 | **+0.47** | 伤害 |
| FD004 | **12.218 ± 0.303**（11.960/12.141/12.554） | 12.38 ± 0.76 | **−0.16** | 小幅正（在 sd 内） |

**3-seed 结论**：detrend 不是“FD001 安全版”，而是 **FD002 稳定正（+0.35，sd 0.04）**、
FD001 中性、FD004 小幅正、**FD003 仍伤害（+0.47）**。raw 与 detrend 的受益子集不同：
raw → FD001，detrend → FD002；两者都在 FD003 伤害。s101 的 detrend FD001 +0.25 没有
在 3 seed 下复现，不能写。

**下一步：combined raw+detrend CUSUM**（9 通道、7 族、feature-type attention）。
让注意力在每个传感器/周期上选择 raw 或 detrend，希望同时拿到 FD001 的 raw 增益和
FD002 的 detrend 增益，并在 FD003 抑制两者。3-seed × 4 子集网格运行中。

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
| A6 | **Leave-one-family-out**（8 臂） | 去掉 dx / ddx / slope / cusum / ema-diff / contrast 之一 | 找贡献/伤害源 | 至少一族去掉后 FD001 变差；FD003 的伤害应能定位 | 运行中（FD001: slope/ema_diff/cusum；FD003: contrast/cusum/slope） |
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

- 3090 `experiments/msfb_grid`：12/12 完成（FD001–FD004 × 3 seed），test 已用 CPU 一次性评估。
- 3090 `experiments/msfb_ctrl`：A2 zero-feature control，FD001/FD003 s101 已完成并 test：
  FD001 10.5639、FD003 10.6243；与 MSFB 10.0482/10.8375 对比见 §A2。
- 3090 `experiments/msfb_drop_*`：A6 leave-one-family-out 运行中
  （FD001: slope/ema_diff/cusum；FD003: contrast/cusum/slope）。
- 完成后：若某族在 FD001 是贡献族、在 FD003 是伤害族，试“去掉该族”的 MSFB 变体；
  若伤害族不唯一，考虑按 regime/子集门控或把 MSFB 限定为 FD001 条件性模块。
