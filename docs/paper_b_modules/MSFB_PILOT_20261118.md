# MSFB：多尺度传感器特征库 + 特征类型注意力（首轮 pilot）

> 日期：2026-11-18。状态：**pilot（单 seed）**；3-seed 全网格与容量对照进行中。
> 代码：`paper_b/experimental_msfb/`；协议：Paper B 匹配协议（FP32、split_seed=7、fold=0、
> seeds 101/202/303、40 epoch、patience 12、cap125、engine-level RMSE、官方 test 每机末窗一次）。

## 1. 为什么要做

DSPT 本身已经有 **multi-scale depthwise CNN**（conv3/5/7）和 **cross-sensor attention**
（`sensor_attn`）；A8 96 格显示它们的独立效应只有 0.2–0.5，且在 seed 噪声内。  
所以“再加一个 CNN/attention”不是创新。可做的是 **传感器数据处理与特征**：

DSPT 的输入只有 `[raw, raw-baseline, causal EMA]`。MSFB 增加 8 个**因果**传感器特征通道：

| 特征 | 含义 |
|---|---|
| `dx` | 一阶差分（退化速度） |
| `ddx` | 二阶差分（退化加速度） |
| `slope5/10/20` | 5/10/20 周期斜率 |
| `cusum` | 偏差累积和（change-point/累积损伤） |
| `ema_fast−ema_slow` | 快慢 EMA 差（趋势/频段） |
| `cross-sensor contrast` | 每个传感器减去当周期跨传感器中位数（相对退化） |

再用 **feature-type attention** 在每个传感器/周期上学习“哪种特征族重要”，
融合后送入 DSPT 原有的 CNN + sensor attention + Transformer。
融合层零初始化：step 0 等于 DSPT 的 base 路径；参数 +6,113（+5.3%）。

## 2. 首轮结果（s101，同一协议）

| 子集 | MSFB validation engine-mean | MSFB official test RMSE | DSPT-V0-Cycle-FP32 test（匹配报告） | Δ |
|---|---:|---:|---:|---:|
| FD001 | **10.2904** | **10.0716** | 10.60 ± 0.44 | **−0.53** |
| FD004 | **11.7960** | **11.8701** | 12.38 ± 0.76 | **−0.51** |

- FD001/FD004 的 test 都低于 DSPT-V0 的 3-seed 均值；单 seed，不能当最终结论。
- NASA：FD001 167.4（DSPT-V0 191.3±21.4）；FD004 661.1（DSPT-V0 888.3±65.0）。
- 训练/验证/测试全部走同一 `train_one` 循环；test 用 `evaluate_run`，`prepare_test` 每机末窗一次。

## 3. 匹配比较的参照（来自 `MATCHED_COMPARISON_REPORT_20261112.md`）

DSPT-V0-Cycle-FP32：**10.60 / 11.03 / 10.68 / 12.38**（FD001–FD004，3 seed 均值）。
五个复现基线：AIDGN 12.25/13.03/12.40/13.06；Dual-Mixer 13.96/14.79/14.56/14.97；
FFVE 12.29/13.34/13.12/13.21；FC-STGNN 13.44/13.51/13.96/13.81；
Two-Phase（额外数据）12.82/14.30/13.18/14.75。

**当前 MSFB 只在 FD001/FD004 s101 上做了 test；FD002/FD003 与 3-seed 全网格正在跑。**
在拿到 3-seed 均值前，不能写“超过 DSPT-V0”或“超过五个基线”。

## 4. 必须补的对照（决定能不能写）

1. **容量对照**：同参数量的 DSPT + 同结构 feature-type attention，但额外特征通道置零
   （或重复 base 特征），没有新传感器信息。若 MSFB 不优于该对照，增益只是容量/优化，
   不是传感器特征。
2. **特征族消融**：去掉 `dx/ddx/slope/cusum/ema-diff/contrast` 各一组，看哪一族贡献。
3. **3 seed × 4 子集**：与 DSPT-V0 及五个基线同协议比较；报均值、sd、配对差。
4. **test 纪律**：MSFB 的 test 只做一次描述性比较；不得根据 test 调特征/结构。
5. **与 LTAG 的关系**：MSFB 是 backbone/特征模块；LTAG 是推理期聚合。两者可叠加，
   但先证明 MSFB 本身在 test 上稳定。

## 5. 当前判断

- MSFB 的 **pilot 信号很强**（FD001/FD004 test 各低 0.5 左右），但只有 s101。
- 创新点是 **传感器特征族 + 特征类型注意力**，不是“又加 CNN/SA”；
  DSPT 已有 CNN/SA，A8 已证明它们的边际效应小。
- 在 3-seed 全网格 + 容量对照完成前，**不能写成论文主结果**；只能说“首轮 pilot 支持继续”。
