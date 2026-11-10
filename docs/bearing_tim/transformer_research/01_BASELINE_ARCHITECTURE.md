# 01 · 基线架构（以代码核对，不靠对话记忆）

> 依据：`scripts/run_bearing2.py`、`scripts/run_bearing3.py`、`scripts/bearing_data.py`。

## 1. 数据与输入张量

| 项 | 值 |
|---|---|
| 原始快照 | 每 0.1 s 一次，2560 行 × 6 列；第 4、5 列 = 水平/垂直加速度（25.6 kHz） |
| 合法标签 | `RUL_i = N − i`（N = 该轴承有效快照数） |
| 降采样 | `--ds 8`，**盒式平均 + 抽取**（不是理想低通；1.6 kHz 处仅 −9.7 dB） |
| 模型输入 | `(B, W, 2, T)`，`W=20`（窗口快照数），`T = 2560/8 = 320` |
| 滑窗步长 | 2 |
| 归一化 | 逐通道 mean/std，**只用训练轴承拟合** |

## 2. 编码器（`class Encoder`）

```
Conv1d(1, 16, k=9, s=2, p=4) → ReLU
Conv1d(16, 32, k=9, s=2, p=4) → ReLU
Conv1d(32, 64, k=9, s=2, p=4) → ReLU
AdaptiveAvgPool1d(1)
```
输入 `(B·W, 1, 320)` → 输出 64 维。

## 3. 四个臂（`run_bearing2.py::Net`）

| 臂 | 编码器 | 融合 | 后处理 |
|---|---|---|---|
| A | **共享**一个 Encoder 过两通道 | `Linear(128, 64)` | — |
| C | 双 Encoder | `Linear(128, 64)` | `out *= sigmoid(Linear(64,1))` |
| C-Fixed | 双 Encoder | `Linear(128, 64)` | `out *= 0.5` |
| C-MLP | 双 Encoder | `Sequential(Linear(128,128), ReLU, Linear(128,64))` | — |

之后统一 `GRU(64, 64, batch_first=True)`，取**最后一步**，`Linear(64, 1)`。

## 4. E0–E3 的骨干（`run_bearing3.py::Net`）

**E0 的骨干 = 双编码器 + `Linear(128,64)` 融合 + `GRU(64,64)`**，即**与 C-Fixed 同构**，不是 A。
E1–E3 在该骨干上把特征拼到 GRU 输入：`GRU(64 + n_feat, 64)`。
（此前对话若称"E0 用 A 骨干"是错的，以本文件为准。）

## 5. 训练与评价

| 项 | 值 |
|---|---|
| 损失 | MSE（RUL 除以训练集最大 RUL 归一化） |
| 优化器 | AdamW, lr 1e-3, wd 1e-4, 梯度裁剪 1.0, batch 64 |
| 轮数 | ≤40，早停 patience 8 |
| **三重互斥** | 留出轴承（评价）/ ES 轴承（早停）/ 训练轴承（其余 15 台） |
| ES 选择 | 按寿命中位数确定性选取 |
| 评价点 | 寿命 50% / 70% / 85% / 95%（只用于定位，输入只含该点前 W 个快照） |
| 设备级指标 | 逐轴承 RMSE/MAE，宏平均；配对统计以**轴承**为重采样单位 |

## 6. 数据完整性

`bearing_data.py::load_bearing_strict`：按内容检测分隔符（1_4 用分号）；逐快照校验
列数 ≥6、行数 ≥1024、数值有限、两通道非常数；**任何失败抛异常**，不零填充、不跳过。
17 台全部 OK（`docs/bearing_tim/data_inventory.csv`）。

## 7. 对 Transformer 分支的约束

- 必须复用同一 `load_bearing_strict` 与同一 LOBO 划分；
- 输入仍为 `(B, W, 2, T)`，历史长度 W ∈ {20, 64}（必要时 128）；
- 评价单位、评价点、早停协议与上表完全一致；
- 参数量与训练预算需显式列出并匹配。
