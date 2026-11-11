# 消融模型定义（Vanilla Transformer 版）

> 2026-11-11 重建。**取代之前的 DSPT-Backbone 消融。**
> 代码：`src/candidates/dspt_vanilla.py`（模型）、`docs/paper_b_code/train_vanilla.py`（训练）

## 0. 为什么要重建

**旧版 base 不是 Vanilla Transformer**，而是「把 DSPT 的模块关掉」：
- 仍含 3 通道特征（raw / 偏差 / EMA）
- 仍含绝对 cycle 编码
- 仍含 Dual-Span Gate
- 仍含可学习的传感器 softmax 池化
- 仍含 33k 未使用的死参数

**这导致「模块贡献」无法归因到标准 Transformer。**

## 1. 新的 BASE（严格 Vanilla Transformer Encoder）

```
输入：(B, L, 14) 归一化传感器 + 二值掩码
  ↓
Linear(14 → 64)                       ← 标准输入投影
  ↓
固定正弦位置编码                        ← 无 cycle、无距离表、无可学习参数
  ↓
2 × TransformerEncoderLayer
    d_model=64, nhead=4, FFN=128
    GELU, norm_first, dropout=0.2
    src_key_padding_mask=~mask
  ↓
LayerNorm
  ↓
Masked Mean Pooling                   ← 对有效时间步求均值
  ↓
Linear(64→64) → GELU → Linear(64→1)
  ↓
RUL
```

**不含任何自研模块**：无卷积、无跨传感器注意力、无 HI、无三通道、无 cycle、无门控。

**参数量 72,257。**

## 2. 三个可开关模块

### MS · 多尺度时间卷积

```
h = embed(x) + PE
h = LayerNorm(h + proj(mean(conv_k(h) for k in 3,5,7)))
```
- 深度可分离卷积（groups=64），padding 保持长度
- **残差分支**，不改变主路径维度
- 参数量 +5,440

### SA · 跨传感器 Transformer

```
e = Linear(1→64)(x)                  → (B,L,14,64)
st = TransformerEncoderLayer(e)      ← 14 个传感器之间互相注意
h = LayerNorm(h + Linear(mean_s(st)))
```
- 2 头注意力，跨 14 个传感器
- **残差分支**，与时间表示相加
- 参数量 **+37,888**（占全模型 88% 的增量）

### HI · 健康指标辅助监督

```
hi = Linear(64→1)(pooled)            ← 只在训练时计算
loss += λ·MSE(hi, hi_target) + μ·mean(ReLU(hi_t − hi_{t−1}))
```
- **HI 标签不作为输入**
- λ=0.5, μ=0.1
- HI 标签**只在当前折的训练发动机上拟合**（KMeans 工况聚类 + z-score + PCA-1 + 分位）
- 参数量 +65

## 3. 八种配置

| 名称 | MS | SA | HI | 参数量 |
|---|:--:|:--:|:--:|---:|
| BASE | ✗ | ✗ | ✗ | 72,257 |
| MS | ✓ | ✗ | ✗ | 77,697 |
| SA | ✗ | ✓ | ✗ | 110,145 |
| HI | ✗ | ✗ | ✓ | 72,322 |
| MSSA | ✓ | ✓ | ✗ | 115,585 |
| MSHI | ✓ | ✗ | ✓ | 77,762 |
| SAHI | ✗ | ✓ | ✓ | 110,210 |
| FULL | ✓ | ✓ | ✓ | 115,650 |

**配置在代码中显式声明**（`--ms` / `--sa` / `--hi`），不是靠推断。

## 4. 统一协议

| 项 | 设置 |
|---|---|
| 数据划分 | 官方训练发动机，80% 训练 / 20% 验证，**按发动机 ID** |
| 划分种子 | 固定 `split_seed=7`，所有配置共享 |
| RUL cap | 125 |
| 窗口 | 同一子集内所有配置共享同一 L |
| 优化器 | AdamW, lr=1e-3, wd=1e-4 |
| 调度 | CosineAnnealing, T_max=epochs, eta_min=1e-5 |
| 早停 | patience=15，按验证 RMSE |
| 梯度裁剪 | 1.0 |
| 损失 | MSE + 0.5 × 非对称 MSE（晚预测 1.8×） |
| 随机种子 | 101 / 202 / 303 |

## 5. 历史结果的处理

| 名称 | 定义 | 状态 |
|---|---|---|
| **Historical Baseline** | 早期 DSPT（77,123 参数，单层 conv，无 sensor_attn/hi_head） | 保留，**不进新表** |
| **DSPT-Backbone** | 关闭 MS/SA/HI 的 DSPT（115,716 参数，含死参数） | 保留，**不进新表** |
| **DSPT-Full** | MS+SA+HI 的 DSPT（115,716） | 保留，**不进新表** |
| **Vanilla Transformer 系列** | 本文件定义的 8 种 | **新表唯一来源** |

**旧 32 组消融（`abl_val.csv` / `abl_test2.csv`）保留为历史证据，不混入新表。**

## 6. 与旧版的关键差异

| 维度 | 旧 DSPT-Backbone | 新 Vanilla BASE |
|---|---|---|
| 输入通道 | 3（raw/偏差/EMA） | **1（raw）** |
| 绝对 cycle 编码 | 有 | **无** |
| 位置编码 | 可学习距离表 | **固定正弦** |
| 池化 | Dual-Span Gate（双跨度 + 门控） | **Masked Mean** |
| 卷积 | 单层 k=3 | **无** |
| 传感器池化 | 可学习 softmax | **无（输入投影直接混合）** |
| 死参数 | 33k | **0** |
| 参数量 | 115,716 | **72,257** |

**新 base 比旧 base 少 43k 参数，因为去掉了三通道、cycle、双跨度门控和死参数。**
