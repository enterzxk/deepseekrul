# 第一阶段：协议审计与修复报告

> 2026-11-10。commit `de5db18`。
> **结论：修复前所有 Paper B 结果作废。** 36 组验证结果、官方测试集结果、循环数依赖分析全部受影响。

## 审计方法

逐行阅读 `scripts/run_dspt.py` 与 `src/candidates/dspt.py`，对每个声称的问题给出代码证据。

## 七个问题与修复

### Bug 1 · `--no-cycle` 只在训练生效【致命】

**证据**：
```python
def evaluate(net, data, device, bs=512):
    out = net(..., cycle=torch.from_numpy(c[i:i+bs]).to(device))  # 永远传 cycle
```

训练时 `cycle=None`，但**早停用的验证集仍然传 cycle**。
后果：所有「无 cycle」实验的 best_val 都是「有 cycle」的验证分数，**循环数依赖分析完全错误**。

**修复**：`evaluate(..., no_cycle=False, mode='dual')`，按 flag 决定是否传 cycle。

### Bug 2 · `--mode long` 训练目标与选模目标不一致【致命】

**证据**：
- 训练：`out["prediction"] = out["y_B"]`
- 早停：`evaluate()` 用 `out['prediction']` = 门控混合 `h_f = g·h_B + (1−g)·h_A`
- 最终评估：`ev(a.mode)` 用 `o["y_B"]`

**早停选的是另一个模型的输出。**

**修复**：`evaluate()` 按 `mode` 选 `y_A`/`y_B`/`prediction`，与训练一致。

### Bug 3 · HI 标签信息泄漏【严重】

**证据**：`mkhi2.py` 在**全部发动机**上拟合 KMeans、z-score 统计量、PCA 方向与分位。

后果：验证折的发动机参与了 HI 标签的构造。

**修复**：新增 `fit_hi(train_units, tr_frame, tr_scaled, seed)`，
KMeans / z-score / PCA / 方向 / 分位**全部只用训练折发动机**。
日志确认：`HI fitted on 80 TRAIN engines only`。

### Bug 4 · Transformer 未屏蔽左填充【严重】

**证据**：
```python
tokens = tokens + self.temporal_pe[:, dist]
for layer in self.layers:
    tokens = layer(tokens)          # 没有 src_key_padding_mask
```

填充位与真实位互相注意。L=160 时约一半位置是填充。

**修复**：
```python
pad = ~mask.bool().unsqueeze(1).expand(B, N, L).reshape(B * N, L)
tokens = layer(tokens, src_key_padding_mask=pad)
```

### Bug 5 · RUL 终点标签偏移【中等】

**证据**：`rul_curve(n) = n − arange(n)`，在 `t = n−1` 时给出 **1**，应为 **0**。

**修复**：`r = (n−1) − arange(n)`。

### Bug 6 · 仓库脚本与服务器不同步【中等】

**证据**：`docs/paper_b_code/run_dspt.py` 是 2026-11-10 早上的版本，
之后所有补丁只打在 lab3090 的 `scripts/run_dspt.py`。

**修复**：从服务器同步回仓库，并验证关键字符串存在。

### Bug 7 · `--full-data` 用训练数据自身做早停【中等】

**证据**：
```python
if a.full_data:
    val_units = sorted(int(x) for x in units)    # val = 全部
    train_units = sorted(int(x) for x in units)  # train = 全部
```

早停指标即训练损失。

**修复**：`--full-data` 时不做选择，取最后一个 epoch。

## 修复效果

| 子集 | 修复前 val RMSE | 修复后 OOF RMSE | 变化 |
|---|---:|---:|---:|
| FD001 | 10.97 | **10.12 ± 0.96** | **−7.7%** |
| FD003 | 11.36 | **9.90 ± 0.75** | **−12.9%** |

**padding mask 与 RUL 端点修复带来实质提升。**

## 作废的结论

| 曾报告 | 作废原因 |
|---|---|
| 循环数依赖 77.6% → 13.2% | Bug 1：验证时仍在用 cycle |
| FD004 依赖度 36.1% → 11.0% | 同上 |
| 36 组验证结果 | Bug 1/2/4 |
| 官方测试集 10.97/10.90/11.36/12.73 | Bug 1/2/4 |
| 窗口扫描（L=70 最优等） | Bug 4 影响长窗口，结论需重跑 |

## 未修复 / 已知限制

1. **多尺度卷积仍在填充位上计算**（kernel 3/5/7）。影响小于 Transformer，但未修复。
2. **EMA 递归已按 mask 处理**，无问题。
3. **HI 单调损失作用在整条序列上**，包括填充位；因为填充位 HI 目标为 0，
   惩罚项 `ReLU(hi_t − hi_{t−1})` 在 0→0 时为 0，影响可忽略。
4. **官方测试集已被观察多次**，后续设计选择需声明此限制。

## 证据路径

- 修复后代码：`docs/paper_b_code/run_dspt.py`（246 行）、`src/candidates/dspt.py`
- OOF 结果：`docs/paper_b_results/oof_summary.csv`
- 冒烟日志：`logs/SMOKE_FIX.log`（lab3090）
