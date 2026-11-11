# P0 可复现性修复记录

> 2026-11-11。**在 P1 归因实验之前完成，所有后续实验使用同一份代码。**

## 修复清单

### P0-1 · 最终评估 `ev()` 无视 `--no-cycle`

**证据**（`run_dspt.py`）：
```python
def ev(mode):
    o = net(..., cycle=torch.from_numpy(VA[4][i:i+512]).to(dev))   # 无条件
```

我此前只修了 `evaluate()`（早停用），**漏了 `ev()`（最终预测用）**。

**修复**：`cycle=(None if getattr(a,'no_cycle',False) else ...)`

**影响**：所有 `--no-cycle` 实验的**最终预测**此前仍在用 cycle。已作废。

### P0-2 · MS 卷积的零填充污染

**证据**（`dspt.py` `encode`）：
```python
hh = h.permute(...).reshape(B*N, D, L)      # 左填充为 0
ms = sum(F.gelu(c(hh)) for c in self.multi_conv) / 3.0   # 直接卷积
```

卷积核跨越「填充区 ↔ 有效区」边界，把 0 混入有效区的特征。

**修复**：
1. 卷积前用**首个有效帧复制填充区**
2. 卷积后把填充位的输出置零

```python
first = mask.bool().float().argmax(dim=1)
fv = hh4[arange(B), :, :, first]
hh = torch.where(m4, hh4, fv.unsqueeze(-1)).reshape(B*N, D, L)
... conv ...
hh = hh * m4.expand(B, N, 1, L).reshape(B*N, 1, L)
```

**影响**：所有含 MS 的配置。**旧结果需要重跑。**

### P0-3 · HI 单调损失未屏蔽填充位

**证据**：
```python
loss = loss + a.hi_mono * torch.mean(torch.relu(hseq[:,1:] - hseq[:,:-1]))
```

对**全部**时间步计算，包括填充位。

**状态**：**未修复**（待办）。当前影响评估：
填充位的 HI 目标为 0，头部无监督；单调项在填充区把相邻输出压成非增，
会通过共享权重间接影响有效区。

**修复方案**：用 `mask` 的相邻差分加权：
```python
step = (mask[:,1:] * mask[:,:-1])          # (B,L-1) 只在两相邻位都有效时为 1
pen = torch.relu(hseq[:,1:] - hseq[:,:-1]) * step
loss = loss + a.hi_mono * pen.sum() / step.sum().clamp_min(1)
```

### P0-4 · `windows_for` 的 `legacy_rul` 参数不匹配

**证据**：函数体用了 `legacy=legacy_rul`，但签名里没有该参数 → `TypeError`。

**修复**：签名加 `legacy_rul=False`。

### P0-5 · HI 分支的 if/else 缺失

**证据**：`if legacy_hi: ...` 之后**没有 else**，导致先加载全局 HI 又被 `fit_hi` 覆盖，
日志同时打印两行。

**修复**：补 `else:`。

### P0-6 · EMA 与 checkpoint 策略

**审计结论：逻辑正确**，不是 bug。
```python
if best_state: net.load_state_dict(best_state)
_vb = evaluate(...)                 # best_state 的验证分
net.load_state_dict(ema_state)
_ve = evaluate(...)                 # EMA 的验证分
if _ve < _vb: keep EMA else: restore best_state
```
两者在验证集上比较后取优，并打印选择了哪个。

### P0-7 · ROOT 定位

`train_vanilla.py` 与 `eval_vanilla.py` 均使用 `Path(__file__).resolve().parents[1]`，
对 `scripts/` 下的脚本正确指向仓库根。

## 当前代码哈希

| 文件 | md5 |
|---|---|
| `scripts/run_dspt.py` | 待记录 |
| `src/candidates/dspt.py` | 待记录 |

## 未完成项

- **P0-3 的 HI 单调损失屏蔽**尚未实施（会改变所有 HI 配置的结果）
- 服务器与仓库的哈希需在每次实验前核对
