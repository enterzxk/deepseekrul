# Phase 0：RHSE 前端 padding × latch Bug 修复

> 日期：2026-11-19。依据：`execution_plan_for_deepseek.md` §0。
> 结论：**§4.4 的 +0.556 clean 代价地板确实是 Bug，不是方法极限；修复后 q=0.5 时
> clean 代价 = 0.000、R3 recovery = 0.945、R5 recovery = 0.923；3 条硬不变量全部通过。**

## 1. 根因

`RHSE.forward` 返回 `w * mask`，左侧 padding（`mask == 0`）处 `w == 0`。
`rhse_rul_eval.py` 之前直接做：

```python
bad_cycle = w < qthr
bad_latch = np.maximum.accumulate(bad_cycle, axis=0)
```

padding 的 `w=0` 在任何正阈值下都成立；正向锁存把它传播到整个窗口，
导致所有 `cut < 70`（有左 padding）的窗口 14 个传感器全部被误判、全部被重构。
这解释了：

- q=0.10/0.25/0.40/0.50 的 clean 代价逐位相同（+0.556）；
- q=0.10/0.25 的 recovery 为负（重构了错误传感器）；
- q=0.50 突然跳到 0.838（锁存开始抓到故障传感器），但 clean 代价仍不变。

## 2. 修复

在阈值比较和锁存前强制只允许有效周期：

```python
bad_cycle = (w < qthr[None, :]) & mask[:, None]
bad_latch = np.maximum.accumulate(bad_cycle, axis=0)
```

`q01` 标定本来已经用 `w[mask]` 过滤；修复只影响检测/锁存/重构路径。
改动文件：`paper_b/modules/rhse_rul_eval.py`。

## 3. 3 条硬不变量（`logs/phase0_sanity.txt`）

```
0.1 clean zero-flag: max|R3c-R0|=0.000e+00
0.1 faulted zero-flag: max|R3-R1|=0.000e+00
0.2 padding: n_pad=15 n_pad_flags=0
0.3 parity: engine=57 cut=55 p0=123.584999 ref=123.584999 diff=0.000e+00
PHASE0 SANITY PASS
```

- **0.1 零标志恒等**：强制无 flag 时 R3c=R0、R3=R1，误差 0。
- **0.2 Padding 免疫**：宽松阈值下 padding 触发数严格为 0。
- **0.3 基线一致**：R0 与冻结 `preds.csv` 的 `pred_k1` 逐样本一致（diff=0）。

## 4. 修复后的 q-scale 扫描（FD001 s101，bias mag=3，5 cuts，sensors 0/5/10）

| q-scale | clean R3c 代价 | R3 recovery | R5 recovery |
|---:|---:|---:|---:|
| 0.10 | **+0.0000** | 0.000 | 0.000 |
| 0.50 | **+0.0000** | **0.945** | 0.923 |
| 1.00 | +1.0024 | 0.805 | −0.186 |

- q=0.10 时无检测 → clean 恒等、recovery 0；符合“q→0 平滑收敛到 0”的验收标准。
- **q=0.50 是当前工作点**：clean 代价 0.000、R3 恢复 94.5%、R5 恢复 92.3%。
- q=1.00 太松：clean 代价 +1.00，R5 崩到负。

## 5. 对 Gate 1 的含义

- **G1.1 clean 零伤害**：q=0.5 时 FD001 bias 3 的 clean 代价 0.000（≤0.15），通过。
- **G1.3 RUL 恢复**：R3 0.945 / R5 0.923（≥0.65），通过。
- **但 G1.3 的“R4 显著优于 R5 +0.10”在 bias 3 上不成立**：R3 只领先 R5 +0.022。
  → 必须在 **drift / stuck / onset=0.20** 上验证 RHSE 重构是否真的优于 last-value；
  如果仍只领先 <+0.10，说明“测量模型重构”相对朴素保持没有显著优势，Phase 2 的
  端到端主张需要重新定位。
- **下一步 Phase 1**：
  1. 冻结 `stress_suite.py`（stuck 原始物理空间注入、全 14 传感器、RNG 规范）；
  2. 实现 `RHSE-TSH`（`s_j` MAD 标定、学生化残差、四路 SPC、2-of-3 锁存、k_max=4）；
  3. 在 `DSPT-V0` 与 `B0` 上跑 1A 检测审计 + 1B `R0–R5`，过 Gate 1。
