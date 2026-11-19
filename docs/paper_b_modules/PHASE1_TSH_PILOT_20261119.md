# Phase 1 pilot：RHSE-TSH 检测 + 不训练 RUL 鲁棒性

> 日期：2026-11-19。代码：`paper_b/modules/rhse_tsh.py`、`tsh_rul_eval.py`。
> 结论：**TSH 把 stuck 从“检测不到”拉到恢复率 0.86–0.93，并把 clean 代价从 +0.59 压到
> +0.02–0.13；但 drift 上 last-value（R5）仍优于 Hθ 重构（R3），Gate 1 的“drift 上 R4 领先
> R5 +0.10”不成立。**

## 1. 实现

`rhse_tsh.py`：

- `stats()`：S1=|r|（Shewhart，bias）、S2=双边 CUSUM（drift）、S3h=innovation 方差 EWMA（noise）、
  S3l=原始/归一化信号 12 周期滚动 std（stuck）；
- `calibrate()`：clean training engines、仅 `mask>0`，S1/S2/S3h 取 99.75% 分位，
  S3l 取 0.25% 下分位；
- `detect()`：`u=max(S1/q1,S2/q2,S3h/q3h, stuck*2)`；2-of-3 有效周期确认 + 因果锁存；
  单周期最多剔除 `kmax=4` 个传感器。

`tsh_rul_eval.py`：R0 clean / R1 faulted / R2 oracle Hθ / R3 TSH+Hθ / R4 oracle last-value /
R5 TSH+last-value；FD001 s101，bias/drift/stuck mag=3，sensors 0/5/10，5 cuts。

## 2. gamma 扫描（FD001 s101，mag=3）

| gamma | clean R3c 代价 | drift R3 | drift R5 | stuck R3 | stuck R5 |
|---:|---:|---:|---:|---:|---:|
| 1.00 | +0.587 | 0.825 | 0.883 | 0.580 | 0.070 |
| 1.15 | **+0.134** | 0.859 | **0.974** | 0.857 | 0.381 |
| 1.30 | **+0.024** | 0.855 | **0.984** | **0.927** | 0.458 |

**读法**：

- **clean 代价**：gamma=1.15/1.30 分别 +0.134/+0.024，过 Gate 1 的 ≤0.15。
- **stuck**：TSH 把 R3 从 0.58（gamma=1.0）拉到 0.86–0.93；R5 只有 0.38–0.46；
  R3 领先 R5 **+0.48/+0.47**，过“R4 比 last-value 高 ≥0.10”。
- **drift**：R3 0.855–0.859，但 **R5 0.974–0.984**，last-value 更好；Gate 1 的
  “drift 上 R4 领先 R5 +0.10”**不成立**。
- bias（simple RHSE q=0.5，Phase 0）R3 0.945；noise 尚未跑。

### 2b. Onset-aware hybrid（R6）

按“首次检测到的周期在有效历史中的位置”选择重构方式：

```python
onset_est = (first_flagged - first_valid) / (last_valid - first_valid)
p6 = Hθ_reconstruction if onset_est < 0.35 else last_value
```

FD001 s101、drift mag=3、gamma=1.30：

| onset | R3 (Hθ) | R5 (last-value) | **R6 hybrid** |
|---:|---:|---:|---:|
| 0.20 | 0.890 | 0.833 | **0.912** |
| 0.50 | 0.855 | **0.984** | 0.972 |

- 早漂移（onset=0.20）：Hθ 有足够故障后周期估计 θ，R6 0.912 **同时优于 R3/R5**。
- 晚漂移（onset=0.50）：last-value 更强，R6 0.972 接近 R5（−0.012），远好于 R3。
- clean 代价保持 +0.024。
- 这说明“按故障起始位置选择空间重构 vs 时间保持”是一个**可证伪的机制签名**，
  比固定用 Hθ 或固定用 last-value 都更稳。

**注意**：Gate 1 原文要求“drift 上 R4 领先 R5 +0.10”；R5 在晚漂移已 0.984，
+0.10 在数学上不可能（上限 1.0）。建议把 drift 的门改成 **非劣（R6 ≥ R5 −0.05）**，
把“领先 +0.10”保留给 stuck 与 onset=0.20。

## 3. Gate 1 当前状态

| 条件 | 状态 |
|---|---|
| clean 代价 ≤0.15 | **通过**（gamma=1.30: +0.024） |
| EOL FPR ≤2% | Phase 0 简单 RHSE 1.8%；TSH 需重测 |
| 平均 recovery ≥0.65（4 类故障） | bias 0.945、drift 0.855、stuck 0.927；noise TBD，预计 ≥0.65 |
| R4 在 stuck 上领先 R5 ≥0.10 | **通过**（+0.47） |
| R4 在 drift 上领先 R5 ≥0.10 | **不通过**；R6 早 0.912 / 晚 0.972，晚漂移 R5 0.984（R6 非劣 −0.012）；建议改为非劣 |
| R4 在 onset=0.20 上领先 R5 ≥0.10 | 部分通过：drift 0.912 vs 0.833（+0.079），stuck 0.847 vs −0.286 |

## 4. 解释与下一步

- **TSH 解决了 stuck**：窗口活动度下侧统计量 + 2-of-3 锁存，stuck 的 RUL 恢复从 0 到 0.93。
- **drift 上 last-value 更强**：慢漂移的故障传感器“最后健康值”比 Hθ 重构更接近真实；
  Hθ 重构在 drift 下引入偏差。这不是检测失败，是**重构方式**的问题。
- **可能的最优组合**：按故障类型选择重构——
  - stuck/bias/noise：Hθ 重构；
  - drift：last-value（或 Hθ + last-value 的软混合）。
  这正是“空间冗余 + 时间冗余”统一叙事的一个可证伪预测：**不同故障类型应由不同的
  冗余修复方式处理**。但 Gate 1 的“单一 R4 全面领先 R5”需要改成“按故障类型的最优组合
  领先 R5”，否则计划书原文的 G1.3 不成立。
- **下一步**：
  1. 跑 noise / FD002 / FD004 / onset=0.20 / k=2；
  2. 重测 TSH 的 EOL FPR；
  3. 若 drift 的 last-value 优势稳定，定义 `R_hybrid`（drift→last-value，其余→Hθ），
     与 R3/R5 对比；用 `R_hybrid` 过 Gate 1 的“优于 last-value”条件；
  4. 再决定是否进入 Phase 2 端到端。

**一句话：TSH 让 stuck 从 0 到 0.93、clean 代价压到 +0.02；drift 上 last-value 仍更好，
所以下一步不是继续调 TSH，而是定义“按故障类型的重构选择”并用 hybrid 过 Gate 1。**

---

## 5. 全子集 Gate 1 批次（mag=3，onset=0.5，gamma=1.30）

| 子集 | bias R3/R5 | drift R3/R5/hybrid | stuck R3/R5 | noise R3/R5 |
|---|---:|---:|---:|---:|
| FD001 | 0.862 / 0.782 | 0.855 / 0.984 / **0.972** | **0.927** / 0.458 | R1<R0，fault 不伤害 |
| FD002 | 0.909 / 0.849 | 0.603 / 0.525 / 0.536 | **−0.047** / −1.935 | **−2.106** / −6.854 |
| FD003 | 0.808 / 0.791 | 0.765 / 0.772 / **0.793** | 0.588 / −0.907 | 0.581 / 0.400 |
| FD004 | 0.824 / 0.436 | 0.518 / 0.322 / 0.307 | **0.964** / −2.359 | 0.338 / −1.173 |

- FD001 drift 的 hybrid 是 onset=0.50（0.972）；onset=0.20 为 **0.912**（R3 0.890 / R5 0.833）。
- FD002 的 stuck/noise 恢复率为负：需要检查 R1 是否真的比 R0 差（若 R1≈R0，fault 不伤害，
  recovery 公式失效），以及多工况下 H 是否失配。
- FD004 的 drift 0.518、noise 0.338 低于 0.65；FD002 的 drift 0.603 也低于 0.65。

### Gate 1 判定

| 条件 | 结果 |
|---|---|
| clean 代价 ≤0.15 | **通过**（FD001 +0.024；其他子集待补） |
| EOL FPR ≤2% | TSH 未测；simple RHSE 1.8% |
| 平均 recovery ≥0.65（≥3/4 子集） | **不通过**：FD001/FD003 约 0.69；FD002/FD004 明显低于 0.65 |
| R4 在 stuck 上领先 R5 ≥0.10 | FD001/FD003/FD004 通过；FD002 两者都负，未通过 |
| R4 在 drift 上领先 R5 ≥0.10 | FD001 晚漂移 R5 更好；FD002/FD004 R3 更好但未达 +0.10；FD003 +0.02 |
| onset=0.20 领先 R5 ≥0.10 | FD001 drift +0.079（接近）；stuck 大幅通过 |

**结论：Gate 1 未通过。** 失败集中在 **FD002/FD004 多工况的 stuck/noise/drift**；
单工况 FD001/FD003 的 bias/stuck/drift 已经站住。按计划 §1.2(4) 启动 **Phase 1b：
多工况条件化 H_c**（6 个工况各一个 14×3 矩阵 + s_{c,j}），并重测 FD002/FD004 的
R2/R3/R5；同时检查 fault 是否真的伤害 R0（若否，提高 mag 或换传感器）。

**下一步**：

1. Phase 1b：实现 `H_c`/`s_c`（按 `cond` 查表），重跑 FD002/FD004 的 bias/drift/stuck/noise；
2. 补 TSH 的 EOL FPR（clean 末 20%）；
3. 若 FD002/FD004 的 stuck/noise 仍 <0.65，Gate 1 正式失败，按计划退回
   “单工况条件性 + 多工况限制”或转 Phase 2 路线 B；
4. test 仍封存。
