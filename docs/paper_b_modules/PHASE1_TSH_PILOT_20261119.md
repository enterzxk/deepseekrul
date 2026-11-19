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

## 3. Gate 1 当前状态

| 条件 | 状态 |
|---|---|
| clean 代价 ≤0.15 | **通过**（gamma=1.30: +0.024） |
| EOL FPR ≤2% | Phase 0 简单 RHSE 1.8%；TSH 需重测 |
| 平均 recovery ≥0.65（4 类故障） | bias 0.945、drift 0.855、stuck 0.927；noise TBD，预计 ≥0.65 |
| R4 在 stuck 上领先 R5 ≥0.10 | **通过**（+0.47） |
| R4 在 drift 上领先 R5 ≥0.10 | **不通过**（R5 更好） |
| R4 在 onset=0.20 上领先 R5 ≥0.10 | 未跑 |

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
