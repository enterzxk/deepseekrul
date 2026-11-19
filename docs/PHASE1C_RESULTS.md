# Phase 1c 结果（进行中，2026-11-20）

> 队列：`scripts/queue_phase1c.sh`，60 RUL + 2 EOL。**test 封存。**
> 当前进度：23/62（RUL 21/60，EOL 2/2，0 failed）。本文件在队列跑完后更新为终版。

## 1. 协议

- FD001/FD003；bias/drift/stuck/noise；mag 2/3/5（stuck 仅 3）；
  seeds 101/202/303；sensors 0/5/10；gamma=1.30；onset=0.5。
- 3 seeds 是同一划分/注入的 checkpoint 重复，不是独立数据重复。
- 主指标 R3（TSH+Hθ）；R5（TSH+last-value）为弱 deployable 对照；
  R6 已撤回（R5 修复后 hybrid 把晚漂移路由到弱 last-value）。

## 2. FD001 已完成 cell（3-seed）

| fault | mag | n | R3 mean (min–max) | R5 mean | R3−R5 | clean cost |
|---|---:|---:|---|---:|---:|---:|
| bias | 2 | 3 | 0.784 (0.655–0.884) | 0.053 | +0.731 | −0.010 |
| bias | 3 | 3 | 0.826 (0.702–0.913) | 0.078 | +0.748 | −0.010 |
| bias | 5 | 3 | 0.782 (0.625–0.892) | 0.015 | +0.767 | −0.010 |
| drift | 2 | 3 | 0.700 (0.605–0.748) | 0.269 | +0.431 | −0.010 |
| drift | 3 | 3 | 0.797 (0.667–0.862) | 0.362 | +0.435 | −0.010 |
| drift | 5 | 3 | 0.814 (0.667–0.894) | 0.363 | +0.451 | −0.010 |
| stuck | 3 | 3 | **0.872** (0.79–0.93) | −0.037 | +0.908 | −0.010 |
| noise | 2 | 1 (s101) | n/a（no_harm） | — | — | −0.010 |
| noise | 5 | 1 (s101) | n/a（no_harm） | — | — | −0.010 |

- **noise mag2 s101 是 no_harm**：R0 12.647、R1 12.580，R1−R0=−0.068；故障不伤害 B0，
  recovery 无定义。此时应报告绝对 RMSE 与 ΔRMSE，而不是 recovery：
  R3 13.661、R5 16.094、**R7 filter 12.224**（比 clean 还好）。
- **noise mag5 s101 仍是 no_harm**：R1−R0=−0.127，R7 12.129、R3 13.443、R5 16.364。
  说明 FD001 的 B0 对加性高斯噪声在 mag≤5 内不敏感；noise 不应作为 recovery 主指标。
- 若 3 seeds 都是 no_harm，noise 不应计入“四类故障平均 recovery”；论文里单列
  “noise 在该幅度下不造成伤害”，并报告 R7 的绝对 RMSE。

- EOL：FD001 0.24% per-sensor-cycle / 5.0% any-window；FD003 0% / 0%。
- 所有 FD001 cell 的 R3 均值 ≥0.65；R3 全面领先 R5。
- **seed spread 是 checkpoint 驱动**：s303 的 clean R0 12.887（s101 12.647、
  s202 12.424），单 seed R3 最低 0.605–0.702；不得事后剔除，论文报 mean/min/max。

## 3. 待完成

- FD001 noise（进行中）：若 `R1−R0≤0.05`，标记 no_harm，不算失败。
- FD003 bias/drift/stuck/noise（全部 3 seeds）。
- 完整 Gate 1：`python -m paper_b.modules.phase1c_gate`。

## 4. 当前判定

- FD001 bias/drift/stuck：**部分 PASS**。
- 完整 Gate 1：**INCOMPLETE**（FD003/noise 未收齐）。
- Phase 2：不启动；路线取舍 PENDING-L2；test 封存。

**一句话：FD001 的 bias/drift/stuck 已全部过部分 Gate，stuck 3-seed R3 0.872 vs R5 −0.037；
等 FD001 noise 与 FD003 收尾后出终版 Gate 1。**
