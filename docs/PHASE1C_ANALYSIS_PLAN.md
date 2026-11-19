# Phase 1c 分析与分支计划（预注册）

> 日期：2026-11-19。队列：`scripts/queue_phase1c.sh`，60 RUL + 2 EOL。
> 本文件只定义“跑完后怎么判”，不引入新数字。test 封存。

## 1. 单元与聚合

- 20 个 cell：2 subsets × (bias/drift/noise × mag 2/3/5 + stuck × mag3) = 20。
- 每个 cell 3 seeds（101/202/303），seed 是 checkpoint 重复，不是独立数据重复。
- 报告：R3/R5/R6/R7 的 **mean / min / max**；clean cost 的 max；EOL per-cycle 与 any-window。
- 主指标：R3 recovery（TSH+Hθ）；R5 是 deployable last-value 对照；R4 是 oracle 上界。

## 2. Gate 1（预注册）

| 条件 | 门 |
|---|---|
| clean cost | 每 (subset, seed) ≤ 0.15 |
| EOL FPR | per-sensor-cycle ≤ 0.02（2/2） |
| R3 recovery | 每 cell n=3 且 mean ≥ 0.65 |
| R3 vs R5 | stuck/drift 上 R3−R5 ≥ 0.10（按 seed 配对） |
| 完整性 | 20/20 cells，0 failed |

判定由 `python -m paper_b.modules.phase1c_gate` 输出 PASS/FAIL/INCOMPLETE。

## 3. 分支 A：Gate 1 通过

若 20/20 cells 通过（noise 的 `n/a` 单独处理）：

1. 写 `docs/PHASE1C_RESULTS.md`：3-seed 表 + 每 cell 的 min/max + clean/EOL。
2. 进入 Phase 2 的**预实验**（不直接训练）：
   - 在 DSPT-V0 上定义 M0（clean）、M0+Aug（传感器噪声/掩码增强）、M1（RHSE-TSH 前端）、
     M2（RHSE-TSH + 端到端 w 偏置）、M_smhi（黑箱可靠性对照）；
   - 5 个配对 seeds；固定 split；Gate 2：M1/M2 显著优于 M0+Aug 与 M_smhi；
   - 先跑 1 个 smoke（FD001 s101）验证代码路径，再排队。
3. 路线取舍仍走 L2（Phase 2 是否启动），不因 Gate 1 通过而自动训练。

## 4. 分支 B：Gate 1 失败

若任一 cell <0.65 或 R3−R5 <0.10：

1. 不进入 Phase 2；写 `docs/PHASE1C_RESULTS.md` 的失败表与原因。
2. 按失败类型处理：
   - **FD003 stuck <0.65**：15 步 run-length 的延迟成本；报告延迟-恢复 trade-off，或缩短 W；
   - **noise `n/a`/低**：R1≈R0，fault 不伤害；改用绝对 ΔRMSE，不强行算 recovery；
   - **mag 2/5 不稳**：只把 mag=3 当 confirmatory，mag=2/5 当敏感性。
3. 论文边界：RHSE-TSH 写单工况条件性；LTAG 主线；多工况负结果写限制。
4. 路线取舍（是否放弃 RHSE 主贡献）标 PENDING-L2，Codex 可用后裁决。

## 5. 禁止事项

- 不用官方 test 选阈值/选模型；
- 不把 3 seeds 当独立数据重复；
- 不把 mag=2/3/5 混池重算 confirmatory；
- 不引用修复前 R5/R6 数字（oracle 泄漏）；
- 不在 Gate 1 未判前启动 Phase 2 训练。
