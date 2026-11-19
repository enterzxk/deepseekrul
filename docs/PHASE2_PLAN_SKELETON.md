# Phase 2 计划骨架（仅计划，未启动；PENDING-L2）

> 日期：2026-11-19。前置：Phase 1c Gate 1 通过 + L2 批准。**test 封存。**
> 本文件只定义 Phase 2 的臂、协议和门，不产生新结果，不启动训练。

## 1. 目标

在强骨干 DSPT-V0 上检验：**端到端 RHSE-TSH 是否比静态前端和黑箱可靠性基线更能把
传感器故障下的 RUL 恢复转化为干净数据不劣化**。

## 2. 臂

| 臂 | 说明 | 训练 |
|---|---|---|
| M0 | DSPT-V0 clean | 复用冻结 checkpoint |
| M0+Aug | DSPT-V0 + 训练期传感器噪声/掩码增强 | 重训 |
| M1 | DSPT-V0 + RHSE-TSH 前端（H 冻结，无端到端） | 复用 |
| M2 | DSPT-V0 + RHSE-TSH，H/w 由 RUL 目标学习（端到端） | 重训 |
| M_smhi | SMHI-RF 黑箱可靠性对照 | 复用/轻量 |

## 3. 协议

- 子集：FD001、FD003（单工况，Phase 1c 已站住）；FD002/FD004 只作边界，不进入主表。
- 划分：冻结 `split_seed=7`、fold=0、val_frac=0.2、cuts 0.4/0.55/0.7/0.85/0.95。
- Seeds：5 个配对 seed（101/202/303/404/505）；同一 split、同一注入。
- 故障：bias/drift/stuck/noise × mag 3（confirmatory）；mag 2/5 只作敏感性。
- 指标：validation engine-mean RMSE、pooled RMSE、NASA score；clean cost；EOL FPR；
  paired bootstrap 95% CI。
- test：不打开；Phase 5 一次性评估。

## 4. Gate 2（预注册）

| 条件 | 门 |
|---|---|
| clean 不劣化 | M1/M2 相对 M0 的 RMSE 增加 ≤0.15 |
| 故障恢复 | M1/M2 在 bias/drift/stuck 上 ≥0.65（5-seed mean） |
| 优于强基线 | M1/M2 显著优于 M0+Aug 与 M_smhi（paired bootstrap 95% CI 不含 0） |
| 端到端增益 | M2 相对 M1 有正增益，或 M1 已接近 oracle R2 |
| 完整性 | 5 arms × 5 seeds × 2 subsets，0 failed |

## 5. 分支

- **Gate 2 通过**：进入 Phase 3（统一系统 + 共形区间 + 跨骨干）。
- **M1 接近 oracle、M2 无增益**：走路线 B（即插即用双冗余层），不强行端到端。
- **M1/M2 不如 M0+Aug**：RHSE 只写单工况条件性/附录，LTAG 主线。
- 任何路线切换走 L2，不在 DSH 内自决。

## 6. 预算

- 5 arms × 5 seeds × 2 subsets = 50 个训练/评估 run；按单 run 1–3 h 估，约 50–150 GPU·h。
- 先跑 1 个 smoke（FD001 s101，M0/M1），确认代码路径与指标口径，再排队。
- 不在 Gate 1 未判前启动。
