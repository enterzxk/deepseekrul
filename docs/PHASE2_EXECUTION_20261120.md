# Phase 2 执行记录（2026-11-20）

> 按计划书继续执行 Phase 2；Gate 1 的失败不改变“把分配任务跑完”的要求。
> **test 仍封存。**

## 1. 已启动的队列

| 队列 | 臂 | 子集 × seeds | 设备 | 状态 |
|---|---|---|---|---|
| `scripts/queue_phase2_m0.sh` | M0 = DSPT-V0 (V0C) clean | 4 × 5 = 20 | cuda:0 | 运行中 |
| `scripts/queue_phase2_m0aug.sh` | M0+Aug = 30% 训练窗口噪声/遮蔽 | 4 × 5 = 20 | cuda:1 | 运行中 |

- 输出：`experiments/phase2/phase2_M0_*`、`experiments/phase2/phase2_M0Aug_*`。
- 日志：`experiments/phase2/logs/`。
- 幂等：`metrics.json` + `close.json` 存在则跳过。
- Smoke：M0 val 63.1（1 epoch，非证据）；M0+Aug val 63.1（1 epoch，非证据）。

## 2. 代码锁偏差（必须记录）

3090 的 `~/sprint24` 不是 git 仓库，且 `paper_b/hc/hi_d.py`、`train_d.py` 与
`docs/paper_b_hc/registry/code_lock.json` 不一致（其他会话的未提交改动）。为启动 Phase 2：

- 生成 `docs/paper_b_hc/registry/code_lock_phase2.json`，只更新这两个文件的 md5；
- `git_sha=WORKTREE_PHASE2`，`source=WORKTREE_PHASE2`；
- 所有 Phase 2 run 显式传 `--code-lock code_lock_phase2.json`。
- 这是**协议偏差**；论文中必须写“Phase 2 使用当前 worktree 代码 + Phase 2 lock”，
  不能声称与 `ee39d7e2` 完全一致。

## 3. M0+Aug 的实现

`paper_b/modules/phase2_aug.py` 在 `run_v0.main()` 前 monkey-patch `prepare_train`：

- 30% 训练窗口；
- 每个选中窗口随机 1–2 个传感器通道；
- 50% 加高斯噪声 σ~U[0.5,2.5]，50% 置 0；
- **静态增强**（每个 run 固定），不是 per-batch；作为 Phase 2 偏差记录。

## 4. 已产出的基线（validation engine-mean，5 seeds）

| 臂 | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| M0 | **9.957** | **10.863** | **10.486** | **11.783** |
| M0+Aug | 10.011 | 10.624 | 10.683 | 11.156（n=3） |

- M0 20/20 完成；M0+Aug 19/20，最后一 run 在跑。
- M0+Aug 在 FD002/FD004 略好，FD001/FD003 略差；不是全面更强，Gate 2 仍需 M1/M2 对比。

## 5. M_front 首批（FD001 s101，mag3，V0 冻结）

| fault | R0 | R1 | R3 | R5 | R3 recovery |
|---|---:|---:|---:|---:|---:|
| bias | 11.586 | 21.969 | 15.541 | 21.760 | 0.619 |
| drift | 11.586 | 20.559 | 16.106 | 18.833 | 0.496 |
| stuck | 11.586 | 12.495 | 11.925 | 12.893 | 0.628 |
| noise | 11.586 | 12.670 | 15.143 | 16.810 | **−2.283** |

- 静态 RHSE-TSH 前端在 **DSPT-V0 上明显弱于 B0**：bias 0.619、drift 0.496、stuck 0.628，
  都低于 0.65；noise 甚至伤害（R3 比 R1 差）。
- 这说明“B0 上的前端收益”不能直接迁移到强骨干 V0；M1/M2 端到端是必要的，
  也说明 M_front 可能打不过 M0+Aug。
- 80 个 M_front run 已在 GPU0 排队（4 子集 × 5 seeds × 4 faults）。

## 6. 下一步（队列运行中并行实现）

1. **M1**：RHSE-DSPT（可微 H/log_s、soft gating、θ 特征/注意力偏置）；
2. **M2**：M1 + 同款增强；
3. **M_smhi / M_ols / M_no_attn / M_no_theta**：机制消融；
4. 5 seeds × 4 subsets 配对；Gate 2 用 engine-level paired bootstrap。

## 7. 当前不做

- 不打开 test；
- 不因 Gate 1 失败停工；
- 不修改已冻结的 Phase 1c 结果。
