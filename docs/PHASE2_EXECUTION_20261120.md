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

## 7. M1/M2 已实现并排队

- `paper_b/modules/model_rhse_dspt.py`：可微 H/log_s、3 步加权 LS、soft gate、
  `f_hi=[theta_last, slope10, slope30, energy, entropy]` 零初始化头。
- `paper_b/modules/phase2_train.py`：M1 clean / M2 +静态增强；RUL + α_meas·测量残差 +
  α_mono·θ 单调损失；5 seeds × 4 subsets。
- Smoke：M1 FD001 s101 1 epoch val 61.9（非证据），11.9 s/epoch。
- 队列：`scripts/queue_phase2_m1m2.sh`，40 runs，GPU1，运行中（M1 8/20，M2 未开始）。
- 偏差：当前 M1 只实现“输入净化 + θ 残差头”，尚未实现注意力偏置（M_no_attn 对照后续补）。

## 8. M_front 80/80 完成（冻结 M0 + 静态 RHSE-TSH）

| subset | fault | R3 mean (min–max) | R5 mean |
|---|---|---|---|
| FD001 | bias | 0.706 (0.619–0.802) | 0.049 |
| FD001 | drift | 0.566 (0.474–0.690) | 0.226 |
| FD001 | stuck | 0.625 (0.530–0.686) | −0.269 |
| FD001 | noise | −2.674 | −6.373 |
| FD002 | bias | 0.890 (0.816–0.935) | −0.037 |
| FD002 | drift | 0.679 (0.601–0.767) | 0.067 |
| FD002 | stuck | 0.420 (0.254–0.662) | −0.490 |
| FD002 | noise | −2.186 | −32.306 |
| FD003 | bias | 0.575 (0.476–0.660) | 0.100 |
| FD003 | drift | 0.469 (0.381–0.578) | 0.113 |
| FD003 | stuck | 0.576 (0.356–0.964) | −0.141 |
| FD003 | noise | 0.865 (0.397–1.090) | −5.742 |
| FD004 | bias | 0.836 (0.791–0.900) | −0.043 |
| FD004 | drift | 0.318 (0.198–0.406) | −0.012 |
| FD004 | stuck | 0.675 (0.557–0.775) | −0.230 |
| FD004 | noise | 0.578 (0.181–0.912) | −10.598 |

- 静态前端在 V0 上只在部分 cell 过 0.65（FD001/FD002/FD004 bias、FD002 drift、FD004 stuck）；
  drift/stuck/noise 大面积失败。
- **M_front 打不过 M0+Aug 的可能性很高**；M1/M2 是主路线。

## 9. M1 部分 clean 结果（5 seeds / 3 seeds）

| arm | subset | n | mean | vs M0 |
|---|---|---:|---:|---:|
| M0 | FD001 | 5 | 9.957 | — |
| M1 | FD001 | 5 | **9.802** | **−0.155** |
| M0 | FD002 | 5 | 10.863 | — |
| M1 | FD002 | 3 | **10.604** | **−0.259** |

- M1 在 clean 上**优于 M0**（FD001 −0.155、FD002 −0.259），这是好信号；
  但还需要故障恢复评估和 M2，才能判 Gate 2。
- M1 FD003/FD004 与 M2 仍在跑。

## 10. M1/M2 fault smoke（FD001 s101，bias mag3）

| arm | faulted RMSE | recovery |
|---|---:|---:|
| M0（baseline） | 21.969 | — |
| M0Aug | 21.092 | 0.084 |
| M1 | 19.621 | **0.226** |
| M2 | 31.007 | **−0.870** |

- M1 只恢复 0.226，远低于 0.65；M2 在故障输入上崩到 31.0（clean 9.885，faulted 31.0）。
- 这是**负信号**：端到端 M1/M2 的 soft gate 没有学到稳定的可靠性；M2 的增强反而放大了故障敏感性。
- 全矩阵 80 runs（4 子集 × 5 seeds × 4 faults）已在 GPU0 排队；等全量结果再判 Gate 2。
- 需要检查：M1/M2 的 `w_soft` 在故障窗口是否真的下降；如果 `w≈1`，净化没生效。

## 11. 当前不做

- 不打开 test；
- 不因 Gate 1 失败停工；
- 不修改已冻结的 Phase 1c 结果。
