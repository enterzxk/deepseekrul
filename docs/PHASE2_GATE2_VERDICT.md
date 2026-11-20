# Phase 2 Gate 2 判定（2026-11-20）

> 依据：`experiments/phase2/`（M0/M0Aug/M1/M2 各 20 runs；故障矩阵 80 runs）、
> `phase2_summary.py`、`phase2_fault_summary.py`。**test 全程封存。**
> 结论：**Gate 2 FAIL。端到端 RHSE-DSPT 没有把 clean 收益转化为故障鲁棒性；
> M1/M2 在多数 cell 不如 M0+Aug，M2 在 FD004 甚至为负。**

## 1. Clean validation（engine-mean，5 seeds）

| arm | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| M0 | 9.957 | 10.863 | 10.486 | 11.783 |
| M0+Aug | 10.011 | 10.624 | 10.683 | 11.323 |
| M1 | **9.802** | 10.694 | 10.892 | 11.728 |
| M2 | 9.885 | **10.329** | 10.688 | 11.442 |

- M1 在 FD001 最好；M2 在 FD002 最好。
- M1/M2 不是全面优于 M0+Aug：FD003/FD004 的 M0+Aug 更好。

## 2. 故障恢复（M0 的 R1/R0 作分母，5 seeds mean）

| subset | fault | M0+Aug | M1 | M2 |
|---|---|---|---:|---:|
| FD001 | bias | −0.111 | 0.099 | −0.317 |
| FD001 | drift | −0.178 | 0.104 | −0.284 |
| FD001 | stuck | −0.270 | 0.197 | −0.397 |
| FD001 | noise | 0.474 | 0.215 | 0.497 |
| FD002 | bias | 0.188 | 0.006 | 0.158 |
| FD002 | drift | 0.094 | −0.099 | 0.023 |
| FD002 | stuck | 0.505 | 0.248 | 0.134 |
| FD003 | bias | 0.570 | 0.367 | 0.154 |
| FD003 | drift | 0.503 | 0.409 | 0.102 |
| FD004 | bias | 0.199 | −0.330 | −0.380 |
| FD004 | drift | 0.122 | −0.217 | −0.189 |
| FD004 | stuck | 0.478 | −0.232 | 0.531 |

- M1 最高只有 0.409（FD003 drift），远低于 0.65。
- M2 在 FD001 bias/drift/stuck、FD004 bias/drift 为负；增强没有带来鲁棒性。
- M0+Aug 在 FD003/FD004 多数 cell 反而更好。

## 3. Gate 2 判定

| 条件 | 结果 |
|---|---|
| clean 不劣化 | 部分：M1 FD001/FD002、M2 FD002 更好；FD003/FD004 不如 M0+Aug |
| 故障恢复 ≥0.65 | **FAIL**：M1/M2 最高 0.409 |
| 显著优于 M0+Aug | **FAIL**：多数 cell 不如 M0+Aug |
| 显著优于 M_smhi | 未跑；但 M1/M2 已低于 M0+Aug，先验不乐观 |
| 端到端增益 | **FAIL**：M2 在 FD004 为负 |

**Gate 2 FAIL。**

## 4. 机制解释（待补 w_soft 检查）

- M1 的 `w_soft = sigmoid(kappa(1-|r|/c))` 在故障窗口可能仍接近 1，净化没有生效；
- M2 的静态增强可能让模型对“训练期噪声”过拟合，反而对测试期故障更敏感；
- 需要检查 `w_soft`、`theta` 在故障窗口的分布；如果 `w≈1`，M1 实际退化为 M0 + θ 头。

## 5. 结论

1. **端到端 RHSE-DSPT（M1/M2）没有通过 Gate 2**；clean 收益不转化为故障鲁棒性。
2. **M0+Aug 是强基线**，在 FD003/FD004 多数 cell 优于 M1/M2。
3. **Phase 3–5 不应启动**；路线回到：
   - LTAG 主线（唯一 test 级验证，15/15，+0.398）；
   - RHSE-TSH 单工况条件性（B0 上 FD001/FD003）作为附录/边界；
   - V0 上的静态前端（M_front）也弱，不能写成通用模块。
4. 路线取舍（是否放弃 RHSE 主贡献、转 LTAG 主线）仍 **PENDING-L2**；
   Codex 2026-11-21 后裁决。

## 6. 下一步（可逆）

- 检查 M1/M2 的 `w_soft` / `theta` 在故障窗口的行为，确认“净化未生效”；
- 补 M_smhi / M_ols / M_no_attn / M_no_theta 消融（若仍要完整 Phase 2 矩阵）；
- 论文边界定稿：LTAG 主贡献 + RHSE 条件性/负结果；
- test 保持封存。
