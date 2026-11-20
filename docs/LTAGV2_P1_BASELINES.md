# LTAG v3 Phase 1：强对照与 Gate 1 判定（2026-11-20 修正版）

> **重要修正**：第一版 Phase 1 使用了 `cycle_frac = cut / true_lifetime`。真实寿命在 test
> 时不可观测，这是**泄漏**；第一版表（Linear-Stack 10.617 等）作废。
> 本版把 `cycle_frac` 替换为可观测的 `cut/250` 与 `min(cut,70)/70`，其余特征不变。
> 数据：`cata_gate/train`（训练发动机）与 `cata_gate/preds`（验证发动机）；
> 5-fold GroupKFold by (subset, engine)。**test 封存。**

## 1. 主表（observable features，validation engine-mean RMSE，越低越好）

| arm | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| K1 | 12.094 | 11.527 | 12.793 | 13.608 |
| Median | 11.429 | 11.704 | 12.984 | 14.399 |
| **LTAG** | 11.334 | 11.210 | 12.372 | 13.522 |
| Global-g | 11.503 | 11.234 | 12.622 | 13.661 |
| **Linear-Stack** | **10.150** | **10.604** | **12.033** | **13.080** |
| **K1-Calib** | **10.931** | **10.970** | **12.129** | **12.839** |
| LTAG-noMW | 11.299 | 11.186 | 12.526 | 13.583 |
| LTAG-noCtx | 11.449 | 11.216 | 12.456 | 13.597 |

**关键**：

- **K1-Calib 在 4/4 子集上优于 LTAG**（−0.404 / −0.240 / −0.243 / −0.683）。
- **Linear-Stack 在 4/4 子集上优于 LTAG**（−1.184 / −0.606 / −0.339 / −0.442）。
- LTAG 只胜过 Global-g（4/4）和 K1/Median（多数），但被最简单的后校准/线性 stack 全面覆盖。
- LTAG-noMW 在 FD001/FD002 甚至略优于 LTAG，FD003/FD004 略差；多窗口特征不是必需的。

## 2. Gate 1 逐条

### G1.1 LTAG 胜过 K1-Calib

Δ = K1-Calib − LTAG（正 = LTAG 更好）：

| 子集 | Δ | 95% CI |
|---|---:|---|
| FD001 | **−0.404** | [−1.530, +0.576] |
| FD002 | **−0.240** | [−0.825, +0.335] |
| FD003 | **−0.243** | [−0.965, +0.443] |
| FD004 | **−0.683** | [−1.154, −0.278] |

- 0/4 子集为正；四子集平均 **−0.393**。
- **G1.1 FAIL**（方向完全相反；FD004 CI 不含 0）。

### G1.2 LTAG 胜过 Global-g 和 Linear-Stack

- vs Global-g：LTAG 4/4 更好（+0.169 / +0.024 / +0.250 / +0.139）。
- vs Linear-Stack：LTAG 0/4 更好（−1.184 / −0.606 / −0.339 / −0.442）。
- 条件要求“同时胜过两者且 ≥3/4” → **G1.2 FAIL**。

### G1.3 时间聚合与集成可叠加

- `Ens3-K1` / `Ens3-LTAG` 未实现 → **G1.3 INCOMPLETE**。

### G1.4 多窗口信息是必需的

- LTAG-noMW 相对 LTAG：FD001 −0.035（noMW 更好）、FD002 −0.024（noMW 更好）、
  FD003 +0.154、FD004 +0.061。
- noMW 只在 2/4 子集更差 → **G1.4 FAIL**。

## 3. Gate 1 判定

| 条件 | 结果 |
|---|---|
| G1.1 LTAG > K1-Calib（≥3/4，平均 >0.10） | **FAIL（0/4，平均 −0.393）** |
| G1.2 LTAG > Global-g 且 > Linear-Stack | **FAIL（Linear-Stack 4/4 胜）** |
| G1.3 Ens3-LTAG > Ens3-K1 | INCOMPLETE |
| G1.4 LTAG-noMW 更差 ≥3/4 | **FAIL（2/4）** |

**Gate 1：FAIL。**

## 4. 结论

1. **LTAG 的学习门控机制不成立**：K1-Calib（单窗口残差校准）和 Linear-Stack
   在四个子集上全部优于 LTAG。
2. **多窗口一致性信息不是必需的**：LTAG-noMW 在 2/4 子集甚至更好。
3. 第一版“Linear-Stack 10.617”是 `cycle_frac` 泄漏的结果；修正后 Linear-Stack
   仍胜，但幅度变小（FD001 10.150）。
4. 按计划 §3 分支：**G1.1 失败 → 暂停 Phase 2；叙事改为“推理期后校准”，
   重新评估期刊。**
5. 原始 B0 LTAG test +0.398 **不因本次泄漏失效**：`learned_gate_global` 使用
   `FEATS_G`（不含 `cycle_frac`），`cata_test_features` 也把 `cycle_frac` 置 1；
   test 特征是可观测的。但 Phase 1 表明 LTAG 不是最佳聚合器——test 只验证了
   “LTAG 相对 K1 有增益”，没有验证“LTAG 优于后校准/线性 stack”。

## 5. 下一步（不改门槛）

- 以 **K1-Calib / Linear-Stack** 为新的候选主方法，做与 LTAG 同样的 test 冻结/一次性评估
  （Phase 5 只做一次；test 仍封存）。
- 核对原始 LTAG test 的特征列表（已核对：无可观测性泄漏），但不要把它当成“最优聚合器”的证据。
- 补 `Ens3` 仅为完成 G1.3，不改变 G1.1/G1.2/G1.4 的 FAIL。
- **Phase 2 不启动**；test 封存。

---

## 6. Phase 1b：G1.3 与 Linear-Stack 消融（post-hoc）

### 6.1 G1.3：Ens3-LTAG vs Ens3-K1

| arm | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| Ens3-K1 | 11.005 | 11.231 | 11.828 | 13.342 |
| Ens3-Median | 10.289 | 11.432 | 11.974 | 14.193 |
| **Ens3-LTAG** | **10.292** | **10.937** | **11.410** | **13.228** |

- Ens3-LTAG 在 4/4 子集优于 Ens3-K1 → **G1.3 PASS**。
- 但 Ens3-LTAG 在 FD001 与 Ens3-Median 持平（10.292 vs 10.289）；多 seed 集成本身已很强。

### 6.2 Linear-Stack 消融

| arm | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| LS-2（p_k1+p_med） | 10.536 | 10.919 | 11.500 | 13.247 |
| **LS-ctx**（+ context） | **9.256** | 10.483 | **11.026** | 12.903 |
| LS-all（+ 多窗口） | 9.334 | **10.431** | 11.086 | **12.797** |

- **context（cut/hist/regime/cap）是主要增益来源**；多窗口特征只在 FD002/FD004 略加分，
  在 FD001/FD003 反而略差。
- LS-ctx 在 4/4 子集优于 Ens3-LTAG（9.256 vs 10.292；10.483 vs 10.937；
  11.026 vs 11.410；12.903 vs 13.228）。

## 7. Gate 1 最终判定

| 条件 | 结果 |
|---|---|
| G1.1 LTAG > K1-Calib | **FAIL（0/4，平均 −0.393）** |
| G1.2 LTAG > Global-g 且 > Linear-Stack | **FAIL（LS 4/4 胜）** |
| G1.3 Ens3-LTAG > Ens3-K1 | **PASS（4/4）** |
| G1.4 LTAG-noMW 更差 ≥3/4 | **FAIL（2/4）** |

**Gate 1：FAIL。** G1.3 的局部 PASS 不能改变 G1.1/G1.2/G1.4 的失败。

## 8. 最终结论

1. **LTAG 的“学习何时聚合”机制不成立**：K1-Calib、LS-ctx、LS-all 在 4/4 子集优于 LTAG。
2. **Ens3-LTAG 优于 Ens3-K1**，说明门控在集成预测上仍有非线性收益；但它仍不如 LS-ctx。
3. **context（当前 cycle、历史长度、regime、cap）是主要增益来源**；多窗口一致性不是必需的。
4. 论文若继续，主方法应改为 **LS-ctx / K1-Calib 的推理期后校准**；创新性明显弱于原
   “学习门控”叙事，需要 L2/期刊决策。
5. **Phase 2 不启动**；test 封存；Phase 3–5 未启动。
