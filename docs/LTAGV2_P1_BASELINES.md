# LTAG v3 Phase 1：强对照与 Gate 1 判定

> 日期：2026-11-20。依据：`ltag_v3_execution_plan.md` §3；数据：`cata_gate/train`（训练发动机）
> 与 `cata_gate/preds`（验证发动机）；5-fold GroupKFold by (subset, engine)。
> **test 封存。**

## 1. 主表（validation engine-mean RMSE，越低越好）

| arm | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| K1 | 12.094 | 11.527 | 12.793 | 13.608 |
| Median | 11.429 | 11.704 | 12.984 | 14.399 |
| **LTAG** | 11.468 | 11.106 | 12.377 | 13.535 |
| Global-g | 11.503 | 11.234 | 12.622 | 13.661 |
| **Linear-Stack** | **10.617** | **10.194** | **11.369** | **12.672** |
| K1-Calib | 11.869 | 11.308 | 12.402 | 13.264 |
| LTAG-noMW | 11.453 | 11.215 | 12.549 | 13.655 |
| LTAG-noCtx | 11.449 | 11.216 | 12.456 | 13.597 |

**核心异常：Linear-Stack 在四个子集上全部大幅优于 LTAG**（−0.851 / −0.912 / −1.008 / −0.863）。

## 2. Gate 1 逐条

### G1.1 LTAG 胜过 K1-Calib

Δ = K1-Calib − LTAG（正 = LTAG 更好）：

| 子集 | Δ | 95% CI | 方向 |
|---|---:|---|---|
| FD001 | +0.401 | [−0.832, +0.018] | LTAG 更好，CI 跨 0 |
| FD002 | +0.202 | [−0.569, +0.158] | LTAG 更好，CI 跨 0 |
| FD003 | +0.025 | [−0.408, +0.338] | 几乎持平 |
| FD004 | −0.271 | [+0.015, +0.519] | **LTAG 更差** |

- 3/4 子集方向为正；**四子集平均 Δ = +0.089 < 0.10**。
- **G1.1 FAIL**（平均门槛未过；FD004 反向）。

### G1.2 LTAG 胜过 Global-g 和 Linear-Stack

- vs Global-g：LTAG 在 4/4 子集更好（+0.035 / +0.128 / +0.245 / +0.126）。
- vs Linear-Stack：LTAG 在 4/4 子集**更差**（−0.851 / −0.912 / −1.008 / −0.863）。
- 条件要求“同时胜过两者且 ≥3/4 子集” → **G1.2 FAIL**。

### G1.3 时间聚合与集成可叠加

- `Ens3-K1` / `Ens3-LTAG` 尚未实现；本报告不伪造结果 → **G1.3 INCOMPLETE**。

### G1.4 多窗口信息是必需的

- LTAG-noMW 相对 LTAG：FD001 −0.015（noMW 略好）、FD002 +0.109、FD003 +0.172、FD004 +0.120。
- noMW 在 3/4 子集更差，但 FD001 反向；差异很小（≤0.17）。
- **G1.4 弱 PASS**，但不足以支撑“多窗口一致性是机制核心”。

## 3. Gate 1 判定

| 条件 | 结果 |
|---|---|
| G1.1 LTAG > K1-Calib（≥3/4，平均 >0.10） | **FAIL（平均 +0.089；FD004 反向）** |
| G1.2 LTAG > Global-g 且 > Linear-Stack | **FAIL（Linear-Stack 4/4 胜）** |
| G1.3 Ens3-LTAG > Ens3-K1 | INCOMPLETE（未实现） |
| G1.4 LTAG-noMW 更差 ≥3/4 | 弱 PASS |

**Gate 1：FAIL。**

## 4. 结论

1. **LTAG 的“学习何时聚合”机制主张不成立**：一个简单的 ridge linear stack 在四个子集上
   全部大幅优于 LTAG；K1-Calib 的平均差距也只有 +0.089（<0.10）。
2. **多窗口特征不是必需的**：LTAG-noMW 与 LTAG 差距 ≤0.17，FD001 甚至 noMW 略好。
3. **Global-g 也被 LTAG 击败**，说明 LTAG 有一点非线性收益；但被 Linear-Stack 完全覆盖。
4. 按计划 §3 的分支：**G1.1 失败 → 暂停，不进入 Phase 2；叙事改为“推理期后校准”，
   重新评估期刊。**
5. 这不是实现 bug 的迹象：Linear-Stack 使用同一训练/验证划分、无 test 泄漏；
   LTAG 的 5-fold 也是 engine-disjoint。需要在汇报中如实呈现。

## 5. 下一步（不改门槛）

- 先补 `Ens3-K1` / `Ens3-LTAG`，完成 G1.3（不改变 G1.1/G1.2 的 FAIL）。
- 分析 Linear-Stack 为何赢：是 `cycle_frac/cap/regime_dist` 的线性校准，还是 p_k1/p_med 的线性组合？
  做特征消融（仅 p_k1+p_med；加 context；加多窗口特征）。
- 若 Linear-Stack 仍全面胜出，论文主叙事必须从“学习门控”改为“推理期线性/后校准”，
  并重新评估是否还值得投 IEEE TIM。
- **Phase 2 不启动**；test 封存。
