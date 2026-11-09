# 最近邻全文差异矩阵（IJPHM 稿 v2）

> D022 要求：逐项区分**研究对象 / 是否同中心对照 / 是否评估尺度收益 / 诊断可用信息 /
> 预注册规则 / 估计稳定性**；未读全文的必须标明。
> 核实等级：**A** = 读过全文或包内文档；**B** = 读过摘要/部分正文；**C** = 仅元数据（不得引用结论）。
> 更新：2026-11-09（本轮新取到 DriftGuard / Robinson / Domingues 三篇全文，均升为 A）。

---

## 1. 矩阵

| 工作 | 等级 | 研究对象 | 单位 | 需要目标标签 | 给边界 | 做修复 | 多 seed | 同中心对照 | 诊断规则检验 | 估计稳定性 |
|---|---|---|---|---|---|---|---|---|---|---|
| **本稿** | — | 保形 RUL 区间 | **发动机** + 格 | 否（诊断只用校准侧） | 经验边界（未达支持标准） | 否 | ✅ 4 seed × 5 折 | ✅ **同中心常数宽度** | ✅ 预注册 G-A…G-D | ✅ 逐格 bootstrap |
| Moussa 2026（DriftGuard） | **A** | label-free 覆盖审计（**边际**） | 目标域整体 | 否（无标签阶段） | ✅ **命题级**：检测界、minimax 不可区分、[0,1] 识别集 | 是（标签预算 + 重校准） | 不适用 | ✗ | 部分（domain discriminability / ESS / weight stabilization） | ✗ |
| Zhou et al. 2026（CPA） | **A** | 条件覆盖评估框架 | 实例（x） | **是**（有标签 D_eval 训练 reliability estimator） | 否（给收敛率 Theorem 1） | 否（含选择算法 CC-Select） | 未报多 seed | ✗ | ✗ | 有理论速率，无逐格 bootstrap |
| Robinson 2026（IJPHM） | **A** | CQR 应用 + 风险偏好 | **边际** | 是 | 否 | 方法本身 | 未报 | ✗ | ✗ | ✗ |
| Domingues & Ather 2026 | **A** | C-MAPSS 条件覆盖崩溃 | RUL 带 / 寿命末端 | 是 | 否 | 否 | 未报 | ✗ | ✗ | ✗ |
| Yan 2026（Zenodo 审计包） | **A** | C-MAPSS/N-CMAPSS 条件覆盖审计 | regime / RUL 带 / flight class | 是（审计用测试标签） | 否 | 部分（ACI/regime-ACI/Mondrian） | 单 seed 为主 | ✗ | ✗ | ✗ |
| HACC 2026 | **B**（摘要 + §1–2.2） | 稀疏 (group, horizon) 单元 | 单元 | 是（回溯审计） | ✅ 算术阈值 $n<\lceil1/\alpha\rceil$ | ✅ 收缩（min-group 0.28→0.72） | 30-seed 合成 | ✗ | ✗ | ✗ |
| CRULP（Piao 2025, TIM） | **B**（摘要 + CrossRef 参考文献；闭源） | CQR + UACQR 方法 | 边际 | 是 | 否 | 方法本身 | 未报 | ✗ | ✗ | ✗ |
| Barber et al. 2021 | **B** | 条件覆盖不可能性 | — | — | ✅ 理论不可能 | — | — | — | — | — |
| Gibbs et al. 2025 | **B** | 组条件覆盖保证 | 组 | 是 | ✅ 组结构 + 样本量 | 方法 | — | — | — | — |
| Bian & Barber 2023 | **B** | 校准集条件覆盖 | — | — | ✅ 覆盖是对校准集条件的随机量 | — | — | — | — | — |
| Tibshirani et al. 2019 | **B** | 协变量漂移下的加权 CP | 边际 | 否（需密度比） | 部分 | 方法 | — | — | — | — |

**读法**：本稿是唯一同时具备「同中心对照 + 多 seed + 预注册诊断检验 + 估计稳定性」四项的工作。
**但这四项是方法学组合，不是新理论**；不得据此写「首次」。

---

## 2. 逐条差异（用于 Related Work 的差异段）

### 2.1 vs DriftGuard（Moussa 2026，全文已读）

**已核实事实**（PDF 原文）：
- 作者 Maha Moussa（Utah State University / Benha University），Research Square，Posted 2026-08-07。
- 摘要：label-free 阶段报告 domain discriminability、density-ratio 的**有效样本量**、weight stabilization、
  visible abstention；建立 **exact and approximate detectability bounds**、**minimax testing consequence**、
  **sharp [0,1] identified set for target coverage under unrestricted conditional shift**。
- 正文命题：Prop.1 Estimated-ratio validity gap；Prop.2 Detectability boundary；
  Cor.1 Minimax indistinguishability；Prop.3 Approximate indistinguishability；
  Prop.4 Sharp coverage identified set；Prop.5 Audit and repair validity。
- **范围**：原文明确写其抽样区间针对 **marginal coverage**，并指出该路线「repairs marginal coverage,
  not conditional」。**没有**单位级 / 分组对象。

**差异句（可写进稿件）**：
> DriftGuard (Moussa, 2026) establishes detectability bounds for *marginal* target coverage and states
> explicitly that it repairs marginal, not conditional, coverage. We make no theoretical claim; we report
> an empirical estimation-stability result for one named calibration-side diagnostic under a frozen protocol,
> where the object is the engine rather than the target marginal.

### 2.2 vs CPA（Zhou et al. 2026，全文已读）

- CPA 把条件覆盖评估变成**有监督**任务：训练 $\hat\eta(x)$ 预测实例级覆盖概率；
  Theorem 1 给出 $\|\eta_n-\eta\|_\infty$ 的收敛界（依赖 $n_{pred}$ 与 $n_{calib}$）；
  并明确「exact conditional coverage in a finite-sample and distribution-free setting is theoretically impossible」。
- **它必须有带标签的评估集**；本稿的诊断只用校准侧信息。
- 差异句：
  > CPA estimates the conditional coverage function from a labeled evaluation set; we ask whether a
  > calibration-side diagnostic can be estimated well enough to guide the choice of scale *before*
  > deployment, and report that at 6 calibration engines it cannot (per-cell bootstrap half-width 0.75–0.90).

### 2.3 vs Robinson（IJPHM 2026，全文已读）

- 已核实：报告 **empirical coverage 88–92% at 90% nominal**（边际）；Table 4 点预测、Table 5 区间指标；
  「per-engine」只出现在**退化曲线**图（Figure 4），**没有**逐发动机覆盖分布。
- 差异句：
  > Robinson (2026) reports marginal empirical coverage (88–92% at 90% nominal) on all four C-MAPSS
  > subsets; per-engine coverage is not reported. We report the per-engine distribution and its lower tail.

### 2.4 vs Domingues & Ather（2026，全文已读）

- 已核实：全局 split conformal 在 61–90 周期剩余寿命的发动机上欠覆盖；近失效区间覆盖
  **collapsed to 37–50%**；并报「as low as 80%」。
- 差异句：
  > Domingues and Ather (2026) document the conditional collapse near end of life (61–90 cycles;
  > 37–50% near failure). We cite it as the phenomenon source and add a controlled scale-only
  > comparison across four seeds plus a pre-registered diagnostic test.

### 2.5 vs Yan（Zenodo 审计包，包内文档已读）

- Yan 审计 regime / RUL 带 / flight class 的条件覆盖，并给机制（band 误指派）；轴是**可观测分层**。
- 差异句：
  > Yan (2026) audits coverage along observable strata and explains band failures; we hold the centre
  > fixed and vary only the scale, which isolates the epistemic scale from the centre shift.

### 2.6 vs HACC（仅摘要 + §1–2.2，B 级）

- HACC 给稀疏单元的算术阈值并**修复**（收缩）。
- 本稿不修复；并明确 HACC 的阈值约束的是**单元内校准分数数目**，不是校准发动机数（D021 的教训）。
- 差异句（**须标注为摘要级比较**）：
  > HACC repairs sparse cells by shrinkage; we do not repair, and we keep the counting object explicit:
  > the number of calibration scores per cell, not the number of calibration engines.

---

## 3. 未核实清单（投稿前必须补）

| 项 | 现状 | 需要 |
|---|---|---|
| HACC 全文 | 摘要 + §1–2.2（ScienceDirect 阻断） | 机构访问或作者版；核对 Theorem 1/2 适用条件 |
| CRULP 全文 | 闭源 | 只能引摘要；稿件须标「abstract-level comparison」 |
| Barber 2021 / Gibbs 2025 / Bian & Barber 2023 / Tibshirani 2019 | 摘要级 | 若正文只引其结论（非细节），可维持 B；引用具体定理时须读原文 |

> 规则：以上任一项在稿件中只能写到已核实的层级；**不得**用摘要推断全文结论。
> 已核实全文的三篇（DriftGuard / CPA / Robinson / Domingues / Yan）可写具体事实并给出处。
