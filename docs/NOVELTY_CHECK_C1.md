# C1（CRC-RUL）查新结论：不可行

> 时间：2026-11-09 23:50。对象：`docs/IDEA_CANDIDATES_TIM.md` 的候选 C1
> （非对称维护成本下的保形风险控制，技术卖点是"非对称成本破坏单调性，需构造单调包络"）。
> 方法：定向检索（Google/Bing 索引 + arXiv + 期刊页 + DOAJ），只记录能定位到出处的条目。

## 0. 结论

**C1 的两个组成分别已被占，不能作为 TIM 的正向方法贡献。**

| 组成 | 被谁占 | 出处 |
|---|---|---|
| **技术难点**：CRC 在损失非单调时的有限样本保证 | 已有专门论文 | arXiv **2604.01502**《Conformal Risk Control under Non-Monotone Losses: Theory and Finite-Sample Guarantees》；Phil. Trans. R. Soc. A **384(2327):20250068**《Conformal risk control for non-monotonic losses》(2026-08) |
| **应用**：保形区间 → 维护调度 → 成本/停机下降 | 已有完整工作 | **Shi, H. (2026)**, *Uncertainty-Aware Predictive Maintenance Scheduling*, J. Engineering, Project, and Production Management **16(6): 2026-0090**（GBRT 点预测 + QRF 区间 + 保形校准 + MILP 调度；**分量消融专门隔离保形校准的价值**；五个名义水平 0.70–0.99 的覆盖验证；成本/停机/不必要更换三类结局） |

补充命中（同一空间的其它占位）：
- **Conformal Risk Training**（NeurIPS 2025）：把 CRC 端到端训练，控制 OCE 风险类。
- **Federated Conformal for Fleet Prognostics**（TechRxiv 2026, DOI `10.36227/techrxiv.176902673.30333692/v1`）：分布式机队的保形 UQ。

## 1. 为什么这不是"查得不够"

C1 的设计是「用 CRC 选区间膨胀量 λ，使期望维护成本 ≤ τ，难点是非对称成本导致损失对 λ 非单调」。
检索直接命中了同名同义的**非单调 CRC 理论论文**（两篇，2026），以及一篇**把保形区间接进
维护调度并做分量消融**的应用论文。这两条正好是 C1 声称的"技术难点"与"应用价值"。

继续做 C1 只能变成：
- 在 C-MAPSS 上复现 Shi 的框架（数据集不同，方法相同）→ 应用复现，不是方法贡献；
- 或换一个非单调损失的形式 → 在已有非单调 CRC 理论下的特例。

两者都过不了 `docs/PRIOR_ART_point_RUL_TIM.md` §5 的五问门第 1、2 条。

## 2. 更大的图景：区间/风险控制这条线在 2026 年末已相当饱和

| 工作 | 占住了什么 |
|---|---|
| CRULP（IEEE TIM 2025） | RUL + UACQR（ensemble CQR + cross-conformal）+ 认知不确定性 |
| Robinson（IJPHM 2026） | RUL + CQR + 风险偏好，C-MAPSS 四子集 |
| Yan 审计包（2026） | regime / RUL 带条件覆盖的测量 + ACI/regime-ACI/Mondrian |
| Similarity-CP（RESS 2026） | 工况迁移下的覆盖保证 + holds/violated/**indeterminate** 弃权诊断 |
| Stratified CP + 传感器残差（IEEE 2026） | 分层保形 + 残差融合 |
| **Shi（2026）** | **保形区间 → 维护成本/停机**，分量消融隔离校准价值 |
| **非单调 CRC（2026 ×2）** | **非单调损失的 CRC 保证** |
| Conformal Risk Training（NeurIPS 2025） | CRC 的端到端训练 / OCE 风险 |
| 训练条件覆盖（2026） | P(覆盖 ≥ 1−α \| 校准集) ≥ 1−δ |
| Federated Conformal（2026） | 机队级分布式保形 |

**这意味着**：在"保形 + RUL + 维护成本"这个交叉点上，剩下的是**很窄的缝隙**，
不是一条可以走的大路。

## 3. 还可能有缝隙的一个方向（未核实，登记为 C4）

我们独有的资产是**逐机覆盖分布**：80 格、4 seed、ICC 0.78（失效是稳定的发动机属性，
见 `docs/CLAIM_AUDIT_ijphm_v2.md`）。现有工作控制的是：
- 边际覆盖（经典保形）；
- 训练条件覆盖（控制"条件覆盖 ≥ 1−α"的概率）；
- 分组/多组有效覆盖（multivalid / Mondrian）。

**没有一篇控制"覆盖低于下限的发动机比例"**，即：

> 目标：以置信度 1−δ，保证**至多 β 比例的发动机**其单机覆盖 < 1−α。

这是一个"覆盖的分布"目标，与"覆盖的期望"（经典）和"条件覆盖的概率"（训练条件）都不同。
技术上它需要：逐机覆盖的有限样本分布 → 选 λ 使 P(#{i: C_i < 1−α}/n > β) ≤ δ。
在引擎级可交换下这可以化为次序统计量问题。

⚠ **未核实**：是否已有"fraction of units below coverage floor"的保形工作（可能藏在
multivalid coverage、group fairness、或 PAC 覆盖的文献里）。**必须先查新再谈立项。**

## 4. 交给 L2 的问题

1. 在 C1 已死的情况下，是否值得把 C4（控制"低覆盖对象比例"）作为下一个候选去查新？
2. 若 C4 也被占，是否承认"区间/风险控制线上当前没有可做的 TIM 级正向方法贡献"，
   转而换**问题**（例如把对象从"区间"换成"预测器的失效模式"）而不是换方法？
3. 用户要求"正向方法创新"，但现有全部资产（80 格、逐机覆盖、稳定性、跨域）都是**评估型**证据。
   在不动用新的大规模训练的前提下，能否构造出正向贡献？如果不能，需要多大规模的实验预算？
