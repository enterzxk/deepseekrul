# 保形 RUL 区间：饱和地图与剩余选项

> 时间：2026-11-10 00:25。目的：把本轮连续三次查新（C1 / C4 / C5）的命中合并成一张图，
> 说明**为什么"再加一个机制"这条路走不通**，并给出三个真正不同的选项。
> 每条命中都可定位到出处；未核实项单独标注。

## 0. 一句话

在「保形 + RUL + 维护决策」这个交叉点上，**后验校准、训练期校准、分组覆盖、风险控制、
弃权诊断**五个方向在 2023–2026 年都已被占。剩余空间不是"再补一个模块"，
而是**换问题**。

## 1. 饱和地图（按被占的机制分类）

### A. 后验校准（post-hoc）

| 工作 | 占住了什么 |
|---|---|
| CRULP（IEEE TIM 2025, `10.1109/TIM.2024.3522678`） | RUL + UACQR（ensemble CQR + cross-conformal）+ 认知不确定性 |
| Robinson（IJPHM 2026, `10.36001/ijphm.2026.v17i1.4724`） | RUL + CQR + 风险偏好，四子集 |
| Yan 审计包（Zenodo 2026, `10.5281/zenodo.21330745`） | regime / RUL 带条件覆盖的测量 + ACI / regime-ACI / Mondrian |
| Similarity-CP（RESS 2026, `S0951832026014365`） | 工况迁移覆盖保证 + holds/violated/**indeterminate** 弃权 |
| Stratified CP + 传感器残差（IEEE 2026, Xplore 11708000） | 分层保形 + 残差融合 |
| **Boosted Conformal Intervals**（Stanford/UChicago） | **用梯度提升后验调整 conformity score，专门让子组覆盖更一致** |

### B. 训练期（training-time）

| 工作 | 占住了什么 |
|---|---|
| **Conformal Training (ConfTr)**（Stutz et al., ICML DFUQ 2022） | 端到端可微 conformal 损失；明确改善 **class-conditional** inefficiency |
| **Fair Conformal Classification via Learned Groups**（2026, arXiv 2605.12195） | 学表示使分组条件覆盖成立 |
| **Batch Multivalid Conformal**（Gupta et al., NSF PAR） | 学阈值函数 f(x) 使**任意组集合**都有效覆盖 |

### C. 风险控制 / 决策

| 工作 | 占住了什么 |
|---|---|
| **非单调 CRC**（arXiv 2604.01502；Phil. Trans. R. Soc. A 384(2327):20250068） | 损失非单调时的 CRC 有限样本保证（= C1 的技术卖点） |
| **Shi (2026)** EPPM 16(6):2026-0090 | 保形区间 → MILP 维护调度；**分量消融隔离校准价值**；五档覆盖验证；成本/停机结局（= C1 的应用） |
| Conformal Risk Training（NeurIPS 2025） | CRC 端到端训练 / OCE 风险 |
| Conformal Tradeoffs（arXiv 2602.18045） | 覆盖之外的保证与运营权衡 |

### D. 统计保证的对象

| 工作 | 占住了什么 |
|---|---|
| 经典容忍区间（Wilks 1941 起） | "n 要多大，实现覆盖才有给定稳定性"（= C4 的目标） |
| 训练条件 / PAC 覆盖（arXiv 2502.07497） | P(覆盖 ≥ 1−α \| 校准集) ≥ 1−δ |
| 层级 / 聚类保形（Sci. Rep. 2026, s41598-026-37450-w） | 簇内不独立时的保形（= 发动机即簇） |
| 子组覆盖坍塌机制（arXiv 2608.04254） | 校准格变小 → 子组覆盖下降的闭式解释 |
| Missingness-aware CP（2026, arXiv 2609.30781） | 跨站点漂移下的池化 vs 站内覆盖分解 |
| **Calibrating a cumulative band（2026-11）** | **"累积带的覆盖受校准窗口数限制，而不是周期数；只有窗口数达到数十时才达到名义"** |
| **Exceedance design effect（Noonan 2026, Zenodo 21798943）** | 依赖数据下覆盖的**分布**（不是界）；校准集 N=100 时"约 46% 会…"的定量论证 |
| **Cluster-level conformal inference（Statistica Sinica SS-2025-0476）** | 簇级目标的保形区间构造 |

> 最后三条把「选项 1（测量协议）」的核心也占掉了：**"覆盖指标取决于校准单元数/聚合方式"是已知结果**，
> 只是没有人在 C-MAPSS 上把它讲一遍。在 RUL 上讲一遍是应用，不是方法。

## 2. 三个死掉的候选与本轮的结论

| 候选 | 死因 |
|---|---|
| C1 非对称成本 CRC | 理论（非单调 CRC）与应用（Shi 2026）分别被占 |
| C4 控制"低覆盖对象比例" | 归约为引擎级 PAC 容忍界；且 β=0.1/δ=0.05 需 29 台独立校准引擎，我们只有 6/16/6/15 |
| C5 训练期让分数跨引擎同质 | ConfTr（分组条件）、Boosted CP（子组一致）、Batch Multivalid（任意组）已占 |

**共同病根**：全部是"把成熟保形工具换一个对象名"。按五问门第 1、2 条，都过不了。

## 3. 三个真正不同的选项（供 10:00 决策，不是现在问）

### 选项 1：换问题——测量有效性（最贴 TIM，且我们已有独特证据）

**问题**：C-MAPSS 的**评价协议本身**决定了报告出来的 RUL 误差与覆盖。
我们已发现两个具体事实（D032 与并行线记录）：

1. `experiments/punet_e2e/*/preds.csv` 的**末窗口是 RUL=0 的寿命终点**；用它做引擎级分析会把覆盖算高。
   改用 test-like 截断点后，冻结 PICP 从"看起来 0.9"变成 **0.80**。
2. cap-125 分段标签使"退化动力学"与"标签形状"不可分离。

**可做成正向贡献的形态**：不是"审计别人的数字"，而是提出一个**截断一致（truncation-consistent）
的评价协议**，并证明它在四子集上改变排序/覆盖结论；给出协议的形式定义、可复算脚本与推荐报告项。
TIM 作为 I&M 期刊，对"测量协议"类贡献有明确位置。

**风险**：用户已否决"审计型"贡献；必须做成"协议 + 纠正方法"，不是"指出别人错"。
**成本**：0 GPU（全部可从 a40 现有预测重算）。
**待查新**：C-MAPSS 评价协议 / 截断偏差 / 末窗口伪影 是否已被系统讨论。

### 选项 2：换对象——同时（simultaneous）轨迹带

**问题**：现有工作全部是"每个时间点/每个窗口的 RUL 覆盖"。维护决策需要的是
**整条剩余退化轨迹**的同时覆盖（一条带，以 1−α 概率包含整条未来路径）。

**机制**：把保形从标量响应搬到**函数响应**（轨迹），用 sup-norm 非一致性分数，
得到同时带；其有限样本保证与逐点覆盖不同（更难，也更有决策意义）。

**风险**：函数型保形（functional CP）与同时置信带是成熟统计工具，可能只是搬运。
**成本**：需要新训练（轨迹级模型），先导 ≤20 GPU-h。
**待查新**：functional / simultaneous conformal for RUL trajectories。

### 选项 3：承认当前不可行，转为方法程序（需用户加预算）

**结论**：若选项 1、2 的查新也失败，则当前资产确实不足以支撑 TIM 级正向方法贡献
（这正是 L2 Q3 的裁决）。此时应把目标改成"设计一个需要 3–6 个月、独立数据与算力的方法程序"，
而不是在冻结矩阵上继续找缝隙。

## 4. 下一步（本轮之后立即执行）

1. 对**选项 1** 做定向查新（0 GPU，≤1 h）：关键词 `C-MAPSS evaluation protocol`、
   `truncation bias remaining useful life`、`last cycle artifact RUL benchmark`、
   `benchmark validity prognostics`。
2. 对**选项 2** 做定向查新（0 GPU，≤1 h）：关键词 `functional conformal prediction`、
   `simultaneous prediction band`、`conformal for trajectories`、`sup-norm nonconformity`。
3. 二者只要有一个通过新颖性门，就写六件套预注册（含数学规格与判定门），
   按 L2 预算（先导 ≤20 A40 GPU-h）执行；都不通过则如实报告"当前无 TIM 级候选"。

## 5. 追加关闭：cap-125 标签工程（2026-11-10 03:00）

候选思路：C-MAPSS 的 PwL-125 截断标签是**标签变换**而非删失（训练机均 run-to-failure，真实 RUL 已知），
因此可以主张「截断标签使区间对长寿命发动机失效」，并做去截断的校准/训练。

**查新结果：已被占。**

| 工作 | 占住了什么 |
|---|---|
| **Phase-Aware Probabilistic Multitask Learning**（IEEE Xplore 11739429，2026-10-15） | 明确指出 PwL-125 截断 + 确定性回归导致过度自信，提出 **Phase-Aware Probabilistic Target Engineering**（相位监督 + 退化感知样本加权） |
| MDPI Aerospace 13(9):826（2026） | 明确分析截断造成 early-life plateau，并说明官方 test RUL 保留用于评估 |
| Ramasso & Gouriveau 的 PwL 约定 | 截断标签的原始出处与理由 |

另：删失保形（Candès et al., Biometrika 2024 及后续 adaptive cut-offs）与「删失 RUL 预测」也已存在，
但**前提不适用**——cap-125 不是删失，真实 RUL 已知。

**结论**：cap-125 这条也关闭。至此六个候选（C1/C4/C5/测量协议/cap-125/同时轨迹带）
全部被占或前提不成立。
