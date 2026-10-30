# 创新差距评估（P1）

日期：2026-10-21
状态：**初稿**。文献部分只记录可核查的元数据；未能取得全文的一律标 `UNVERIFIED`，
**不引用任何未读到的数字**。

---

## 1. 先把候选的数学写清楚（以真实代码为准）

当前最强单元（M4）实际做的事：

```
参考：  x_ref(t) = n/(n+τ)·x̄_(i,m(t)) + τ/(n+τ)·μ_pop,m(t)      τ=2
证据：  e = [n/5, 1{n≥1}, exp(-age/30), exp(-|x-x_ref|), 1/(1+σ_ref), |x-EMA(x)|]  ∈ R^6
门控：  g(t) = σ( MLP(e(t)) + b_sensor )  ∈ (0,1)^14
表征：  z(t) = Proj_raw([x, EMA(x)]) + g(t) ⊙ Proj_ref(Δx)
```

**其中没有**：Sensor-ID（`use_sensor_id` 未启用）、注意力图、对比学习、不确定性头。
e4 是传感器空间偏离，不是工况中心距离。

## 2. 哪些部分是"常见技术"，不能当创新

| 技术 | 在文献中的常见程度 | 我们的判断 |
|---|---|---|
| 减去早期健康基线（residual referencing） | PHM 中长期使用 | 不是创新 |
| EMA 去噪、cycle 坐标 | 极其常见 | 不是创新 |
| 门控融合两个特征分支 | 通用结构 | 不是创新 |
| 按工况聚类的归一化 | 多工况 RUL 常见做法 | 不是创新 |

**结论：RACDR 目前不是"新方法"，它是"参考残差 + 证据门控"的组合。**
若要在高水平期刊立住，必须给出**别人没有的、可证伪的机制性主张**。

## 3. 三条可证伪的候选主张（择一深挖）

**H1（可靠性可辨识）**：门控值 g(t) 应当与"参考实际误差 |x_ref − x_true_healthy|"
负相关；即模型确实学到了"参考何时不可信"，而不只是学到一个平滑权重。
*判别实验*：在验证集上，用同一台发动机的群体参考/个体参考差异近似参考误差，
计算 `corr(g, -|x - x_ref|)` 与分位数分层误差；若 g 与参考误差无关 → H1 假，
"可靠性"只是命名。

**H2（资源配置效应）**：收益来自"把参考信息按可信度分配到通道"，而不是参数量。
*判别实验*：固定门控（全 1 / 全 0.5 / 可学习常量标量）对比证据驱动门控；
参数量对齐。若固定门控同效 → H2 假。

**H3（工况条件性是必要的）**：参考的"分工况"属性是收益来源。
*判别实验*：M4 vs M4b（群体先验 + 门控）。已有 3 seeds：15.077 vs 16.214 → 支持 H3。
但需注意这与"无门控时群体更好"的交互，说明是**交互效应**而非单因素。

**当前证据：H3 有 3-seed 支持（FD002）；H1、H2 尚未检验。**

## 4. 路线（≤2，优先 A）

**路线 A（优先）：Evidence-Identifiable Reference Fusion**
把"参考可信度"从隐式门控变成**可辨识、可校验**的量：
- 显式定义参考误差代理 `r(t) = |Δx(t) − E[Δx | 健康阶段]|` 或跨参考一致性；
- 训练时加入"门控—参考误差"一致性约束（用可观测的参考离散度作代理标签）；
- 校验时报告 `corr(g, reference disagreement)` 与分层误差，作为机制证据。
*与已有工作的差异*：不是"再加一个门控"，而是**把门控变成可测量、可反驳的估计量**，
并给出校准曲线。若相关性不成立，路线 A 直接失败（可证伪）。

**路线 B（备选）：Late-Risk-Aware Reference Fusion**
把晚报风险显式建模为条件风险（按工况/寿命段分层），用分位数或单调约束控制
severe-late。当前 M4 的 severe-late 反而更差（13.0 vs 11.3），说明这条有真实缺口。

## 5. 文献对照表（只列可核查元数据，未读全文）

| # | 工作/主题 | 来源（可核查 URL） | 机制 | 协议 | 结果 | 与我们差异 | 核实状态 |
|---|---|---|---|---|---|---|---|
| 1 | 多工况 RUL（TIM 2025 族） | ieeexplore 11690590 | — | — | — | 待读 | UNVERIFIED |
| 2 | RUL 综述/结构健康 | sage 10.1177/14759217251410489 | — | — | — | 待读 | UNVERIFIED |
| 3 | C-MAPSS 原始数据 | NASA DASHlink 2008 PHM | 数据集 | FD001–FD004 | — | 数据来源 | 可核查（数据集） |
| 4 | RUL MEMS 应用 | flore.unifi.it RUL_MEMS.pdf | — | — | — | 待读 | UNVERIFIED |
| 5 | 多工况退化建模（arXiv 2604.13459） | browse-export.arxiv.org | — | — | — | 待读 | UNVERIFIED |
| 6 | 电池/设备 HI 双任务 | — | — | — | — | 待读 | 待检索 |
| 7 | 参考基线/unit-wise 校正 | — | — | — | — | 待读 | 待检索 |
| 8 | 非对称风险损失 | — | — | — | — | 待读 | 待检索 |

**说明**：本次检索只拿到链接，未取得可读正文；IEEE TIM 全文为付费墙。
按目标要求"不编造数字"，表中不填任何结果。补齐方式：把 PDF 放入
`docs/lit/`，下一轮可逐篇填表；或由主控提供可访问副本。

## 6. 对"RACDR 新在哪里"的当前回答

**尚不能回答"新在哪里"**。目前能说的是：
- 它比"纯参考残差"多了一个**证据驱动的逐通道门控**；
- 该门控与"分工况参考"存在**交互效应**（3 seeds，FD002），这是可测的现象；
- 但"门控是否真的在估计可靠性"（H1）与"收益是否只来自参数/结构"（H2）**都未验证**。

在 H1/H2 完成前，**不把 RACDR 作为论文方法名**，也不声称方法创新。

## 7. 下一步（本轮内可做）

1. 低成本判别实验：从已有 M4 checkpoint 计算 `g` 与参考不一致度的相关性（H1 的第一层检验，不需重训）。
2. 若 H1 不成立 → 立即转向路线 B（晚报风险），并在计划中登记切换原因。


---

## 8. 可核实的文献扫描（arXiv API，2026-10-22，含摘要）

扫描方式：`export.arxiv.org/api/query?search_query=all:"C-MAPSS"`，按提交日期倒序取 15 条，
原文摘要已存 `results/dsh/arxiv_scan.json`。以下为**已读到摘要**的条目（非付费墙推测）。

### 8.1 与"晚报风险权衡"直接重叠（重要）

**Calibrating Prediction Timeliness Through Multi-Objective Hyperparameter Optimization
for Remaining Useful Life Prediction**（arXiv 2610.01530，2026-10-01）

原文摘要要点：*"early and late RUL prediction errors carry asymmetric consequences,
yet hyperparameter optimization typically targets a single accuracy metric that treats
both directions equally... treats the optimization objective itself as a design variable.
Five architectures ... evaluated under three regimes"*。

**对本项目的影响**：我的"精度–安全 Pareto 曲线"（非对称偏移、severe-late 降 30–40%、
代价 3.4–8.4% RMSE、不存在全域最优）**与该文的动机与结论高度重叠**。
差异仅在于我用的是**后验校准**（单一偏移），他们用的是**多目标超参优化**。
**因此该条不能作为独立创新主张**，只能作为"与已有工作一致的独立复现"或"更轻量的实现"。

### 8.2 与"评估可靠性"部分重叠

**Do RUL explanations hold up? Faithfulness and stability of attributions on C-MAPSS**
（arXiv 2610.04278，2026-10-03）
研究 C-MAPSS 上归因图的可信性与稳定性，使用官方 FD001/FD003 划分与 PHM08 非对称评分。

**影响**：与我的"单点评估不稳定"主题相邻但不同——他们评估的是**归因解释的稳定性**，
我评估的是**指标本身的采样方差**。可作为相关工作引用，但需要在正文中明确区分。

### 8.3 方法论亲缘

**Serverless gossip training of LSTM failure detectors: A matched-protocol comparison...**
（arXiv 2609.35792）明确使用 "matched-protocol comparison" 的思路。
说明"公平协议对照"在本领域已有实践，我的评估主张需要更具体的增量（采样方差分解）。

### 8.4 结论：创新的真实位置被压缩了

| 原拟贡献 | 文献状态 | 处置 |
|---|---|---|
| 精度–安全 Pareto 权衡 | **已被 arXiv 2610.01530 覆盖** | 降级为复现/轻量实现，不作创新 |
| 评估协议批判 | 部分重叠（2610.04278 邻近） | 需收窄到"采样方差"这一具体点 |
| **逐传感器特征必须配通道独立主干（打乱消融 2.9×/4.1×）** | **未在本次扫描中出现** | 目前唯一未被覆盖的候选 |
| 分工况参考有害 | 未出现（但属负结果，价值有限） | 作为支撑性证据 |

**因此后续研究的唯一候选创新点是 8.4 第三行**，而它还需要：
（i）针对"通道独立"相关工作（PatchTST 类）的定向检索；
（ii）确认"特征–主干交互"未被报告。


---

## 9. 定向检索：通道独立（CI）文献（arXiv API，2026-10-22）

查询 `all:"channel independence" AND all:"time series"` 返回的**已读标题**：

* CSformer: Combining Channel Independence and Mixing for Robust Multivariate TS Forecasting (2023-12)
* **Is Channel Independent strategy optimal for Time Series Forecasting?** (2023-10)
* **The Capacity and Robustness Trade-off: Revisiting the Channel Independent Strategy for Multivariate TS Forecasting** (2023-04)
* IConv: Focusing on Local Variation with Channel Independent Convolution (2025-09)
* Anomalies in Multivariate Time Series Benchmarks Are Mostly Univariate (2026-06)

查询 `all:"remaining useful life" AND all:"channel-independent"` → **0 条**。

### 9.1 这对我的"最后候选创新"意味着什么

我的交互发现（A 主干 = 通道独立 + 共享时序 vs tfA = 通道混合）**本质上就是
CI vs CM 的比较**，而这个比较在时序预测领域已有系统研究，尤其是
"Capacity and Robustness Trade-off" 一篇直接讨论 CI 的收益与代价。

**因此我的发现很可能是一个已知权衡在 RUL + 逐传感器工程特征上的特例**，而不是新机制。
我的增量只在于两点：

1. 在 RUL 场景（而非预测场景）上验证；
2. 用**打乱消融**证明"特征是逐传感器的、错配即崩"，把 CI 的必要性从"经验更优"
   变成"语义上必须"。

第 2 点是唯一可能站得住的增量，但它是一个**论证方式**的增量，不是方法增量。

### 9.2 诚实的创新评估

| 候选 | 文献覆盖情况 | 判断 |
|---|---|---|
| 精度–安全 Pareto | arXiv 2610.01530 | **先占，不可用** |
| 评估采样方差 | 部分（归因稳定性那篇相邻） | 弱 |
| CI-vs-CM 交互 | **已有时序预测系统研究** | **属特例，非新机制** |
| 打乱消融作为 CI 必要性的语义论证 | 未见 | **唯一可辩解的增量，但偏方法学** |

**结论：当前方向已无足以支撑"高于 TIM 常规水准"的方法创新空间。**
继续在此方向上追加实验，收益极低。


---

## 10. 新假设（P1 重启）：问题的重心从"谁更好"转向"能否分辨"

在确认方法创新空间耗尽后，我把问题重述为一个**可量化、可复现、且与测量学直接相关**的形式：

> **在 C-MAPSS 上，文献中常见的"方法 A 比 B 好 1–2 RMSE"这类结论，
> 在给定的发动机数量下能被数据分辨吗？**

这不是模型问题，是**测量分辨率**问题——正是 TIM 关心的对象。

### 10.1 排序可分辨性（`results/dsh/ranking_resolvability.csv`）

13 个单元、52 台发动机、78 个方法对，随机单点重采样 1000 次：

* **35/78（44.9%）的方法对排序未定**（0.25 < P(A 更好) < 0.75）；
* **19/78 是"抛硬币"**（0.4 ≤ P ≤ 0.6）。

### 10.2 功效分析（`results/dsh/power_analysis.csv`）

逐发动机 RMSE 配对差的**标准差中位数为 4.24**。
按 80% 功效、α=0.05 计算所需发动机数：

| 效应量 | 所需发动机数 |
|---|---:|
| 2.16 RMSE（本研究中最大差距） | **371** |
| 1.76 RMSE | 604 |
| 中位数 | **1683** |

**当前 52 台发动机下，78 个方法对中 0 个可被分辨。**

### 10.3 必须说明的修正与边界

* 初版功效分析误用逐发动机 **MSE**（重尾，sd≈150），算出"两百万台"的荒谬结论；
  已改为逐发动机 **RMSE**（sd≈4.2），本文件记录的是修正后结果。
* 本计算基于**本项目协议**（80/20 划分，52 台验证机）。文献常用"全量训练 + 官方 test"，
  其发动机数约为 100，方差结构亦可不同。因此**不能直接断言文献结论都不可分辨**，
  只能说：**在同等方差量级下，1–2 RMSE 的差距需要数百台发动机**，这是一个可核查的量级提示。
* 该结论不依赖任何模型优劣，是一个纯粹的统计分辨率事实。

### 10.4 为什么这条可能是可发表的

1. 它解释了为什么 C-MAPSS 上同类方法的排序经常互斥——不是方法问题，是分辨率问题；
2. 它给出可操作建议（报告配对 CI、报告所需样本量、避免单点截断）；
3. 它可以用任何人现有结果复算（脚本已提供）；
4. 它符合 TIM 对"测量可靠性、可复现性"的要求，而不需要新模型。
