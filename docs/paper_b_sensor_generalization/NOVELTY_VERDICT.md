# NOVELTY_VERDICT — E4 Sensor Identity / E5 Low-Rank Sensor Interaction

> study: `paper_b_sensor_generalization` · 分支 `research/paper-b-sensor-generalization`
> 判定日期：2026-11-15 · 判定人：文献员
> 依据：`LITERATURE_MAP.md`（同目录，逐条附 DOI/arXiv 与阅读深度）+ 仓库代码事实 + 团队已冻结的实验结论。
> 口径：只判断「机制本身是否已被公开工作包含」，不判断「在 DSPT 上有没有增益」。增益是实验结果的事，见 `SENSOR_RESEARCH_PLAN.md` §7。

---

## 0. 结论（先给，不绕）

**E4 不构成新颖性。** 它不是「可能不够新」，而是**已经被精确做过**：Xu et al. 2026（*Meas. Sci. Technol.* 37 245102）式 (6) 就是 `z_{t,i} = u_{t,i} + δ_i`，逐传感器可学习向量加在进入跨传感器模块之前；同一篇还在 C-MAPSS 14 传感器、d=64 的设置上做了 sensor attention。更硬的一条是内部事实：本仓库 `paper_b/models/dspt.py` 的 `sa_mode='identity'` 就是这段代码，`SAM_identity_L70` 已经 12/12 跑过、未进冻结清单。

**E5 机制层面不构成新颖性。** 它的代数结构——低秩因子 + 回投 + 小尺度 λ + 保留原表示——与 Xu et al. 2026 §3.3 的「低秩残差扰动」同构（rank=8、λ=0.10、gate-on-update），而「低秩 + 残差」在 LMF（ACL 2018）、LoRA（ICLR 2022）、low-rank bottleneck（ICML 2020）里各自独立成立。E5 与它唯一的实质差异是**作用对象从邻接矩阵换成传感器特征张量**；这是位置差异，不是机制差异。

**因此：本轮不能靠 E4/E5 形成 IEEE TIM 的「方法创新」。** 若实验只显示「不比 baseline 差 / 略好一点」，按团队已冻结的口径（96 格消融 28 项对比均未达区分门槛）不能主张模块更优，E5 也不成立。唯一可能站住的写法是把问题从「提出模块」改写成「传感器交互的低秩假设在什么条件下成立/失效」的**机制研究**，其上限是 ENGINEERING-ONLY，且需要额外证据（见 §4）。

> **定稿补记（2026-11-16）**：E5 实测 `lr4` 变差 **+0.6714** [+0.2401, +1.1129]，低秩路线关闭；E1a/E1b 确认集复现（4/4），达 ENGINEERING-ONLY。最终判定见 **§10**：**总档位 ENGINEERING-ONLY（E1a/E1b）；方法创新 STOP、低秩路线关闭**——不是总档位 STOP。不构成新颖性的终稿清单与可发表的刻画也在 §10。

---

## 1. E4 逐条判断

**被判断的主张**：在注意力前注入 sensor id，或把轻量 id adapter 放在逐传感器时间 Transformer 输入端，能让模型区分传感器身份并改善 RUL。

| 问题 | 回答 |
|---|---|
| 与哪一篇的哪一部分重叠 | **Xu et al. 2026, §2.2 式 (6)**：`z_{t,i} = u_{t,i} + δ_i`，`δ_i ∈ R^K` 是逐传感器可学习向量，「captures sensor-specific biases and identity-related characteristics」，加在进入 GLSTM 之前。我们的 `st = st + self.sensor_id` 是同一行代数，只是注入点在 `sensor_attn` 之前。 |
| 还有谁覆盖 | **Garzón et al. 2025（IEEE ETCM）**：C-MAPSS 上的 sensor attention + operational context embeddings，已经把「传感器身份 + 工况上下文」一起做掉；**iTransformer（arXiv:2310.06625）** 的 variate token 让每个传感器自带身份；**TFT（IJF 2021）** 的 static covariate encoders 是同一归纳偏置。 |
| 差异是什么 | (i) 注入位置：`sensor_attn` 前 vs GLSTM 前；(ii) 我们只有 id，没有工况 context；(iii) 「id adapter」只是同一向量的另一种参数化（查表 vs 小 MLP 生成），不是新机制。 |
| 差异是否构成贡献 | **否**。三项都是实现位置/参数化差异，审稿人一句「standard sensor embedding」即可归位。 |
| 内部事实 | `dspt.py:147-148` 定义 `sensor_id`；`:240-243` 执行加法；`RUN_INVENTORY.md:69` 记录 `SAM_identity_L70` / `SAM_cond_L70` 各 12/12 已跑、未进冻结清单。**E4 在本项目里不是新模块，是一个已有开关。** |
| 判定 | **不构成新颖性。建议不作为论文贡献；最多作为「我们试过、无效/无差异」的控制实验，或不跑。** |

**补充**：任务书里 E4 的三臂对比（base SA / 注意力前 id / 输入端 id adapter）在方法学上是**消融**，不是方法。它唯一的价值是回答「身份信息在 DSPT 里是否已被 sensor attention 隐式学到」——这是一个诊断问题，答案大概率是「已学到」，因为 `sensor_score` 的 softmax 池化本身就可以给不同位置不同权重。

---

## 2. E5 逐条判断

**被判断的主张**：`H: [B,T,14,D]` 经 14×r 基矩阵 `U` 投影到 r 维，`G` 变换后回投，`H_new = H + λ·U·G(UᵀH)`，r=4，作为完整 sensor attention 的替代。

| 问题 | 回答 |
|---|---|
| 与哪一篇的哪一部分重叠 | **Xu et al. 2026, §3.3**：把图变化建模为 prior adjacency 周围的低秩残差扰动，`A_dyn = normalize(A_prior + λ·sym(a_t b_tᵀ))`；低秩因子由 context 产生；Table 6 给出 **rank=8、residual scaling=0.10、gate-on-update enabled**。这与 E5 的「低秩因子 + 回投 + 小尺度 λ + 保留原表示」是同一代数结构。 |
| 还有谁覆盖 | **LMF（ACL 2018）**：每个模态一个低秩因子矩阵，交互在低维因子空间完成——E5 的 `U` 与 `G` 与它是同一角色；**LoRA（ICLR 2022）**：`W + BA`，低秩增量 + 残差加回 + 小尺度初始化；**low-rank bottleneck（ICML 2020）**：低秩会限制表达力。 |
| 差异是什么 | (i) 作用对象：邻接矩阵（14×14）vs 特征张量（14×D，逐时间步）；(ii) 低维变换：双线性因子 vs 非线性 MLP；(iii) 定位：替代 sensor attention vs 叠加在 GLSTM 上；(iv) r=4 vs 8。 |
| 差异是否构成贡献 | **否，四项相加仍不足。** (i) 是位置差异，而且丢掉图先验；(ii) 在 LMF/LoRA 有先例；(iii) 是架构选择，任务书自己也把它写成「先替代、再考虑叠加」；(iv) 是超参。 |
| 有没有真正的空白 | 有**一条缝**：我没有检索到「在 C-MAPSS 的传感器特征维（而非图/邻接）上做低秩投影—非线性—回投—残差」的直接前作。但这是**作用位置**的空白，不是**机制**的空白。IEEE TIM 不会因为「同一个低秩残差结构换了个轴」给方法贡献。 |
| 替代解释（必须排除） | A4 的结论直接适用：r=4 是在传感器维制造低秩瓶颈。如果 r=4 比 full-rank 好，可能是正则化；如果更差，可能是欠拟合。**必须先证明 r=4 不是容量效应**，否则任何增益都归因不到「低秩结构」。 |
| 判定 | **机制新颖性不足。** 可以作为「机制探针」跑，但不能作为方法创新写。 |

**一句可被审稿人验证的 delta（如果一定要写）**：
> On DSPT/C-MAPSS, a rank-4 sensor-space interaction matches full-rank (r=14) and beats an equal-parameter MLP, and the learned 14×4 basis is stable across seeds and maps onto physically interpretable sensor groups.

这句话里真正值钱的是**后半句**（基的稳定性与可解释性），不是「rank-4 有效」。如果实验只能给出前半句，那它就是一次普通的消融，不构成贡献。

---

## 3. 明确不构成新颖性的部分（清单）

1. **把 sensor id 拼进/加到输入**——Xu et al. 2026 式 (6)、iTransformer variate token、TFT static covariate 都已覆盖；本仓库 `sa_mode='identity'` 已实现并跑过 12 格。这是多传感器融合与多变量时间序列的常识。
2. **「轻量 id adapter」**——查表向量与小 MLP 生成是同一向量的两种参数化；换参数化不是新机制。
3. **低秩投影 + 非线性 + 回投 + 残差**——LMF、LoRA、low-rank bottleneck 各自独立成立；Xu et al. 2026 已把同一结构用在传感器图上。
4. **条件 embedding / 工况门控 / 条件自适应低秩**——CTNet（Measurement 2025）、Garzón 2025、Aasi 2026、TFT 2021、Xu 2026（context-dependent 低秩因子）、Li 2026（condition modulation）都已覆盖；stage-3 的方案是这些工作的组合。
5. **训练期 mask/dropout + 显式 availability mask**——缺失数据文献（BRITS 2018、RobustUQ 2025、Tang 2026）与旧 study 的 36 格已覆盖；clean 通道全负。
6. **等参 MLP / full-rank / no-branch 对照**——这是审稿人**期待**的对照，做了不加分，不做会被打。
7. **r=2/4/8 的敏感性扫描**——超参扫描不是机制贡献。
8. **在 DSPT 上「加一个模块」本身**——团队已冻结：96 格消融 28 项对比均未达区分门槛，base 本身已很好，硬证模块更优会给审稿人靶子。

---

## 4. 如果一定要做：什么条件下 E5 才可能成立

不是「跑出正结果就成立」，而是**先预注册机制假设，再用实验判它**。可接受的写法只有一种：把 E5 当探针，回答「传感器交互是否真的低秩」。

预注册建议（在跑之前写死）：

1. **H1（低秩假设）**：DSPT 学到的传感器交互矩阵的有效秩显著低于 14，且 rank-4 重构能保留大部分交互能量。检验：对 `sensor_attn` 的 QK 相似度或传感器协方差做谱分析，报告前 4 个奇异值占比。
2. **H2（不是容量效应）**：r=4 与 full-rank r=14、等参 MLP、no-branch 三臂对比；只有 r=4 不劣于 full-rank 且优于等参 MLP，才排除「容量/正则化」解释。
3. **H3（结构可解释）**：`U` 的 4 个基向量在 3 个 model seed 间稳定（子空间主角度），且与物理传感器组（温度/压力/转速/流量）有可复现对应。
4. **H4（条件刻画）**：低秩交互的增益/损害集中在特定工况或特定缺失/噪声水平；报告「何时有用、何时有害」，而不是平均 ΔRMSE。

只有 H1–H4 至少三条成立，E5 才能从「换位置的消融」升级为「关于传感器交互低秩性的实证发现」。此时论文的主张也必须是**发现**，不是**模块**。

反面条件（出现即停）：

- r=4 显著差于 full-rank → 低秩假设在 DSPT 上不成立，停。
- r=4 与等参 MLP 无差异 → 增益来自参数量而非低秩结构，停。
- `U` 在 seed 间不稳定 → 「传感器基」不可解释，停。
- 平均 ΔRMSE 落在种子噪声内 → 按团队口径不能主张，停。

---

## 5. Rebuttal 预演（审稿人最可能说的一句话）

> "Sensor identity embedding and low-rank residual sensor interaction are already present in Xu et al. (2026, *Meas. Sci. Technol.* 37 245102), on the same C-MAPSS 14-sensor, d=64 setting. What is new?"

我们能答的只有：

> "They apply the low-rank residual to the adjacency matrix; we apply it to the sensor feature tensor, replacing full sensor attention."

这个回答的问题在于：**它承认了机制相同，只主张位置不同**。审稿人下一句会是「那为什么不直接用他们的图，图还有物理先验？」——而我们没有证据回答。除非 H3（基的可解释性）成立，否则这条 rebuttal 站不住。

**审稿人最可能引用的三篇**：Xu et al. 2026（A1）、Garzón et al. 2025（C2）、Li et al. 2026（B2）。备选：CTNet（C1）。

---

## 6. 给经理的建议（对应 GO / ENGINEERING-ONLY / STOP）

> **结果后**：本节的条件建议已被 §10 取代；最终档位见 §10.1/§10.5——总档位 **ENGINEERING-ONLY**，方法创新 STOP，低秩路线关闭。

| 选项 | 建议 | 理由 |
|---|---|---|
| **E4 立项** | **不立**（0 GPU） | 机制已被精确覆盖 + 本仓库已实现并跑过 12/12。再跑一遍 split_seed=17 只能得到「又一个不显著」，不产生新信息。若要留证据，把 `SAM_identity_L70` 的旧结果按新口径重算并注明协议即可。 |
| **E5 立项** | **只作为机制探针**，不作为方法候选 | 若跑，最小网格：`r=4` + `full-rank r=14` + `等参 MLP` + `no-branch`，4 子集 × 3 seed = 48 格；先跑 FD001/FD004 两子集做预检，不通过就停。预注册 §4 的 H1–H4。 |
| **论文定位** | 传感器部分写成「条件分析/负结果」，不写「新模块」 | 团队已冻结「不主张模块更优」；文献上 E4/E5 也不支持方法创新。真正可能成立的贡献是「低秩假设在什么条件下成立」的机制发现。 |
| **档位** | E4/E5 单独**不足以 GO**；E5 步骤 0 触发 → 低秩路线关闭。总档位由 E1/E2 决定：E1a/E1b 达 **ENGINEERING-ONLY**，不是 STOP（S-1 前件不成立） | GO 要求「稳定、可归因、有新颖性基础」；E5 只有「可归因」的一半，没有「新颖性基础」；而 §7.1 的 STOP 要求「没有任何臂达到 ENGINEERING-ONLY」，E1a/E1b 达到了。 |

**一句话给决策会**：E4 关掉；E5 可以花 48 格买一个「低秩假设是否成立」的答案，但买不到方法创新。如果目标是 IEEE TIM 的方法贡献，应把资源转回协议/基线/机制归因这条线，而不是继续在 DSPT 上叠模块。

---

## 7. 核查记录

| 文献 | 标识 | 核验方式 | 核验到的关键内容 |
|---|---|---|---|
| Xu et al. 2026 | DOI 10.1088/1361-6501/ae7463 | IOP 原文页（open access），读到 §2.2 式 (6)、§3.3、Table 6/9 | `δ_i` 加法；低秩残差扰动；rank=8、λ=0.10、gate；sensor attention 64,064 参数；C-MAPSS FD001–FD004 |
| Garzón et al. 2025 | S2 CorpusID 284229047 | Semantic Scholar 题录页 | sensor attention + operational context embeddings；C-MAPSS；ETCM 2025 pp.1–6 |
| CTNet | DOI 10.1016/j.measurement.2024.116345 | Crossref + ScienceDirect 原文页 | 工况—传感器图 + causality attention；C-MAPSS、N-CMAPSS、QAR；作者 Liu/Sun/Wang/Zhang |
| Li et al. 2026 (HPVSGNN) | DOI 10.1016/j.ast.2026.112160 | Crossref + ScienceDirect 原文页 | condition modulation + 动态图 + VIB 门控 + 时频解耦；噪声/缺失鲁棒 |
| HDA-GNN | DOI 10.1109/TIM.2025.3569885 | IEEE 页 + 两个二手题录 | 异构动态图、约束关系变化；IEEE TIM 74:3538014 |
| ADFG | DOI 10.1109/JSEN.2025.3546498 | Crossref + IEEE 页 | 先验图 + 动态相似图自适应融合；GAT + BiGRU；C-MAPSS |
| RobustUQ | DOI 10.1016/j.ress.2025.111177 | ScienceDirect + NSTL 题录 | 随机缺失/部分故障 + 扩散模型 + conformal prediction |
| Tang 2026 | DOI 10.1371/journal.pone.0347312 | PLOS 全文 | WGAN-GP 补全 + CNN-LSTM；去生成器 RMSE 恶化 39% |
| Aasi et al. 2026 | DOI 10.1016/j.eswa.2026.131582 | Crossref | per-regime scaling + 双注意力；C-MAPSS FD002 |
| TFT | DOI 10.1016/j.ijforecast.2021.03.012 | Crossref + arXiv:1912.09363 | static covariate encoders + variable selection + gating |
| iTransformer | arXiv:2310.06625 | arXiv abs 页 | variate token；注意力在 variate 之间 |
| LMF | arXiv:1806.00064 | arXiv abs 页（ACL 2018 见 comments） | 低秩张量融合 + modality-specific factors |
| TFN | arXiv:1707.07250 | arXiv abs 页（EMNLP 2017 见 comments） | full tensor 模态交互 |
| low-rank bottleneck | arXiv:2002.07028 | arXiv abs 页 | head 维度造成低秩瓶颈，限制表达力 |
| LoRA | arXiv:2106.09685 | arXiv abs 页 | 低秩分解增量 `BA`，残差加回 |
| BRITS | arXiv:1805.10572 | arXiv abs 页 | 双向 RNN 直接学缺失值 |
| LATC | arXiv:2006.10436 | arXiv abs 页 | sensor×time 低秩张量补全 |

**未核实项**：Garzón 2025 的 DOI（未找到）；RobustUQ 完整作者列表（只核到前两位）；iTransformer/LoRA/low-rank bottleneck/BRITS 的会议 DOI（arXiv 已核验，会议录用信息未在 Crossref 建 DOI）。

---

## 8. E1a/E1b 若复现，贡献定位

> 2026-11-16 增补。触发：确认阶段 E1a（mask p=0.10，clean Δ −0.545，seed sd 0.040）与 E1b（p=0.20，−0.689）在发现阶段过了预注册判据，经理问「这条能不能作为贡献」。
> **确认后定稿（2026-11-16）**：情形 A 成立——确认集 4/4 同号（E1a −0.3599 [−0.5779,−0.1491]、E1b −0.3418 [−0.6120,−0.0752]），多工况未出现 Wang et al. 报告的恶化；定稿措辞与边界见 **§10.3(a)**。

### 8.1 三个问题的直接回答

**Q1：有没有论文已经在 C-MAPSS / RUL 上报告过「训练期传感器通道 dropout 提升干净集精度」？**

**有，而且几乎同协议。** Wang, Q.; Liu, Z.; Jin, Z. 等，*Availability-Aware Remaining Useful Life Prediction for Aero-Engines with Unavailable Sensor Channels*，*Aerospace* 2026, 13(9):826，DOI 10.3390/aerospace13090826。Table 6（FD001，complete input，10 seeds 42–51）：DA-CNN-LSTM **16.403 ± 0.783** → 加 SD **15.663 ± 0.748**，**干净集 RMSE 降 0.740 周期**。同协议要素：14 传感器、整通道、训练期、每窗每通道独立 Bernoulli、至少留 1 路、125 cap、engine equal weighting。差别：L=30（我们 70）、CNN-LSTM（我们 DSPT）、干净集 2×2 只在 FD001 报。

另有 MSCCAN（Zhang, T.; Jiang, L.; Huang, R.; Zhang, X., *A Multiscale Cross-Channel Attention Network for RUL Prediction With Variable Sensors*, IEEE TIM 2025, DOI 10.1109/TIM.2025.3551011）用 variable-sensor augmentation + mask-aware aggregation，是同一族。

**Q2：有没有人报告过「dropout 提升的是干净精度而不只是鲁棒性」？**

**有，就是同一篇。** 他们明确把「SD 是否带来 complete-input penalty」作为一个问题问出来，答案是 FD001 上没有惩罚、反而降 0.74（§3.1）。更一般地，dropout 作为正则化提升泛化是 Srivastava et al. 2014（JMLR 15:1929–1958）的经典结论；masking 类增强（Cutout、SpecAugment）也以提升干净精度著称。所以「反直觉」在文献里并不反直觉——审稿人不会觉得新鲜。

**Q3：如果两者都有，我们唯一能主张什么？**

不能主张「现象」。能主张的是**复现 + 扩展**；扩展必须具体到 §8.3 的三条之一，否则只能写成复现。

### 8.2 一个必须先查的陷阱：逐子集方向

Wang et al. 的干净集增益**只发生在单工况子集**。Table 13（complete input）：

- FD001：16.403 ± 0.783 → +SD **15.663 ± 0.748**（变好 0.740）
- FD002：38.012 ± 11.077 → +SD **41.019 ± 11.107**（**变差 3.007**）
- FD004：32.848 ± 1.776 → +SD **38.793 ± 8.320**（**变差 5.945**）
- FD003：干净集未报

所以他们的结论是「单工况有增益、多工况有惩罚」。**在写 headline 之前，必须把 E1a/E1b 的逐子集 ΔRMSE 列出来**：

- 若我们是 FD001/FD003 改善、FD002/FD004 恶化 → 与 Wang et al. 同向，合并 −0.5 是 Simpson 悖论，只能写「单工况现象 + 多工况边界」，不能写「四子集稳定增益」。
- 若我们四个子集都改善（预注册要求 ≥3/4 同号，需要确认第 4 个是谁）→ 与 Wang et al. **相反**，这是唯一可能有点意思的差异，但必须先排除协议差异（L=70、DSPT 的 sensor attention/cycle、pooled 选轮），再谈机制。

**这条比任何 novelty 措辞都重要**：如果逐子集方向与 Wang et al. 相反而我们没解释，审稿人会用他们的 Table 13 直接质疑。

### 8.3 能写进 introduction 的一句话（取决于 8.2 的结果）

**情形 A（我们四个子集都改善，与 Wang et al. 相反）：**

> Training-time whole-channel sensor dropout, which reduces complete-input RMSE on FD001 with a CNN–LSTM backbone (Wang et al., 2026), also reduces clean-input engine-level RMSE on a strong Transformer across all four C-MAPSS subsets — including the multi-condition subsets where the CNN–LSTM study reported a penalty.

这句主张的是 **backbone/regime dependence**，是 empirical finding，不是 method。要成立，必须给出协议对照（同 L、同 cap、同 metric）排除「只是 L=70 或 backbone 不同」。

**情形 B（我们也是单工况改善、多工况恶化）：**

> We reproduce the FD001 complete-input gain of training-time sensor dropout on a strong Transformer, and show that it does not transfer to the multi-condition subsets, where the sign reverses.

这句是 boundary/negative result，可以写，但只能作为一节，**不足以单独支撑一篇 TIM**。

### 8.4 「剂量关系」和「强基线仍有效」能不能单独主张

- **剂量关系**：Wang et al. 只在缺失协议下扫 ps（Fig. 5b），没有在 clean 上报剂量。我们的 p=0.10→0.20 是 clean 上的一个增量，但**两个点不构成剂量关系**（至少要 3 个点 + 单调性检验）。可以写「p=0.20 的效应大于 p=0.10」，不能写「收益随屏蔽概率单调」。
- **强基线仍有效**：Wang et al. 的 Table 9 有 Transformer + SD（FD001 complete 14.245 ± 0.506），但**没有 no-SD Transformer 对照**，所以「SD 在 Transformer 上是否仍有效」在他们的论文里没有被隔离。这是唯一真实的缝，但它是「补一个对照」，不是「新机制」。
- 两条都只能作为 replication/extension 的一部分。

### 8.5 判定

- E1a/E1b 若在 split_seed=29 复现：可以写成 **「训练期传感器集增强在强 Transformer 上的复现与边界分析」**，属于 ENGINEERING-ONLY 的证据，**不是方法创新**。
- **不足以单独支撑一篇 IEEE TIM。** 它可以是：(i) 另一条主贡献下的一节；(ii) 与 E5 的机制分析合并成「传感器冗余/低秩」的故事——但 E5 本身也不新（见 §2）；(iii) 一篇以 boundary/negative result 为主的小文（TIM 不太收）。
- 如果本轮只剩 E1 能站住：**按 ENGINEERING-ONLY 处理**（不是 STOP——S-1 要求「没有任何臂达到 ENGINEERING-ONLY」），把 E1 写成复现，不要包装成贡献。

### 8.6 核验记录

| 文献 | 标识 | 核验方式 | 关键数字 |
|---|---|---|---|
| Wang et al. 2026 | DOI 10.3390/aerospace13090826 | MDPI 全文（Table 6、Table 13、§3.1、§3.4） | FD001 clean 16.403→15.663（−0.740，10 seeds）；FD002 clean 38.012→41.019；FD004 clean 32.848→38.793 |
| Zhang et al. 2025 (MSCCAN) | DOI 10.1109/TIM.2025.3551011 | Crossref | variable-sensor augmentation + mask-aware aggregation；IEEE TIM 2025 |
| Srivastava et al. 2014 | JMLR 15:1929–1958 | 经 Wang et al. 参考文献 [39]（未独立核 JMLR 页） | dropout 作为正则化提升泛化 |

**未核实**：FD003 的 clean SD 效应（Wang et al. 未报）；Cutout/SpecAugment 的具体编号与结论（未独立核，正文不引具体数字）。

---

## 9. E5 三种结局的论文定位（结果出来前写死）

> 2026-11-16 增补。E5 尚未起跑（48 格 0 格，两卡空闲），本节先写定位，避免结果出来后往有利方向解释。
> **结果后定稿（2026-11-16）**：E5 落在结局 ③，且不是「无增益」而是「变差」——`lr4` 合并 **+0.6714** [+0.2401, +1.1129]；按预注册 §4 步骤 0 触发，H1–H4 判 **N/A**（不是不成立）。定稿见 **§10.3(c)**。
> 依据：`E5_PREREG.md`（FROZEN v1.0 + v1.1 跑前预测）、本文 §2、E4 的实测（合并 −0.4098，FD001 是唯一变差 +0.3544）。

### 9.1 H1–H4 的可发表性审计（逐条）

| 假设 | 它实际检验什么 | 可证伪性 | 主要弱点 | 通过后能支撑什么 |
|---|---|---|---|---|
| **H1 低秩谱** | 学到的 `U` 是否对齐 `H` 的高方差子空间（ρ ≥ 0.60 且高于随机 95% 分位） | 中 | E0 上最优 rank-4 已达 **0.79–0.88**，随机 4 维只有 **0.41–0.43**；0.60 的门槛主要区分「支路学到东西」vs「支路接近 no-op」，**不区分「低秩假设成立」vs「数据本来就低秩」** | 只能说「`U` 不是随机切片」；不能单独说「低秩结构有效」 |
| **H2 非容量** | `lr4` 有效应、`mlp` 无效、`mlp` 收益 < `lr4` 的 50% | 低 | 「`mlp` CI 含 0」是**未检出**，不是**排除**；这是把「没证据」当「证据表明没有」。正确做法是报 `lr4 − mlp` 的直接配对 CI（见 9.5） | 若通过，是最有价值的一条：把低秩与容量分开。但写作时必须补直接配对对比 |
| **H3 基稳定** | 三个 seed 的 `U` 子空间最大主角度 < 30° | 高 | 稳定 ≠ 有意义：若 `U` 收敛到 top-4 PCA，H1 和 H3 会**同时**通过，但那是「模块重新发现了 PCA」，不是新结构 | 若通过且 `U` 与 PCA 有显著差异，才能说「学到了数据协方差之外的、与 RUL 相关的子空间」 |
| **H4 工况刻画** | `G(F)` 池化后能否线性解码工况簇（balanced acc ≥ 0.50，chance≈0.167） | 低 | **零分布选错**：打乱标签只能排除「偶然解码」，不能排除「`H` 本身就含工况信息」。`G(F)` 是 `H` 的函数，支路一旦有实际作用，工况信息很可能直接漏过去 | 通过只能说明「`G(F)` 保留了工况信息」，不能说明「支路刻画了工况」。必须补 `H` 池化 / 原始传感器作对照，只有增量才算 |

**结论**：四条里只有 **H2 和 H3** 有真正的判别力；H1 是必要条件、H4 的零分布需要补对照。**「H1–H4 多数成立」不能按字面用**——预注册 §4 是顺序门（H1→H2→H3→H4），前一条不成立则后面的不成立；写作时必须按顺序报，不能数「4 条里过了 3 条」。

### 9.2 效应量的校准：0.4 周期不再是证据

E4 的实测：加一个零初始化身份嵌入（+896 参数、0.77% 容量）就拿到合并 **−0.4098** [−0.7088, −0.1080]。这意味着**这个强基线本身有约 0.4 周期的「任意旋钮可及空间」**。

所以 E5 的 ΔRMSE 落在 −0.3 ~ −0.5 时，**与 E4 的 trivial knob 不可区分**；单靠效应量不能主张「低秩结构有效」。E5 要成为机制证据，必须同时满足：

- H2 通过（`lr4` 有效应、`mlp` 没有，**且 `lr4 − mlp` 的直接配对 CI 不含 0**）；
- H3 通过（`U` 跨 seed 稳定），**且** `U` 与 top-4 PCA 子空间有显著差异（见 9.5 第 1 条）；
- 效应量最好显著超过 0.4 周期，否则只能写「与 trivial knob 同量级」。

### 9.3 三种结局的定位

| 结局 | 判据（按 §4 顺序） | 能写 | 不能写 | 论文含义 |
|---|---|---|---|---|
| **① 有增益且机制成立** | `lr4` 有效应；H1/H2/H3 通过（H4 可选） | 「在强 Transformer 上，rank-4 传感器交互优于等参 MLP、与 full-rank 相当，且基跨 seed 稳定；增益不是容量」 | 「我们提出了低秩交互模块」（Xu 2026 §3.3 已覆盖）；「SOTA」；「因子是健康指标」 | **ENGINEERING-ONLY 的机制发现**。可作一节，或与 E1 的复现合并成「传感器冗余的两种利用方式」。**仍不足以单独撑 TIM**，除非 `U` 能映射到物理传感器组（预注册没要求，需额外做） |
| **② 有增益但机制不成立** | `lr4` 有效应；H1 或 H2 或 H3 失败 | 「增益存在，但不能归因于低秩结构」；按失败的那条写（容量/不稳定/谱不对齐） | 「低秩结构有效」；「因子可解释」；「机制成立」 | 与 E4 同类：**一个不可解释的旋钮**。只能作控制/负结果，不能作贡献 |
| **③ 无增益** | `lr4` vs `none` > −0.3，或 CI 含 0 | 「预注册的四臂设计未检出低秩交互的效应」；按 §8.2 报 ρ，区分「表示已低秩」vs「没学到」 | 「趋势性改善」；「某些子集有信号」；「需要更多 seed」 | **E5 是 null result**。与 E4 合并成「基线有约 0.4 周期可及空间，但传感器结构机制没有稳定、可归因的增益」——负结果/边界贡献，不是方法贡献 |

**结局 ③ 的两种写法必须分开**（预注册 §8.1 已写死预测，这里只提醒不要合并）：

- 若 `lr4` 的 ρ 接近可达上限（0.79–0.88）→ 写「表示本身已低秩，rank-4 瓶颈没有引入新的结构约束」；
- 若 ρ 显著低于可达上限 → 写「模块没能找到低秩结构」，不能说「低秩想法错了」。

### 9.4 四条全不成立时的明确写法

如果出现「有效应但 H1–H4 全部不成立」或「无效应」：

> **E5 不构成机制证据。** 低秩传感器交互在本协议下没有表现出可归因于「低秩结构」的效应；不得把 H1–H4 的任何单条通过当作机制成立的证据，也不得用「趋势」「部分子集」替代预注册判据。

**禁止的写法**（出现即视为过度解释）：

- 「低秩交互有效，但机制尚待解释」——机制不成立时不能说「有效」；
- 「H1/H3 通过说明低秩结构成立」——见 9.1，H1/H3 可被 PCA 退化同时满足；
- 「H4 通过说明刻画了工况」——零分布选错，需对照 `H`；
- 「合并 CI 不含 0 但逐子集不一致，说明有潜力」——预注册要求 ≥3/4 同号，不一致就是不成立；
- 把 `lr14` 的容量优势当作「低秩更好」的反证或正证——两者不可互替（预注册 §1）。

### 9.5 给军师的三个补充诊断（不改判据，只加报告项）

这三条不改变 §4 的判定，但决定「机制成立」能不能写：

1. **`U` vs top-4 PCA**：算 `U` 的列空间与 `H` 协方差前 4 个特征向量的最大主角度。若 < 10°，说明 `U` ≈ PCA，H1+H3 的通过是退化的，不能主张「学到了新结构」。
2. **`lr4 − mlp` 直接配对**：报 `ΔRMSE(lr4) − ΔRMSE(mlp)` 的 95% CI。H2 的「`mlp` CI 含 0」是未检出，不是排除；直接对比才是 H2 真正要的证据。
3. **H4 的 `H` 对照**：报 `G(F)` 与 `H` 池化特征的 balanced accuracy 之差。只有 `G(F)` 显著高于 `H`，才能说支路「刻画了工况」；否则只是把 `H` 里已有的工况信息带过去。

### 9.6 一句话

E5 的论文价值不在「低秩模块有没有用」，而在「能不能证明这个强基线的传感器表示已经是低秩的、低秩瓶颈因此没有新约束可加」。如果是后者，它是一个干净的负结果；如果是前者，它也只是 ENGINEERING-ONLY 的机制发现。两种都不是 IEEE TIM 的方法贡献。

---

## 10. 定稿（2026-11-16，全部结果落地后）

### 10.1 最终判定与三档对齐

| 候选 | 确认集结果 | 判定 | 触发条款 | 论文用途 |
|---|---|---|---|---|
| **E4 identity** | −0.4098 [−0.7088,−0.1080]，3/4，FD001 唯一变差 +0.3544 | **只作控制** | 已有开关 + Xu 2026 §2.2 式(6) 精确重叠 | 不进候选 |
| **E1a** mask p=0.10 | −0.3599 [−0.5779,−0.1491]，4/4 | **ENGINEERING-ONLY** | 确认集过线，但效应约为发现阶段一半、贴 −0.3 | 复现 + 边界 |
| **E1b** mask p=0.20 | −0.3418 [−0.6120,−0.0752]，4/4 | **ENGINEERING-ONLY** | 同上 | 复现 + 边界 |
| **E2a** noise α=0.25 | −0.1275 [−0.3713,+0.1199]，3/4 | **不复现** | CI 含 0 | 不作贡献 |
| **E2b** noise α=0.50 | 无增益 | **无增益** | — | 不作贡献 |
| **E5 lr4** | **+0.6714** [+0.2401,+1.1129]，变差 | **路线不成立** | 预注册 §4 步骤 0（Δ > −0.3 且为正） | 负结果 |
| **总档位** | — | **ENGINEERING-ONLY**（E1a/E1b） | §7.1：有臂达到 ENGINEERING-ONLY 且确认集复现，S-1 前件不成立 | 保留原 DSPT |
| 方法创新 / 低秩路线 | — | **STOP / 关闭** | E4/E5 不构成方法贡献；E5 步骤 0 触发 | 不写方法主张 |

**E5 的 H1–H4 必须写成 N/A，不能写成「不成立」。** 预注册 §4 是顺序门，步骤 0 触发后后面根本不进入判定；「不成立」会被误读为「测过并证伪」。这一点与牢莫、军师的更正一致。

### 10.2 哪些做法不构成新颖性（终稿清单）

1. **sensor id 嵌入（E4）**：Xu et al. 2026 §2.2 式(6) `z_{t,i} = u_{t,i} + δ_i` 精确同构；Garzón 2025、iTransformer、TFT 同类；本仓库 `sa_mode='identity'` 已是现成开关并跑过 12 格。零初始化身份嵌入 +896 参数买到 −0.41，与 dropout 同量级——是旋钮，不是机制。
2. **低秩残差传感器交互（E5）**：Xu et al. 2026 §3.3 的 prior-adjacency 低秩残差扰动（rank=8、λ=0.10、gate）同构；LMF、LoRA、low-rank bottleneck 各自独立覆盖「低秩 + 残差」的代数结构。实测还变差，连「有效」都不成立。
3. **条件感知低秩 / 工况门控**：Xu 2026 的 context-dependent 低秩因子 + Li 2026 的 condition modulation 已覆盖；CTNet、Garzón、Aasi、TFT 覆盖四种落点。
4. **训练期整通道 sensor dropout（E1a/E1b）**：Wang et al. 2026（*Aerospace* 13(9):826）已在 C-MAPSS FD001 报过 clean −0.740（10 seeds）；E1 是复现，不是新现象。
5. **训练期噪声注入（E2a/E2b）**：方法上不新；实测 E2a 不复现、E2b 无增益。
6. **等参 MLP / full-rank / no-branch 对照**：审稿人期待的对照，做了不加分，不做会被打。
7. **r 的敏感性扫描**：超参，不是机制。
8. **「在强基线上加模块」本身**：96 格消融 28 项对比未达区分门槛，base 已很强。

### 10.3 哪些是可发表的刻画（本轮唯一能写的）

必须把「可发表」限定为**刻画（characterization）**，不是方法贡献。三条：

**(a) 复现 + 协议差异（E1a/E1b）**

可写的一句话：

> Training-time whole-channel sensor dropout, previously shown to reduce complete-input RMSE by 0.74 cycles on FD001 with a CNN–LSTM backbone (Wang et al., 2026), reproduces on a strong Transformer baseline with a smaller but consistent effect (−0.36 cycles, 4/4 subsets), and the multi-condition penalty reported by that study does not appear under our protocol.

边界：合并效应 −0.36，约为 Wang et al. FD001 单子集值 0.74 的一半（协议不同，不能逐点比）；子集强弱排序在两个划分之间翻转，所以**不能写「哪些子集有效」**；只能写合并层面复现 + 多工况未观察到恶化。协议差异必须列全：L=70 vs 30、DSPT vs CNN-LSTM、5 seeds/split 29 vs 10 seeds、engine-mean-of-roots vs 其 complete-input RMSE。

**(b) 基线可及空间（E4 + E1 合并）**

可写的一句话：

> Two unrelated, near-zero-cost interventions — a zero-initialized sensor identity embedding (+896 parameters) and whole-channel sensor dropout — each move the same strong baseline by about 0.4 cycles, indicating an accessible space of that order that is not specific to either mechanism.

边界：这是 **n=2 的旁证**，不是受控估计；「0.4」是量级不是精确值；不能写成「基线有 0.4 周期的上限」。用途是解释「为什么这个量级的单模块增益不能当机制」，只能进讨论/归因段，不能进主结果。

**(c) 预注册四臂负结果（E5）**

可写的一句话：

> A pre-registered four-arm test (none / equal-parameter MLP / rank-4 / rank-14) finds that a rank-4 sensor-space interaction is worse than no branch (+0.67 cycles, 95% CI [0.24, 1.11]), so the low-rank hypothesis fails at the effect level; by the pre-registered sequential gate, H1–H4 are not tested (N/A).

边界：不能说「低秩在一般情况下无效」，只能说「在本协议、本骨干、本实现下无效」；H1–H4 是 N/A 不是被证伪；`rel` 三点排序（0.868 > 0.608 > 0.473）只能作假设（n=3），不能写成机制。

### 10.4 这三条刻画够不够一篇 TIM

**不够。** 逐条说：

- (a) 是复现 + 一个协议差异，不是新机制；TIM 的方法论文要「新方法 + 优于强基线」，我们没有。
- (b) 是两条旁证（n=2），不是受控实验；它解释现象，不提出方法。
- (c) 是负结果；TIM 不接受「我们试了没成」作为主贡献，除非它是某个更大主张的关键反证。
- 三条合起来能支撑的是一篇**协议 / 基线 / 机制归因**性质的论文（例如「强 Transformer RUL 基线上传感器增强与低秩交互的边界：一个预注册研究」），但贡献是「评测与归因」，不是「方法」；这需要重新定位投稿类型与期刊（TIM 的 scope 偏方法/仪器，负结果 + 评测的接受度低）。
- 如果一定要投 TIM：主贡献必须换成协议 / 基线 / 评测（统一协议下的强基线复现 + 传感器增强的边界刻画），并把本轮定位成 benchmark / characterization；E1/E4/E5 只能作为其中的节。

### 10.5 与 FINAL_RESEARCH_DECISION 的对齐

- 总档位：**ENGINEERING-ONLY**（E1a、E1b）。**S-1 未触发**：§7.1 的 STOP 条款是「没有任何臂达到 GO 或 ENGINEERING-ONLY；或确认集不复现」，而 E1a/E1b 达到了 ENGINEERING-ONLY 且确认集 4/4 复现。
- **方法创新：STOP**（E4/E5 均不构成方法贡献）。
- **低秩路线：关闭**（E5 步骤 0 触发，`lr4` 变差 +0.6714）；这是单条路线关闭，不是总档位的 STOP。
- 保留原 DSPT。
- E1a/E1b 的 ENGINEERING-ONLY 是「增强有效」的工程结论，不是方法贡献；可进工程报告/应用文，不进 TIM 方法主张。
- E4 只作控制；E5 路线关闭。
- 机制：没有一条站得住的解释（三条探针里一条方向相反、一条有训练分布混淆、一条只有 n=3；见 `SENSOR_MECHANISM_ANALYSIS.md`），不得用旁证顶。

### 10.6 一句话

本轮没有产生 IEEE TIM 意义上的方法贡献；可发表的是三条刻画（复现、可及空间、负结果），它们能支撑「为什么这类模块增益不能当机制」，但撑不起一篇方法论文。**总档位是 ENGINEERING-ONLY（E1a/E1b），不是 STOP；STOP 只落在「方法创新」与「低秩路线」这两支上。**

---

## 11. 官方 TEST 口径对照（开 TEST 之前写死）

> 2026-11-16 增补。触发：用户要求在官方 TEST 上看结果；经理要求先查清 Wang et al. 2026 与对照方法报的是什么口径，否则无法同口径对比。

### 11.1 结论先说

- **Wang et al. 2026 的 complete-input 数字是官方 TEST**：每台测试机末窗一个预测、pooled RMSE、125 cap、10 seeds（42–51）。不是验证集。
- **Xu et al. 2026 的 C-MAPSS 数字也是官方 TEST**：pooled RMSE / NASA Score / MAPE、单 seed（42）。
- 我们验证集的 `val_engine_mean_roots_rmse`（逐机 5 截断再平均）**在 TEST 上没有对应量**；验证集 Δ −0.36 不能直接和 TEST Δ 比。
- TEST 上同口径可比的只有 **pooled RMSE 的臂间 Δ**（E1a/E1b − E0），与 Wang 的 −0.740 同量纲；绝对 RMSE 不可比（骨干不同：他们 CNN-LSTM 16.4，我们 DSPT 验证 ~8.4–11.4）。

### 11.2 逐篇口径表

| 文献 | 评测集 | 口径 | seed | 关键数字（FD001） | 与我们的可比性 |
|---|---|---|---|---|---|
| **Wang et al. 2026** | **官方 TEST**（RUL_FD001 标签） | complete-input = 末窗一预测/机；pooled RMSE；125 cap；engine equal weighting | **10（42–51）** | 无 SD **16.403±0.783**；+SD **15.663±0.748**（Δ **−0.740**）；RUL-DARNet 15.350±1.170；Transformer+SD 14.245±0.506 | **同口径**：我们的 TEST pooled Δ 可直接对 −0.740；绝对数不可比 |
| **Wang et al. 2026（多工况）** | 同上 | 同上 | 10 | FD002 **38.012→41.019**（+3.007）；FD004 **32.848→38.793**（+5.945） | 同口径；这是「多工况未恶化」主张的直接对照 |
| **Xu et al. 2026（MST）** | **官方 TEST** | pooled RMSE / NASA Score / MAPE | **1（42）** | 完整模型 FD001 **11.79**（Table 4） | 口径可比、seed 少；**不是性能对照**，是新颖性前作 |
| CTNet / HPVSGNN / HDA-GNN / ADFG | 标准协议通常为官方 TEST | pooled RMSE | 未逐一核实 | **未在本轮核实具体数字** | 只作背景，**不拿数字做对照** |
| 仓库内上一轮 TEST | 官方 TEST | `test_metrics.json` | 混合 | 83 份：FD001 均值 **11.37**（35 份）、FD002 **11.13**（16）、FD003 **11.24**（16）、FD004 **12.41**（16） | **量级锚点，不是基线**；模型混合，不能当 E0 的 TEST 预测 |

### 11.3 开 TEST 前必须写死的三种解释

1. **E1a/E1b 的 TEST Δ 接近 0 或为正**：说明验证集的 −0.36 没有迁移到官方 TEST；「复现」只能限定在验证协议内，写进限制，不改档位。这与 Wang 的 −0.740 不矛盾——他们的基线是 16.4 的 CNN-LSTM，正则化对弱基线更受益；我们的基线验证 ~8.4–11.4，可及空间更小。
2. **TEST Δ 为负且 FD002/FD004 也为负**：与 Wang 的 FD002 +3.007 / FD004 +5.945 **相反**；这是「多工况未恶化」主张的直接证据，但仍只是复现 + 协议差异，不是新机制。
3. **TEST Δ 为负但 FD002/FD004 为正**：与 Wang 同向，说明「多工况恶化」在两个骨干上都出现；我们验证集的 4/4 是划分特性，必须收回「多工况未恶化」这句话。

### 11.4 不能做的比较

- 不能把我们的 TEST RMSE 和 Xu 的 11.79 比「谁更好」：单 seed、不同骨干、不同训练预算。
- 不能把验证集 Δ −0.36 和 TEST Δ 混在一张表里当同一个量。
- 不能用 TEST 结果重判 GO / ENGINEERING-ONLY / STOP（目标已结项；TEST 只作事后稳健性，写进 `FINAL_RESEARCH_DECISION.md` §9）。
- 不能把 Wang 的 −0.740 当成「我们应该复现的量」——他们的基线弱得多；同口径比的是**方向与量级**，不是数值相等。
- 不能拿 CTNet / HPVSGNN / HDA-GNN / ADFG 的绝对 RMSE 做对照——本轮没有核它们的 TEST 数字与 seed 数。

### 11.5 一句话

Wang et al. 2026 报的是官方 TEST、pooled RMSE、10 seeds、末窗一预测/机；我们的 TEST 报告必须用同一口径（pooled RMSE、Δ vs E0、125 cap），并且**在开跑前**把三种结果的解释写死，否则 TEST 一出来就会往有利方向读。
