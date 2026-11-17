# LITERATURE_MAP — 传感器建模最近邻文献地图（E4 / E5）

> study: `paper_b_sensor_generalization` · 分支 `research/paper-b-sensor-generalization`
> 版本：v0.1 初稿（2026-11-15）
> 核验方式：arXiv abs 页、Crossref `api.crossref.org`、期刊原文页（IOP / ScienceDirect / PLOS / IEEE Xplore）、Semantic Scholar 题录。每条给出 DOI 或 arXiv 编号；无 DOI 的会议短文给 Semantic Scholar CorpusID。
> 阅读深度逐条标注：**全文** / **摘要+引言** / **仅摘要**。只写核验过的内容；查不到的写「查不到」，不用记忆补。
> 本文件服务于 `SENSOR_RESEARCH_PLAN.md` §10「E4/E5 必须先过文献员的新颖性判断」；判定见 `NOVELTY_VERDICT.md`。

---

## 0. 判断基准：我们这两个机制到底是什么

先把被判断的对象钉死，避免和文献里的近似说法混淆。

**E4 Sensor Identity（代码事实，不是设想）**

`paper_b/models/dspt.py` 里已经存在这个开关：

- `:147-148`：`if sa_mode in ('identity', 'identity_cond'): self.sensor_id = nn.Parameter(torch.zeros(1, num_sensors, d_model))`
- `:240-243`：跨传感器注意力之前执行 `st = st + self.sensor_id`，其中 `st` 的形状是 `(B·L, 14, D)`。
- `:257-258`：随后 `sensor_score` 对 14 个传感器做 softmax 池化。

也就是说，E4 的「在注意力前注入 sensor id」在本仓库里就是 `sa_mode='identity'`，不是待实现的新模块。另据 `docs/paper_b_baselines/RUN_INVENTORY.md:69`，`SAM_identity_L70` 与 `SAM_cond_L70` 各 **12/12 已跑完**（split_seed=7），且「未进冻结清单」。

**E5 Low-Rank Sensor Interaction（任务书定义）**

`H: [B, T, 14, D]`；`U ∈ R^{14×r}`；`F = UᵀH`（14→r）；`ΔH = U·G(F)`；`H_new = H + λ·ΔH`；`r=4`；`λ` 小尺度初始化；保留原表示残差支路；先作为完整 sensor attention 的**替代**而非叠加。

两点结构特征决定了它该和谁比：**(a)** 低秩作用在**传感器维（14）**，时间与隐维不动；**(b)** 形态是「低秩投影 → 非线性 → 回投 → 小尺度残差」。

---

## 1. 线 A：低秩传感器表示 / 传感器交互的隐变量建模

| # | 文献 | 年份 / venue | 标识 | 做了什么 | 数据 | 与 E5 的差异 | 深度 |
|---|---|---|---|---|---|---|---|
| A1 | Xu, Z.; Bashir, M.; Zhang, Q.; Beer, M.; Wang, J.; Yang, Z. *Reliable prognosis for complex systems based on spatial-temporal representation modeling and uncertainty-aware dynamic Fusion* | 2026, *Meas. Sci. Technol.* 37 245102 | DOI 10.1088/1361-6501/ae7463 | 逐传感器可学习嵌入 `δ_i` 加到进入 GLSTM 前的特征（式 6）；跨传感器注意力（64,064 参数）；把**图变化**建模为 prior adjacency 周围的**低秩残差扰动** `A_dyn = normalize(A_prior + λ·sym(a_t b_tᵀ))`，rank=8、λ=0.10、gate-on-update；不确定性加权融合 | C-MAPSS FD001–FD004，14 传感器，d=64 | **E5 的同代数结构**：低秩因子 + 回投 + 小尺度 λ + 保留原表示。差别只在作用对象（邻接矩阵 vs 特征张量） | 全文 |
| A2 | Liu, Z. 等（arXiv 显示 6 位作者）*Efficient Low-rank Multimodal Fusion with Modality-Specific Factors* | 2018, ACL | arXiv:1806.00064 | 用低秩张量分解做多模态融合，每个模态有**自己的因子矩阵**，把模态间交互写成低秩因子组合，复杂度从指数降到线性 | 多模态情感/说话人/情绪三个任务 | 与 E5 的 `U`（传感器基）和 `G`（低维变换）同构；没有「残差加回」，且模态数远少于 14 个传感器 | 摘要 |
| A3 | Zadeh, A. 等（arXiv 显示 5 位作者）*Tensor Fusion Network for Multimodal Sentiment Analysis* | 2017, EMNLP | arXiv:1707.07250 | 用外积构造模态间 full tensor，显式建模 intra/inter-modality 交互 | 多模态情感 | 是 A2 与 E5 要压缩的那个 full-rank 对象；说明「低秩」的动机是算力，不是新交互 | 摘要 |
| A4 | Bhojanapalli, S. 等（arXiv 显示 5 位作者）*Low-Rank Bottleneck in Multi-head Attention Models* | 2020, ICML | arXiv:2002.07028 | 证明 head 维度小于 embedding 维度会形成**低秩瓶颈**、限制注意力表达力；提出把 head size 提到序列长度 | 语言建模 | 说明「低秩」首先是一个 capacity 问题：E5 把 r 压到 4 是在传感器维制造同样的瓶颈，必须排除欠拟合解释 | 摘要 |
| A5 | Hu, E. J. 等（arXiv 显示 8 位作者）*LoRA: Low-Rank Adaptation of Large Language Models* | 2022, ICLR | arXiv:2106.09685 | 冻结原权重，注入 rank decomposition 的 `BA` 低秩增量，`W + BA`，无额外推理延迟 | RoBERTa/DeBERTa/GPT-2/3 | 与 E5 的「低秩 + 残差加回 + 小尺度初始化」同构；差别只是作用在权重矩阵而非传感器维 | 摘要 |

延伸（不占名额）：Chen, X.; Sun, L. *Low-Rank Autoregressive Tensor Completion for Multivariate Time Series Forecasting*, arXiv:2006.10436（摘要）——把 sensor×time 矩阵升为三阶张量做低秩补全，说明「传感器维低秩」在多变量时间序列里是成熟假设，E5 的 `U` 是该假设的线性版。

**这条线的结论**：「低秩因子 + 回投 + 小尺度残差」已经在**三个互不相干的文献**里各自成立（张量融合 A2/A3、注意力容量 A4、参数高效微调 A5），并且已经在 **RUL 的传感器图上**被完整用过一次（A1）。E5 若成立，只能是「换了个被低秩分解的对象」，不是新机制。

---

## 2. 线 B：动态图 RUL 预测

| # | 文献 | 年份 / venue | 标识 | 做了什么 | 数据 | 与我们的差异 | 深度 |
|---|---|---|---|---|---|---|---|
| B1 | Xu, Z. 等（同 A1）*Reliable prognosis…* | 2026, *Meas. Sci. Technol.* | DOI 10.1088/1361-6501/ae7463 | 传感器为节点、GLSTM 为骨干；图不是每步重学，而是用 Bayesian event-triggered 机制在「证据足够」时更新；低秩残差扰动生成候选图 | C-MAPSS + 传感器攻击场景 | 我们的 E5 **不建图**；差异是放弃了图先验这条归纳偏置 | 全文 |
| B2 | Li, J.; Sun, Y.; Wang, H.; Li, Y. *A dynamic spatio-temporal graph neural network adaptive to operating conditions: Learning evolutionary correlations in multi-sensor data for aero-engine* | 2026, *Aerosp. Sci. Technol.* 176:112160 | DOI 10.1016/j.ast.2026.112160 | 「condition modulation trend」把工况当调制信号构造动态图；变分信息瓶颈图滤波把节点建成高斯分布、按方差抑制高噪声节点；时频因果解耦 | C-MAPSS + QAR 工程数据 | 我们的 stage-3「条件感知低秩」正是 B2 的 condition modulation 与 A1 的 context-dependent low-rank 的组合 | 摘要+引言 |
| B3 | Jin, Y.; Meng, Z.; Hu, Q.; Chen, D. *Heterogeneous Dynamic-Aware GNN for RUL Prediction of Aeroengine* | 2025, *IEEE TIM* 74:3538014 | DOI 10.1109/TIM.2025.3569885 | 把退化刻画为「多变量之间约束关系的变化」，用异构动态感知的 graph flow 学习建模动态交互 | 航空发动机 RUL | 同属「传感器关系随时间变」的建模范式；E5 的共享 `U` 是**时不变**的，反而更弱 | 仅摘要 |
| B4 | Cao, X.; Cheng, B.; Yang, X.; Duan, Y.; Gao, J. *Remaining Useful Life Prediction Method for Turbofan Engines Based on Graph Neural Networks* | 2025, *IEEE Sensors J.* 25(8):13566–13578 | DOI 10.1109/JSEN.2025.3546498 | MSCNN 提节点特征；动态图构造实时捕捉传感器关系；自适应融合「固定先验图 + 动态相似图」；GAT + 注意力 + BiGRU | C-MAPSS | 与 B1 同思路（先验 + 动态），说明这条路线已进入常规工程化 | 仅摘要 |

**这条线的结论**：传感器动态图在 RUL 里已经是拥挤赛道，区分点集中在**图怎么来**（先验+动态 B4、事件触发 B1、工况调制 B2、异构流 B3）。E5 完全不建图，这既是它与 B1–B4 的真实差异，也意味着它放弃了这个赛道里被反复验证有效的归纳偏置——审稿人不会把这当成优势。

---

## 3. 线 C：条件（工况）感知的注意力或门控

| # | 文献 | 年份 / venue | 标识 | 做了什么 | 数据 | 与我们的差异 | 深度 |
|---|---|---|---|---|---|---|---|
| C1 | Liu, H.; Sun, Y.; Wang, H.; Zhang, H. *CTNet: Improving the non-stationary predictive ability of remaining useful life of aero-engine under multiple time-varying operating conditions* | 2025, *Measurement* | DOI 10.1016/j.measurement.2024.116345 | 在**工况描述符与传感器节点之间建图**，用 causality attention 把因果信息注入 self-attention，且不改变窗口内 Q/K 相似度计算 | C-MAPSS、N-CMAPSS、QAR | 我们只把窗口末端工况簇当 embedding 加进传感器注意力（`cond_emb`），没有「工况—传感器」因果结构 | 摘要+引言 |
| C2 | Garzón, J.; Flores-Moyano, R.; Baldeon-Calisto, M.; Vega-Sánchez, J. D. *A Transformer Framework for Remaining Useful Life Prediction Using Sensor Attention and Operational Context Embeddings* | 2025, *IEEE ETCM* 2025, pp.1–6 | S2 CorpusID 284229047（未核到 DOI） | 在 C-MAPSS 上把 **sensor attention + operational context embeddings** 放进同一个 Transformer 框架 | C-MAPSS | **E4 与工况嵌入的组合已被这篇占住**；我们只做 id、不做 context | 仅摘要 |
| C3 | Aasi, A.; Ebrahimi Bajgani, S.; Mahdavi, O. *Condition-Aware AI for Predictive Maintenance: Dual-Attention CNN-GRU with Per-Regime Scaling* | 2026, *Expert Syst. Appl.* | DOI 10.1016/j.eswa.2026.131582 | 按工况 regime 做 per-regime scaling；channel attention 给传感器退化特征加权；sequence attention 定位退化阶段 | C-MAPSS FD002 | 条件信息用于**归一化/缩放**而非低秩交互；说明「按工况调传感器权重」已是常规操作 | 仅摘要 |
| C4 | Lim, B.; Arık, S. Ö.; Loeff, N.; Pfister, T. *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting* | 2021, *Int. J. Forecast.* | DOI 10.1016/j.ijforecast.2021.03.012 | static covariate encoders 处理时不变身份/上下文；variable selection + 一系列 gating 抑制无用分量 | 多领域预测 | E4 的 per-sensor id 与 TFT 的 static covariate 编码是同一类归纳偏置；TFT 还多一层变量选择 | 摘要 |

**这条线的结论**：条件感知在 RUL 里有四种成熟落点——图（C1）、context embedding（C2）、per-regime scaling（C3）、static/gating（C4）。任务书里的 stage-3「Condition-Adaptive Low-Rank Sensor Interaction」不是新组合，而是 **A1 的 context-dependent low-rank factors + B2 的 condition modulation** 的直接合体。

---

## 4. 线 D：传感器缺失与噪声鲁棒性

| # | 文献 | 年份 / venue | 标识 | 做了什么 | 数据 | 与我们的差异 | 深度 |
|---|---|---|---|---|---|---|---|
| **D0** | **Wang, Q.; Liu, Z.; Jin, Z. 等** *Availability-Aware Remaining Useful Life Prediction for Aero-Engines with Unavailable Sensor Channels* | **2026, *Aerospace* 13(9):826** | **DOI 10.3390/aerospace13090826** | RUL-DARNet：CNN-LSTM 骨干 + **训练期整通道 Sensor Dropout（SD，ps=0.2，至少留 1 路）** + Mask-Aware 特征注意力 + 时间注意力；10 seeds × 10 配对 mask，C-MAPSS 四子集，**官方 TEST 末窗一预测/机**、pooled RMSE、125 cap、engine equal weighting | C-MAPSS FD001–FD004 | **这是 E1 的直接前作**：同协议要素几乎逐项对应（14 传感器、整通道、训练期、每窗 Bernoulli、至少留 1 路、125 cap、engine equal）。差别：L=30 vs 70；CNN-LSTM vs DSPT；clean 2×2 只在 FD001 报。**Table 6：FD001 clean 16.403→15.663（−0.740）**；**Table 13：FD002 clean 38.012→41.019、FD004 clean 32.848→38.793（多工况变差）** | 全文 |
| D1 | Wang, W.; Wang, Z. 等（作者列表未完整核实）*Robust uncertainty quantification for online remaining useful life prediction with randomly missing and partially faulty sensor data* | 2025, *Reliab. Eng. Syst. Saf.* 262:111177 | DOI 10.1016/j.ress.2025.111177 | 用扩散模型拟合随机缺失/部分故障数据的分布，结合深度度量学习与 conformal prediction 做在线 RUL 不确定性量化 | 设备 RUL（含在线场景） | 属于「先补后预测 + UQ」范式；我们的 E1/E2 是训练期增强，不做显式补全 | 仅摘要 |
| D2 | Tang, D. *A deep learning framework for remaining useful life prediction of turbofan engines with partial sensor failure* | 2026, *PLoS ONE* 21(4):e0347312 | DOI 10.1371/journal.pone.0347312 | WGAN-GP 缺失参数生成器补全受损传感器序列，CNN-LSTM 预测 RUL，对抗损失与回归损失联合优化；去掉生成器 RMSE 恶化 39% | C-MAPSS，缺失率扫描 | 端到端「修复—预测」联合训练；我们的 mask 增强不重建数据，只让模型见过缺失 | 全文 |
| D3 | Li, J. 等（同 B2）*HPVSGNN* | 2026, *Aerosp. Sci. Technol.* | DOI 10.1016/j.ast.2026.112160 | 变分信息瓶颈图滤波按方差抑制高噪声节点，直接面向数据碎片化与高噪声 | C-MAPSS + QAR | 架构内鲁棒；我们靠训练期扰动 | 摘要+引言 |
| D4 | Xu, Z. 等（同 A1） | 2026, *Meas. Sci. Technol.* | DOI 10.1088/1361-6501/ae7463 | 传感器攻击场景下用不确定性加权融合抑制被污染传感器的贡献 | C-MAPSS + 攻击注入 | 推理期加权；我们是训练期增强 | 全文 |
| D5 | Cao, W. 等（arXiv 显示 6 位作者）*BRITS: Bidirectional Recurrent Imputation for Time Series* | 2018, NeurIPS | arXiv:1805.10572 | 双向 RNN 直接在动态系统里学缺失值，不预设数据生成过程 | 空气质量/医疗/定位 | 缺失值建模的方法学祖先；与 RUL 无直接关系 | 摘要 |

延伸（不占名额）：Xiao, L.; Tang, J.; Zhang, X.; Bechhoefer, E.; Ding, S. *Remaining useful life prediction based on intentional noise injection and feature reconstruction*, *Reliab. Eng. Syst. Saf.* 2021, DOI 10.1016/j.ress.2021.107871（摘要+原文页）——把高斯白噪声**注入 LSTM 输出层/目标**（不是输入传感器），在 C-MAPSS 上声称噪声注入不降反升。它与 E2 的「输入噪声」不是同一机制，但说明「有意加噪提升 RUL」这个现象已有先例。

**这条线的结论**：缺失/噪声鲁棒有两条成熟范式——**先补后预测**（D1/D2/D5）与**架构内鲁棒**（D3/D4）。我们的 E1/E2 是第三条：训练期增强 + 显式 availability mask。**D0 已经把 E1 的同款做法在 C-MAPSS 上做掉了**（整通道 SD，FD001 clean −0.740，10 seeds），所以 E1 是复现而不是新现象；旧 study 的 36 格（clean 全负）说明这个效应还高度依赖协议。E2 同理，不承担新颖性。详见 `NOVELTY_VERDICT.md` §8。

---

## 5. E4 最近邻清单（按重叠度排序）

| 排名 | 文献 | 重叠的具体部分 | 差异 | 差异是否构成贡献 |
|---|---|---|---|---|
| 1 | **A1 / Xu et al. 2026, MST** | §2.2 式 (6)：`z_{t,i} = u_{t,i} + δ_i`，`δ_i` 是逐传感器可学习向量，加在进入跨传感器模块之前；随后接 sensor attention | 注入点在他们那里是 GLSTM 前，在我们这里是 `sensor_attn` 前；他们还同时注入工况上下文 | **否**。同一行代数、同一目的（让模型区分传感器身份），只是模块位置不同 |
| 2 | **C2 / Garzón et al. 2025, ETCM** | C-MAPSS 上的 sensor attention + operational context embeddings | 我们只有 sensor id，没有工况 context | **否**。他们已把「传感器身份 + 工况上下文」一起做掉；我们做的是它的子集 |
| 3 | **Liu, Y. 等 2024, iTransformer（arXiv:2310.06625）** | 把每个 variate 的时间点嵌入成 **variate token**，注意力在 variate 之间跑；token 的身份由序列位置承载 | 我们把身份显式化为可学习向量 | **否**。显式化身份是 tokenization 的等价改写，不是新机制 |
| 4 | **C4 / Lim et al. 2021, TFT** | static covariate encoders 处理时不变的身份/上下文；gating 抑制无用分量 | 我们只加一个 per-sensor 向量 | **否**。TFT 的做法更完整 |
| 5 | **本仓库 `paper_b/models/dspt.py:147-148, 240-243`** | `sa_mode='identity'` 就是 `st = st + self.sensor_id` | 无 | **否**，而且它已经存在；`SAM_identity_L70` 12/12 已跑（`RUN_INVENTORY.md:69`） |

**E4 判定**：不构成新颖性。四篇外部文献 + 一条内部代码事实，指向同一结论：这是多传感器融合与多变量时间序列里的标准做法。

---

## 6. E5 最近邻清单（按重叠度排序）

| 排名 | 文献 | 重叠的具体部分 | 差异 | 差异是否构成贡献 |
|---|---|---|---|---|
| 1 | **A1 / Xu et al. 2026, MST** | §3.3：把图变化建模为 prior adjacency 周围的**低秩残差扰动**，`A_dyn = normalize(A_prior + λ·sym(a_t b_tᵀ))`；rank=8、λ=0.10、gate-on-update（Table 6） | (i) 作用对象是**邻接矩阵**而非特征张量 `H`；(ii) 低维变换 `G` 是双线性因子而非 MLP；(iii) 我们**替代** sensor attention，他们**叠加**在 GLSTM 上；(iv) r=4 vs 8 | **否（四项相加仍不足）**。(i) 是位置差异且丢掉图先验；(ii) 在 A2/A5 有先例；(iii) 是架构选择；(iv) 是超参 |
| 2 | **A2 / LMF 2018, ACL** | 每个模态一个低秩因子矩阵，交互在低维因子空间完成 | 没有「残差加回」；模态数少 | **否**。E5 的 `U` 与 LMF 的 modality-specific factor 是同一角色 |
| 3 | **A5 / LoRA 2022, ICLR** | `W + BA`：低秩增量 + 残差加回 + 小尺度初始化 | 作用在权重矩阵，不是传感器维 | **否**。残差低秩更新的范式已经固定 |
| 4 | **A4 / Bhojanapalli 2020, ICML** | 低秩瓶颈会限制表达力 | 它是分析而非模块 | **否**，但它给出 E5 必须排除的替代解释：r=4 的增益可能只是容量变化 |
| 5 | **A3 / TFN 2017, EMNLP** | full tensor 交互 | 是 E5 要压缩的对象 | **否**。低秩是它的近似，不是新交互 |

**E5 判定**：机制新颖性不足。**我没有检索到**「在 C-MAPSS 的传感器特征维（而非图/邻接）上做低秩投影—非线性—回投—残差」的直接前作——这是唯一的空白，但它是**作用位置**的空白，不是**机制**的空白。按 IEEE TIM 的方法贡献标准，位置差异不足以单独支撑一篇论文。

---

## 7. 明确不构成新颖性的做法（可直接引用进 rebuttal 准备）

1. **把 sensor id 拼进输入 / 加到表示上**：A1 式 (6) 已给出逐传感器可学习向量；iTransformer 的 variate token、TFT 的 static covariate 是同一归纳偏置的另外两种写法；本仓库 `sa_mode='identity'` 已实现并跑过 12 格。这是常识，不是贡献。
2. **低秩投影 + 非线性 + 回投 + 残差**：A2（低秩因子融合）、A5（低秩残差更新）、A4（低秩瓶颈）各自独立成立；A1 已把同一代数结构用在传感器图上。换一个被分解的对象不构成新机制。
3. **条件 embedding / 工况门控**：C1（工况—传感器因果图）、C2（operational context embeddings）、C3（per-regime scaling）、C4（static/gating）、A1（context-dependent 低秩因子）都已覆盖。
4. **训练期 mask/dropout + 显式 availability mask**：缺失数据文献（D1/D2/D5）与旧 study 的 36 格已经把这条路走完；clean 通道全负。
5. **等参 MLP / full-rank / no-branch 对照**：这是审稿人**期待**的对照（任务书也要求），做了不加分，不做会被打。
6. **r=2/4/8 的敏感性扫描**：超参扫描不是机制贡献；只有「rank 谱对应可解释的传感器结构」才可能变成发现。

---

## 8. 检索缺口与未核实项（诚实登记）

- **IEEE Xplore 页面不返回作者列表**：HDA-GNN 的作者经两个二手题录（ablesci、eurekamag）核对为 Jin, Meng, Hu, Chen；ADFG 的作者经 Crossref 核对为 Cao, Cheng, Yang, Duan, Gao。两者 DOI 均经 Crossref/IEEE 页确认。
- **Garzón et al. 2025（ETCM）未核到 DOI**，只有 Semantic Scholar CorpusID 284229047 与会议名/页码；引用时建议只写会议与页码。
- **RobustUQ 的作者列表**只核到前两位（Wang, W.; Wang, Z.），其余未核实。
- **iTransformer / LoRA / low-rank bottleneck / BRITS 的会议录用**未在 Crossref 建 DOI（arXiv 版已核验）；正文引用建议写「arXiv:xxxx + 会议名」，不要编造 DOI。
- **SADeepSense（INFOCOM 2019, DOI 10.1109/INFOCOM.2019.8737500）** 摘要只说「用自注意力学传感器间相关性与贡献度」，没有说使用 sensor embedding，故**不列入** E4 近邻，避免夸大。
- **检索到的 2026 年文献多为 online-first**，卷期/页码可能随后变化；本文件以核验当日的页面为准。
- 未检索到「feature-space 低秩传感器交互」在 C-MAPSS 上的直接前作（见 §6 判定）；这是检索结论，不是「不存在」的证明。

---

## 9. 对四条线的方法学含义（一段话）

低秩这条线（A）在**机制**上已经饱和、在**作用位置**上还有缝；动态图（B）与条件感知（C）都已进入「组合已有模块 + 换图构造方式」的阶段；缺失/噪声（D）的三条范式都已有人做，且我们的训练期增强在旧 study 里没有站住。因此 E4 没有可主张的增量，E5 的增量只能落在「feature-space vs graph-space」这一条缝上——它是否值得写，取决于实验能否给出**机制证据**（rank 谱、因子稳定性、跨工况/缺失下的行为），而不是「比 baseline 好一点」。详见 `NOVELTY_VERDICT.md`。
