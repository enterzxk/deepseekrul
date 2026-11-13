# 文献登记

> 只登记**已核实**来源（DOI / arXiv / 可访问链接）。禁止凭印象写条目。
> 现有笔记权威源：docs/literature_notes.md、docs/LITERATURE_BENCHMARK.md、docs/papers/。

## 登记表

| ID | 年份 | 出处 | 核心贡献（一句） | 与本研究差异 | 状态 |
|---|---|---|---|---|---|
| `piao2025crulp` | 2025 | IEEE TIM 74:1–11，DOI [10.1109/TIM.2024.3522678](https://doi.org/10.1109/TIM.2024.3522678) | DL RUL 预测器 + UACQR（ensemble CQR + cross-conformal），以**认知不确定性**构造区间 | 同刊直接竞争；我们用 split conformal + 学到的条件半宽，无 ensemble/cross-conformal；**未在同一协议下比较** | ✅ 三源核实（CrossRef/OpenAlex/S2），全文 closed |
| `robinson2026ijphm` | 2026 | IJPHM 17(1)，DOI [10.36001/ijphm.2026.v17i1.4724](https://doi.org/10.36001/ijphm.2026.v17i1.4724) | GBR 点预测 + CQR + 航空风险偏好，覆盖 C-MAPSS 全四子集 | 同为 CQR 应用；我们用端到端双分位；未比较 | ✅ CrossRef 核实，代码公开 |
| `rossellini2023uacqr` | 2023 | AISTATS；[arXiv:2306.08693](https://arxiv.org/abs/2306.08693) | 分离 aleatoric/epistemic，按特征空间区域差异化调整分位（UACQR） | 我们**没有**实现 UACQR；不得把 MC-Dropout 结果当作对它的评价（D019） | ✅ 摘要核实 |
| `yan2026audit` | 2026 | Zenodo 软件包，DOI [10.5281/zenodo.21330745](https://doi.org/10.5281/zenodo.21330745) | C-MAPSS FD001–FD004 + N-CMAPSS DS02 的保形区间**审计**：split / jackknife+ / ACI / regime-ACI / Mondrian（预测 RUL 带、产品分类）、5 个基础预测器、CPA 逐组条件覆盖、MILP 决策层 | **最近邻**。轴是 regime 与 RUL 带；我们的是**引擎（对象）级**；他们测条件覆盖并做补救，**不预测哪些对象会失效**（见 difftable §7） | ✅ 包内 methodology/phase 摘要已读；论文本体未见 |
| `yan2026ncmapss` | 2026 | Zenodo 软件包，DOI [10.5281/zenodo.21281080](https://doi.org/10.5281/zenodo.21281080) | N-CMAPSS 九个数据集、99 台发动机的保形校准基准；文稿已投 RESS | 数据集不同；方法学立场相近（审计 + 有效性范围） | ✅ 元数据核实，正文未读 |
| `similarity_cp` | 2026 | RESS，[S0951832026014365](https://www.sciencedirect.com/science/article/abs/pii/S0951832026014365) | 工况迁移下的无数据覆盖保证 + holds/violated/indeterminate 运行时判据 | 他们做迁移保证，我们做对象级失效识别；都涉及"逐对象判据" | ⚠ 仅摘要，需读正文 |
| `stratified_cp_srf` | 2026 | IEEE Xplore [11708000](https://ieeexplore.ieee.org/document/11708000) | 分层保形 + 传感器残差融合（SR-Conformal） | 分层补救路线；与我们已关闭的 Mondrian 方向重叠 | ⚠ 仅摘要 |
| `diao2026sensors` | 2026 | Sensors 26(7):2249，DOI [10.3390/s26072249](https://doi.org/10.3390/s26072249) | LSTM 分位回归 + 保形校准，含跨工况实验 | 方法族相同，评价口径不同 | ⚠ 仅摘要 |
| `onilede2026techrxiv` | 2026 | TechRxiv，DOI [10.36227/techrxiv.177220374.46388084/v1](https://doi.org/10.36227/techrxiv.177220374.46388084/v1) | 涡扇 RUL 的分布无关 UQ + CQR | 预印本，未同行评审 | ⚠ 仅摘要 |
| `liu2025stgatn` | 2025 | IEEE TIM 74，DOI [10.1109/TIM.2025.3550613](https://doi.org/10.1109/TIM.2025.3550613) | 气流图 + 机械结构图两条物理先验图，张量外积跨图融合，ConvLSTM 时间建模 | **点预测**，无区间/覆盖；真实数据实验是参数估计不是 RUL 终点预测 | ✅ 元数据核实（Scopus）；RMSE 表格为图形层，未引数值 |
| `jin2025hdagnn` | 2025 | IEEE TIM 74:3538014，DOI [10.1109/TIM.2025.3569885](https://doi.org/10.1109/TIM.2025.3569885) | MIC/MAS 异构图 + GRU-Attention 动态关系 + GCN + Transformer | **点预测**，无区间/覆盖；动态图权重未证明对应退化而非工况 | ✅ 元数据核实（NSTL/eurekamag）；RMSE 表格为图形层，未引数值 |
| `kim2025stkdformer` | 2025 | IEEE TIM，DOI [10.1109/TIM.2025.3625335](https://doi.org/10.1109/TIM.2025.3625335) | 时空 Koopman 自编码器（z_{t+1}≈Kz_t）+ 双分支 Transformer + 动态损失权重 | **点预测**，无区间/覆盖；隐空间无直接物理含义 | ✅ 元数据核实；RMSE 四数在 PDF 文本层可追溯 |
| `qu2026sfisthgat` | 2026 | IEEE TIM（Xplore 11732126），DOI ⚠ 待补 | 工况–传感器协同交互（非线性基 + 哈希 + FFT 域）+ 分层时空图（窗口内/窗口间） | **点预测**，无区间/覆盖；可解释性权重不等于因果 | ⚠ 缺 DOI；RMSE 四数在 PDF 文本层可追溯 |

## 最近邻差异矩阵（本轮已核实部分）

| 最近邻工作 | 它做了什么 | 我们没做什么（诚实） | 我们多做了什么 | 可被质疑点 |
|---|---|---|---|---|
| CRULP (TIM 2025) | ensemble CQR + cross-conformal，认知不确定性尺度 | 没有 ensemble、没有 cross-conformal、没有认知不确定性尺度 | 引擎级覆盖诊断；固定 α 的评分口径；跨域 FEMTO | "只是 CQR 的又一应用" |
| Robinson (IJPHM 2026) | GBR + CQR + 四子集 + 风险偏好 | 没有风险偏好/决策层 | 端到端双分位联合训练；引擎级失效清单 | 数据集相同、方法族相同 |
| Yan 审计包 (2026) | regime / RUL 带条件覆盖的测量 + ACI/regime-ACI/Mondrian 补救 + MILP | 没有 regime 分层、没有 ACI 补救、没有决策层 | **对象（引擎）级**失效与**可识别性**（用校准集信息预测哪些对象会失效） | 若他们已做对象级预警则新颖性不成立（待读论文本体） |


## 待读队列

- 从 docs/papers/ 现有 PDF 中筛出与"条件覆盖 / 局部化 conformal / 退化阶段异方差"直接相关的条目，
  逐条补登记表；优先级：能证伪本方法适用范围的工作 > 基线复现所需 > 综述。

## 检索与核实规范

1. 每条给可点击链接；无法访问的标"未核实"，不得进入 Related Works。
2. 引用前核对：作者、年份、 venue、主张是否被原文支持（不得二手转述）。
3. 新颖性检查：至少覆盖 conformal / CQR / 局部化 conformal / RUL 不确定性 四条检索线。

## 2026-11-09 新增（D021 立项草案 v2 的文献基础）

> 核实方式：OpenAlex Works API DOI 直查 + 出版方页面 + 独立子代理对抗式通读。
> 等级：A=读过全文/包内文档；B=读过摘要；C=仅元数据（不得引用结论）。

| ID | 年份 | 出处 | 核心贡献（一句） | 对本立项的威胁 | 等级 |
|---|---|---|---|---|---|
| `driftguard2026` | 2026 | Research Square 预印本（评审中），DOI [10.21203/rs.3.rs-10396495/v1](https://doi.org/10.21203/rs.3.rs-10396495/v1) | 明确问「带标签源 + 无标签目标协变量能否判定覆盖可信」；给出**精确/近似可检测界、minimax 检验推论、条件漂移下目标覆盖的 [0,1] 锐识别集**，再用目标标签预算跨越边界 | **HIGH**：label-free 覆盖审计的**边界类型**已被定理级占住 | B |
| `hacc2026` | 2026 | Machine Learning with Applications，DOI [10.1016/j.mlwa.2026.101038](https://doi.org/10.1016/j.mlwa.2026.101038) | 稀疏 (group,horizon) 单元的经验分位数**不存在**（$n<(1-\alpha)/\alpha$，90% 时 9 点）；log-additive 收缩把 min-group 覆盖 **0.28–0.31 → 0.51–0.72** | **HIGH**：算术阈值 + 收缩修复都被占；「6<9 是常识」是最强反驳 | B |
| `domingues2026` | 2026 | JAIMLNN 6(2):106–119，DOI [10.55529/jaimlnn.62.106.119](https://doi.org/10.55529/jaimlnn.62.106.119) | C-MAPSS 上边际 89–92%，真实 RUL 61–90 周期覆盖 68–76%，寿命末端工况漂移下 37–50%；FD001→FD003 迁移 | **HIGH（现象）**：C-MAPSS 条件覆盖崩溃已发表 | B |
| `zhou2026cpa` | 2026 | [arXiv:2603.27189](https://arxiv.org/abs/2603.27189) | 条件覆盖评估 = **有监督**任务：isotonic 校准的 reliability estimator 预测实例级覆盖；CVI 指标；建议 ≥1000 评估样本 | **MEDIUM**：预测任务类型被占（但需标签） | B |

**对 D021 草案的净影响**：第一版「识别边界」框架作废（见 `failures.md` F001）。
第二版收窄为「协议算术下界 + 后果量化 + ICC 可复现性 + 文献审计」的测量学审计，
并正面回答 HACC 反驳（草案 §2.3）。


- N-CMAPSS已核BayesRUL/AIDGN/MOO_ELM/Concept等，完整DOI、固定SHA与协议风险见 docs/paper_b_ncmapss_response_20261113/NEW_DATASET_BASELINE_AUDIT.md。
- 最近邻R2-D2、Function Encoders、operation-profile DA、physics/deep/PF：均有原始来源；不主张可微ridge本身创新。
