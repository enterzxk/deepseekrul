# 校准层新颖性审计（CALIBRATION_NOVELTY_AUDIT）

审查对象：66 参数事后 RUL 概率校准层（location-scale + 离散 Student-t(ν=5) + 右删失 NLL + 覆盖率目标尺度乘子）
审查标准：只依据可核查文献；结论用于判断能否在 IEEE TIM 宣称"方法创新"。
审查日期：2026-11-01

---

## 1. 结论先行

**无方法层增量，仅应用层组合，外加一个此前 C-MAPSS 文献未明确做过的细节校正（把 125 封顶当右删失）。**

一句话理由：`μ(ŷ)、σ(ŷ) + 固定形状 + 标准化残差 (y−μ)/σ 的经验/参数化重标定` 是 Kuleshov et al. 2018 §3.4 明确写出的做法，CRUDE（Zelikman et al., arXiv:2005.12496）把它做成完整方法并与 inductive conformal 证成等价；`位置-尺度 + 右删失似然` 是 Tobit（Tobin 1958）/crch（Messner et al. 2016）的标准形式；`σ 乘子` 是 Levi et al. 2022 的 STD scaling；`交叉拟合` 是 cross-conformal / CV+。唯一未被现成覆盖的是"C-MAPSS 的 125 封顶按删失而非点质量处理"，但它本质是一次 Tobit 重参数化，且引入了一个新的估计量—评价指标错配（见 §4.3）。

可发表定位：**测量与协议贡献（measurement / protocol），不是方法贡献。** 若论文标题或摘要出现"new calibration method / novel probabilistic framework"，审稿人一击致命。

---

## 2. 逐项对照表

"重合"= 我们的组件在该文献中已存在；"不能声称"= 该文献已占据、我们再声称即构成重复。

| 方法 | 分布形式 | 是否处理删失 | 校准机制 | 保证类型 | 与我们重合的部分 | 我们不能声称的部分 |
|---|---|---|---|---|---|---|
| **Nix & Weigend 1994**（MVE） | 高斯 N(μ(x), σ²(x)) | 否 | 直接以 NLL 联合训练 μ、σ | 无（MLE 一致性） | location-scale 异方差、softplus/exp 保证 σ>0、NLL 训练 | "用神经网络同时输出位置与尺度"、"σ 依赖输入" |
| **Kuleshov et al. 2018**（PIT 等渗） | 任意（对 CDF 做单调重标定） | 否 | 在 calibration set 上对 PIT 值拟合单调映射 R:[0,1]→[0,1]（isotonic） | 渐近（"given enough i.i.d. data"），非有限样本 | **§3.4 明确写：用 heteroscedastic 的 (y−μ(x))/σ(x) 作为重标定特征**；交叉拟合（§3.2 建议 K-fold） | "用标准化残差做校准变量"、"事后不重训骨干"、"K 折交叉拟合校准" |
| **Levi et al. 2022**（STD scaling） | 高斯（σ 可学习） | 否 | 单一标量 s 乘 σ，在 re-calibration set 上最小化 NLL | 无 | σ 乘子、用验证集定标度、ENCE/分箱可靠性图 | "尺度再校准（variance recalibration）"作为方法；"分箱诊断校准" |
| **Song et al. 2019**（distribution calibration） | 多输出 GP + Beta link | 否 | 学习 P(Y≤y \| 预测分布) 的映射（distribution-level，强于 quantile-level） | 无（GP 后验） | "对整条预测分布而非单个分位数做校准"的思想 | "distribution calibration"这个术语与概念本身 |
| **CRUDE**（Zelikman et al. 2020, arXiv:2005.12496） | **误差形状固定、任意**，由 μ(x)、σ(x) 平移缩放 | 否 | z=(y−μ)/σ 的经验分布 → 分位数映射回 μ+σ·z_p | **与 inductive conformal 等价 → 可交换下有限样本边际** | 几乎全部结构：固定形状 + location-scale 重标定 + 用 hold-out 定标定 | "固定形状 + 位置尺度重标定"作为方法；"与 conformal 有理论联系" |
| **EMOS**（Gneiting et al. 2005） | 高斯，μ=a+Σb_i f_i，σ²=c+d·S² | 否 | 最小化 CRPS（或 NLL） | 无 | **本质就是"以点/集合预报为条件的 location-scale 后处理"**；μ、σ 都是预报值的函数 | "把点预测后处理成校准分布"这一整套范式 |
| **Censored EMOS / CSG**（Scheuerer & Hamill 2015；Baran & Nemoda 2016） | 左删失于 0 的 shifted gamma / GEV | **是（左删失）** | 删失似然 + 最小化 CRPS | 无 | **"删失似然 + 位置尺度分布回归 + 事后后处理"三件套** | "把删失写进分布后处理的似然"是新的 |
| **crch**（Messner, Mayr & Zeileis 2016） | 左/右删失或截断的 Gaussian / logistic / **Student-t** | **是** | 条件异方差删失回归的 MLE，location 与 scale 可有不同协变量 | 无 | **location-scale + Student-t + 右删失 = 我们的模型类的教科书形式** | "异方差删失回归"这一模型类；"用 t 分布吸收厚尾" |
| **Tobit**（Tobin 1958） | 潜变量 + 上下界 | **是** | 边界观测的似然取尾概率 | 无 | y=min(125,T) 的似然 = P(Y≥125) | "把封顶值当删失而非点质量"在统计学上的任何新意 |
| **Avati et al. 2020**（Survival-CRPS） | 对数正态等参数生存分布 | **是（右 + 区间删失）** | 用 Survival-CRPS 替代 MLE 训练 | 无 | "删失 + 全分布 + proper scoring rule 评价"的组合 | "生存框架下用删失似然/评分规则做校准" |
| **Haider et al. 2020**（D-Calibration） | 任意 ISD | **是**（删失样本按均匀分摊进后续区间） | 诊断：PIT 桶是否均匀，χ² 检验 | 诊断，无保证 | **"分桶检查整条分布是否校准"就是我们的 §4 分层覆盖** | "用分箱 PIT/覆盖诊断删失下的分布校准"是新的 |
| **Candès, Lei & Ren 2023**（conformal survival） | 任意生存模型 + conformal wrapper | **是（Type I 右删失）** | 加权 conformal，对删失分布做 inverse-probability 重加权 | 有限样本边际（外生删失）；条件独立删失下 doubly robust 近似 | "删失 + conformal + 覆盖保证" | "在删失数据上给出覆盖保证" |
| **Split conformal / CQR**（Vovk；Lei et al. 2018；Romano et al. 2019） | 区间（无分布假设） | 否 | 残差/分位数残差的经验分位数 | 有限样本**边际**（需可交换） | 我们的 CQR/分裂共形基线；覆盖率目标定标度的思路 | 任何覆盖保证；"覆盖率达到名义值"作为贡献 |
| **Normalized / locally-adaptive CP**（Papadopoulos et al. 2008；Lei et al. 2018） | 区间，宽度随 x 变 | 否 | 残差除以局部尺度估计 ρ̂(x) | **近似条件**（需 ρ̂ 一致） | "用 σ(x) 归一化残差" | "自适应区间宽度带来条件覆盖" |
| **DCP**（Chernozhukov, Wüthrich & Zhu 2021, PNAS） | 由条件分布模型给出，经秩置换 | 否 | PIT + 秩置换 | **近似条件有效**（需条件分布估计一致）；误设下退化为近似边际 | 用 PIT/秩做校准、异方差下的条件有效性 | "PIT 类方法只能给边际保证"这一说法（它是反例） |
| **Mondrian / 分组 conformal**（Vovk 2012） | 区间，按 taxon 分组 | 否 | 组内单独定分位数 | **组条件精确**（组内可交换） | 分层（按 ŷ 分箱）报告覆盖 | 把分层经验覆盖说成"条件覆盖的证据" |
| **Xu, Gao, Wang & Zio 2026**（RESS 275:112763） | BDL 高斯 + 等渗校准 σ + split CP | 否 | **等渗回归校准异方差 σ**，再套 split conformal | 边际（SCP，可交换下） | **C-MAPSS 上"事后校准预测方差 + conformal"已发表** | "在 C-MAPSS 上首次做概率 RUL 校准"；"用单调映射校准 σ" |
| **本方法** | 离散 Student-t(ν=5)，126 格，末格为 ≥125 | **是（人为右删失）** | location-scale MLP(ŷ) + 覆盖率目标 σ 乘子（交叉拟合） | **无**（分层覆盖是描述性诊断） | — | 上述全部 |

**表的核心信息**：我们恰好落在 `EMOS 的结构` × `crch 的似然` × `CRUDE/Kuleshov 的重标定` × `Levi 的尺度乘子` 的交点上。四个方向都有 5–30 年的文献，交点本身没有空缺。

---

## 3. CRUDE 的考证

### 3.1 结论

**CRUDE = "CRUDE: Calibrating Regression Uncertainty Distributions Empirically"，作者 Eric Zelikman、Christopher Healy、Sharon Zhou、Anand Avati（Stanford），arXiv:2005.12496（2020-05-25 首版，v6 2021）。**

- 它**不是** conformal 方法，**不是** RUL 方法，**不是**删失方法。
- 它是**回归不确定性的事后分布校准**：假设误差形状在输出空间上恒定，只随预测均值平移、随预测标准差缩放。
- 方法（其 Algorithm 1）：在 held-out 校准集上算 z=(y−μ(x))/σ(x)，排序；对分位数 p 取 z_p=Z_C[⌊p·L⌋]；输出 q=μ(x\*)+σ(x\*)·z_p。
- 论文明确声明"a direct mathematical equivalence with conformal inference"，即 CRUDE 与 inductive conformal prediction（ICP）等价。

证据：
- <https://arxiv.org/abs/2005.12496>（HTML 全文：<https://arxiv.org/html/2005.12496v6>）
- 摘要原文："CRUDE makes the weaker assumption that error distributions have a constant arbitrary shape across the output space, shifted by predicted mean and scaled by predicted standard deviation. We detail a theoretical connection between CRUDE and conformal inference."

### 3.2 与我们方法的关系（这是本次审计最不利的一条）

CRUDE 的结构是：`固定形状 + μ(x) 平移 + σ(x) 缩放 + hold-out 定标定`。
我们的结构是：`固定形状（Student-t, ν=5）+ μ(ŷ) 平移 + σ(ŷ) 缩放 + dev 定标定`。

差别只有三点，且没有一项是方法级的：

1. 形状参数化（Student-t）而非经验分布。→ 这是**参数化 vs 非参数**的取舍，CRUDE 那篇的 §5 limitation 已经把"放松 constant shape"列为未来工作。用参数形状换样本效率，是标准权衡，不是新方法。
2. 条件变量是 ŷ 而非 x。→ 这恰恰是 Kuleshov §3.4 的 `(y−μ(x))/σ(x)` 特征，也是 EMOS 的 `μ=a+b·f`。用预报值本身做条件变量在 EMOS 里是 20 年的默认做法。
3. 删失。→ 见 §4。

**审稿人最可能的原话**："This is CRUDE with a Student-t shape and a censored likelihood. What is new?" 我们没有硬答案。

### 3.3 如果 CRUDE 不是你们想的那篇

若你们内部的 "CRUDE" 指别的东西，最接近的候选是：

| 候选 | 全称 | 关系 |
|---|---|---|
| **CRULP** | Conformalized RUL Predictor（HKUST；ensemble CQR + cross-conformal，航空发动机数据集） | 缩写接近，但方法是 conformal RUL，不是分布校准 |
| **CRPS** | Continuous Ranked Probability Score | 拼写接近，是评价指标不是方法 |
| **CRUDE**（本尊） | 见上 | 与本方法结构重合度最高 |

仓库内检索 `CRUDE` 无任何命中（`grep -ril CRUDE` 在全部 `.md/.py` 中为空），说明这个缩写不是本项目内部产生的术语，应是从外部文献带入——**建议核对原始出处，因为这直接决定你们是否在无意识复现**。

---

## 4. 删失处理的文献现状

### 4.1 统计学：这是 1958 年的标准做法

- **Tobin (1958)** 给出上限/下限删失回归的 MLE：边界观测的似然取尾概率。`y=min(125,T)`、观测到 125 时似然取 P(T≥125)，就是 Tobit 似然，一字不差。
- **Klein & Moeschberger (2003), p.74** 给出右删失似然的教科书形式：`L ∝ S(C_r)`。
- **Messner, Mayr & Zeileis (2016)** 的 `crch` 包把"条件异方差 + 左/右删失/截断 + Gaussian/logistic/**Student-t** + location 与 scale 可有不同协变量"做成了一个通用模型类。我们的模型（位置与尺度都是 ŷ 的函数、形状 t、右删失）**就是 crch 的一个实例**，只是把线性 predictor 换成 66 参数的 tanh MLP。

### 4.2 分布后处理：删失似然也早就是标配

- **Scheuerer & Hamill (2015)**、**Baran & Nemoda (2016)**：降水量的 CSG-EMOS，**左删失于 0 的 shifted gamma**，用删失似然 + CRPS 最优估计。这是"删失 + 位置尺度 + 事后后处理"的完整先例。
- 结论：**"把删失写进分布后处理的似然"没有任何新颖性**，只是过去用在左删失（降水）而非右删失（RUL）。

### 4.3 RUL/PHM 领域：删失是标准，但"封顶即删失"不是

| 工作 | 删失来源 | 处理方式 | 与我们是否重合 |
|---|---|---|---|
| Noot, Birmele & Rey 2024（PHME 8(1), DOI 10.36001/phme.2024.v8i1.3974） | 轨迹在失效前被截断（真实删失） | LSTM/Transformer 训练时纳入删失样本；C-MAPSS 上验证 | 概念重合（删失进训练），数据机制不同 |
| Rahat & Kharazian（SurvLoss, PHME） | 同上 | 改造 MAE/MSE 使其能处理删失样本；C-MAPSS + SCANIA | 概念重合 |
| Avati et al. 2020 | 右删失 + 区间删失 | Survival-CRPS | 概念重合（删失 + proper score） |
| Candès et al. 2023 | Type I 右删失 | 加权 conformal | 覆盖保证层面重合 |
| C-MAPSS 主流做法（Saxena et al. 2008；Heimes 2008 起） | **无删失**，只是把标签 clip 到 125 | 直接当点标签回归 | 不重合 |

**关键判断（未检索到反例，但不敢断言不存在）**：我没有找到任何一篇 C-MAPSS 论文把 125 封顶**显式建模为右删失**。所以"cap-as-censoring 在 C-MAPSS 上未被做过"这句话目前可以站住——**但它的价值远低于它带来的风险**，原因见下。

### 4.4 这个"删失"其实不是删失（审稿人最可能击中这里）

C-MAPSS 训练集是 run-to-failure，**真实失效时刻 T 完全已知**。`y = min(125, T−t)` 是一次**确定性的标签变换**，不是观测机制的限制。由此产生三个可被逐条攻击的漏洞：

1. **估计量换了，但论文没说。** 把 y=125 当删失，等价于把目标从"封顶标签 Y"换成"潜在未封顶 RUL T−t"。这两个量的条件均值不同。`CALIBRATION_LAYER_RESULTS.md` §6.2 报告"层的均值会移动 ≤0.2 个周期"，这正是估计量切换的代价，但文中把它解释成"真实的偏差修正"。
2. **估计量与评价指标错配。** §2 的 RMSE 是对**封顶标签**算的，而分布是对**未封顶 T** 建的。对 ŷ≈112–115 的发动机，潜在分布的均值必然 >125，对封顶标签算 RMSE 会系统性吃亏。这可以直接解释 §4 里 `[100,126)` 段一致的 −8.5 到 −12.1 负偏——**那是评价口径造成的，不是"测量学发现"**。
3. **"条件均值在 125 之外不可辨识"这句是错的。**（`CALIBRATION_LAYER_RESULTS.md` §4）训练集里 T 已知，125 之外的 RUL 完全可辨识；不可辨识的只是"如果只给你封顶标签、不给你 T"的情形。这句话会被懂生存分析的审稿人当场指出。

**如果一定要保留这一块**，正确写法是：不叫 right-censoring，叫 **Tobit / grouped likelihood on a clipped target**，并明确声明估计量是未封顶 RUL，同时把评价指标改成对未封顶 T 计算（C-MAPSS 训练集可算），或明确声明只对 Y 评价且放弃 >125 段的结论。

---

## 5. 条件覆盖的现状

### 5.1 硬结论：分布无关的精确条件覆盖不可能

- **Lei & Wasserman (2014)**：任何具非平凡有限样本条件有效性的预测带，在连续分布下几乎处处期望长度无穷。
- **Barber, Candès, Ramdas & Tibshirani (2021), Thm 3.1**：即使放松到"对每个概率 ≥δ 的子群覆盖"，也只能靠把边际水平抬到 1−αδ 来平凡实现，宽度按 1/δ 发散。原文："is impossible to attain beyond the trivial solution"。

### 5.2 有理论保证的，只有下面这几类

| 类型 | 保证 | 前提 | 代表 |
|---|---|---|---|
| Split conformal / CQR / jackknife+ / CV+ | **边际**，有限样本 | 可交换 | Vovk；Lei et al. 2018；Romano et al. 2019；Barber et al. 2021 |
| Mondrian / 分组 conformal | **组条件**，有限样本 | **组内**可交换 | Vovk 2012 |
| Normalized / locally-adaptive CP | 近似条件 | 局部尺度估计 ρ̂(x) 一致 | Papadopoulos et al. 2008；Lei et al. 2018 |
| Distributional CP（DCP） | **近似条件**（对全 x 向量） | 条件分布估计一致；误设下退化为近似边际 | Chernozhukov et al. 2021, PNAS |
| Conformal survival | 边际（外生删失）；doubly robust 近似 | 删失机制或条件生存函数之一估好 | Candès et al. 2023 |
| 贝叶斯/参数化（Nix-Weigend、EMOS、crch、本方法） | **无** | — | — |

### 5.3 我们的分层经验覆盖属于哪一类

**属于"诊断"，不属于"保证"。** 它是可靠性图（reliability diagram）的表格版，与 Haider et al. 2020 的 D-Calibration 桶检验同构，只是桶用 ŷ 而非 PIT 划分。它可以说"在这一批测试发动机上，这一段的经验覆盖率是 0.67"，不能说"这一段的条件覆盖率是 0.90"。

若要把它升级成保证，唯一现成路径是 **Mondrian conformal**（按 ŷ 分箱，箱内单独定分位数）——但那要求箱内 dev 与 test 可交换，而 §3 的消融已经证明跨舰队不可交换（FD003 per-subset 0.88→0.77）。所以这条路自己堵死了。

---

## 6. C-MAPSS 概率 RUL 校准的近期工作（2022–2026）

按"校准机制"归类。★ = 与本方法重合度最高。

| # | 工作 | 年/刊 | 校准机制 | 报告指标 | 处理封顶删失 | DOI / 链接 |
|---|---|---|---|---|---|---|
| 1 | de Pater & Mitici, *Novel metrics to evaluate probabilistic RUL prognostics* | 2022, PHME 7(1):96–109 | 无（MC-dropout CNN，直接评估） | **CRPS、weighted CRPS、α-coverage、Reliability Score** | 否 | 10.36001/phme.2022.v7i1.3320 |
| 2 | Javanmardi & Hüllermeier, *Conformal prediction intervals for RUL estimation* | 2023, IJPHM 14(2) | 三种 CP 算法套在 CNN / GBM 上 | 覆盖率、区间宽度 | 否 | 10.36001/ijphm.2023.v14i2.3417 |
| 3 | Xu & Zio, *A general data-driven framework for RUL with UQ using split conformal* | 2023, ICSRS | split CP（声称处理非可交换） | 覆盖率、宽度 | 否 | 10.1109/ICSRS59833.2023.10381131 |
| 4 | Mao, Li & Zhao, *RUL prediction based on time-series features and CQR* | 2024, Meas. Sci. Technol. 35(12) | **CQR** | 区间覆盖、宽度 | 否 | 10.1088/1361-6501/ad762c |
| 5 | Zhan, Kong, Wang, Jin & Xu, *RUL prediction with UQ based on multi-distribution fusion* | 2024, RESS 251:110383 | MDF + MLE（多分布融合） | 区间预测（PICP 类） | 否 | 10.1016/j.ress.2024.110383 |
| 6 | Wang et al., *A parallel prognostic method integrating UQ* | 2024, Processes 12(12):2925 | MC dropout（贝叶斯近似） | 点精度 + 不确定性 | 否 | 10.3390/pr12122925 |
| 7 | Noot, Birmele & Rey, *LSTM and Transformers for RUL considering censored data* | 2024, PHME 8(1) | 训练时纳入删失样本 | RMSE（无覆盖类指标） | **是（真删失，非封顶）** | 10.36001/phme.2024.v8i1.3974 |
| 8 | Basora, Viens, Chao & Olive, *A benchmark on UQ for deep learning prognostics* | 2025, RESS 253:110513 | BNN / MCD / DE / HNN 横评 | 精度 + 不确定性质量 + 分解 | 否（N-CMAPSS） | 10.1016/j.ress.2024.110513 |
| 9 | Zhou, Wei & Ma, *RUL prediction and UQ with multi-branch deep networks and probabilistic calibration* | 2025, IEEE SDPC | snapshot ensemble + MC dropout + **CP** | RMSE、Score、区间覆盖 | 否 | 10.1109/SDPC68151.2025.11347806 |
| 10 | ★ **Xu, Gao, Wang & Zio, *A novel uncertainty-aware framework for RUL integrating UQ and calibration*** | 2026, RESS 275:112763 | **BDL + 等渗回归校准异方差 σ + split conformal** | 精度 + 校准 + 覆盖 | 否 | 10.1016/j.ress.2026.112763 |
| 11 | Belaunzaran et al., *Bifurcated RUL prediction* | 2026, PHME 9(1) | 健康/退化分段的 Weibull + MC dropout + 校准 sigmoid | 不确定性带 | 否 | 10.36001/phme.2026.v9i1.4903 |

**这张表告诉我们三件事：**

1. **"C-MAPSS 上做概率 RUL + 报告 PICP/MPIW/CRPS"已经饱和**。第 1、2、3、4、9 条都做了。我们不是第一个。
2. **"事后校准 σ + conformal" 在 C-MAPSS 上已被发表**（第 10 条，RESS，2026）。这条对我们最致命：它同时占据了"BDL 不确定性"+"单调校准"+"conformal 保证"三个位置，而我们的 66 参数层在表达能力上不如它、在保证上不如它。
3. **没有任何一篇处理 125 封顶删失**。这是唯一空位——但如 §4.4 所述，这个空位是"统计学上早已解决、只是没人在这份数据上写过"的类型，属于细节而非方法。

---

## 7. 我们还剩什么可以主张

按"可辩护强度"降序。★ = 一击致命风险。

### 7.1 可辩护（弱但站得住）

**(A) 封顶标签的删失化处理，在 C-MAPSS 概率校准语境下首次被显式实现并量化其影响。**
- 支撑：Tobin 1958；crch 2016；CSG-EMOS 2016（形式先例）；§6 表中无一篇做封顶删失（检索证据）。
- 反例风险：中。审稿人会说"这是 Tobit，1958 年就有了"。**必须**把它写成"我们指出并修正了 C-MAPSS 概率评价中一个被普遍忽略的口径错误"，而不是"我们提出了删失校准方法"。
- ★ 若保留当前"条件均值不可辨识"的表述，直接被击穿（§4.4 第 3 点）。

**(B) 一个可复现的协议：池化四舰队 dev + 折外预测 + 交叉拟合定标度。**
- 支撑：Kuleshov §3.2（K 折建议）、cross-conformal（Vovk）、CV+（Barber et al. 2021）。
- 反例风险：低（协议本身不声称新），但**贡献强度也低**——这是工程规范，不是科学发现。

**(C) 66 参数、骨干无关、不重训的事后层在 3 骨干 × 4 舰队上的一致 CRPS 改善。**
- 支撑：自有实验（`CALIBRATION_LAYER_RESULTS.md` §2）。
- 反例风险：★ **高**。基线只有"同方差"。没有对比 CRUDE、Kuleshov 等渗、EMOS、Nix-Weigend MVE、normalized CP。审稿人一句"your only baseline is a constant-variance Gaussian"就结束了。这一条**现在不能主张**，除非补三个基线（CRUDE 与 normalized split CP 最好实现，各约 50 行）。

### 7.2 不可辩护（会被一击致命，建议直接删除）

**(D) ★ "首次在 C-MAPSS 上做概率 RUL 校准"。**
反例：§6 表第 1、2、3、4、9、10 条。直接死。

**(E) ★ "提出了一种新的 RUL 概率校准方法 / 新的概率 RUL 建模范式"。**
反例：CRUDE + Kuleshov §3.4 + EMOS + crch 四重覆盖。死。

**(F) ★ "我们的方法提供覆盖保证 / 条件覆盖"。**
反例：我们没有任何保证。分层覆盖是诊断。且 §3 自证跨舰队不可交换，连 conformal 的边际保证都不成立。若写"guarantee"，死。

**(G) ★ "单舰队 dev 不足以认证校准（100 台发动机太少）"。**
这是误诊。FD003 的 0.88→0.77 是 11 个百分点；n=100、p=0.9 时二项标准误约 3%，11 点远超抽样误差。真实原因是**分布偏移**（dev 与 test 不同分布），不是样本量。把它写成"数据量被低估"会被统计审稿人指出逻辑错误。正确的说法是"交叉拟合估计的是 dev 分布的覆盖率，对 test 舰队无约束力"——但这句话等于承认"我们的定标度对 test 无效"，反而伤自己。
反例风险：★ 极高，且是自伤。

**(H) "把封顶原子从点质量改成尾概率，修正了 PIT 顶端格偏热"。**
问题：PIT 顶端偏热（0.13–0.22 vs 0.10）恰恰可能是**评价口径**造成的（用封顶标签去检验未封顶分布），不是模型缺陷。把它当成"发现"会被反问："你自己造的口径不一致，凭什么算发现？"

### 7.3 若一定要投 TIM，最诚实的定位

把论文从"方法"改成"**测量学 / 评价协议**"：
- 主题：*RUL 概率预测中的封顶标签口径问题及其对覆盖与 CRPS 评价的影响*。
- 贡献：证明（i）把 125 当点质量会让顶端分箱覆盖与 CRPS 系统性失真；（ii）Tobit 化可以修正，但代价是估计量切换，必须同时改评价口径；（iii）在四个舰队上量化这个效应。
- 这样写，Tobit 的古老性不再是弱点（我们用的是它，不是发明它），而 §6 表里没人做过封顶删失这件事才变成真正的空位。
- 但即使这样，也**只能声称 measurement contribution，不能声称 method contribution**。TIM 接受前者，前提是标题和摘要不越界。

---

## 8. 参考文献

**校准与分布回归**

1. Nix, D. A., & Weigend, A. S. (1994). Estimating the mean and variance of the target probability distribution. *Proc. IEEE ICNN'94*, 55–60. DOI: 10.1109/ICNN.1994.374138
2. Nix, D. A., & Weigend, A. S. (1994). Learning local error bars for nonlinear regression. *NIPS 7*. <https://papers.nips.cc/paper/1994>
3. Kuleshov, V., Fenner, N., & Ermon, S. (2018). Accurate uncertainties for deep learning using calibrated regression. *ICML*, PMLR 80:2796–2804. <https://proceedings.mlr.press/v80/kuleshov18a.html> ｜ arXiv:1807.00263
4. Levi, D., Gispan, L., Giladi, N., & Fetaya, E. (2022). Evaluating and calibrating uncertainty prediction in regression tasks. *Sensors*, 22(15), 5540. DOI: 10.3390/s22155540 ｜ arXiv:1905.11659
5. Song, H., Diethe, T., Kull, M., & Flach, P. (2019). Distribution calibration for regression. *ICML*, PMLR 97:5897–5906. <https://proceedings.mlr.press/v97/song19a.html> ｜ arXiv:1905.06023
6. Zelikman, E., Healy, C., Zhou, S., & Avati, A. (2020). **CRUDE: Calibrating regression uncertainty distributions empirically.** arXiv:2005.12496. <https://arxiv.org/abs/2005.12496>
7. Gneiting, T., Raftery, A. E., Westveld, A. H., & Goldman, T. (2005). Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation. *Monthly Weather Review*, 133(5), 1098–1118. DOI: 10.1175/MWR2904.1
8. Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. *JRSS-B*, 69(2), 243–268. DOI: 10.1111/j.1467-9868.2007.00587.x
9. Scheuerer, M., & Hamill, T. M. (2015). Statistical post-processing of ensemble precipitation forecasts by fitting censored, shifted gamma distributions. *Monthly Weather Review*, 143, 4578–4596.（卷期页据 CRAN `ensembleMOS` 文档；DOI 未核实）
10. Baran, S., & Nemoda, D. (2016). Censored and shifted gamma distribution based EMOS model for probabilistic quantitative precipitation forecasting. *Environmetrics*, 27, 280–292. arXiv:1512.04068. <https://arxiv.org/abs/1512.04068>

**删失回归**

11. Tobin, J. (1958). Estimation of relationships for limited dependent variables. *Econometrica*, 26(1), 24–36. DOI: 10.2307/1907382
12. Messner, J. W., Mayr, G. J., & Zeileis, A. (2016). Heteroscedastic censored and truncated regression with crch. *The R Journal*, 8(1), 173–181. DOI: 10.32614/RJ-2016-012
13. Klein, J. P., & Moeschberger, M. L. (2003). *Survival Analysis: Techniques for Censored and Truncated Data* (2nd ed.), p. 74. Springer.

**生存分析校准**

14. Haider, H., Hoehn, B., Davis, S., & Greiner, R. (2020). Effective ways to build and evaluate individual survival distributions. *JMLR*, 21(63), 1–63. <https://jmlr.org/papers/v21/18-772.html>
15. Avati, A., Duan, T., Zhou, S., Jung, K., Shah, N. H., & Ng, A. Y. (2020). Countdown regression: sharp and calibrated survival predictions. *UAI*, PMLR 115:145–155. <https://proceedings.mlr.press/v115/avati20a.html> ｜ arXiv:1806.08324
16. Goldstein, M., Han, X., Puli, A., Perotte, A. J., & Ranganath, R. (2020). X-CAL: explicit calibration for survival analysis. *NeurIPS 33*. <https://proceedings.neurips.cc/paper/2020/hash/d4a93297083a23cc099f7bd6a8621131-Abstract.html>
17. Candès, E. J., Lei, L., & Ren, Z. (2023). Conformalized survival analysis. *JRSS-B*, 85(1), 24–45. DOI: 10.1093/jrsssb/qkac004 ｜ arXiv:2103.09763

**共形预测与条件覆盖**

18. Vovk, V. (2012). Conditional validity of inductive conformal predictors. *ACML*, PMLR 25:475–490. <https://proceedings.mlr.press/v25/vovk12.html>
19. Lei, J., & Wasserman, L. (2014). Distribution-free prediction bands for non-parametric regression. *JASA*, 109(506), 1090–1101.（DOI 未核实）
20. Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *JASA*, 113(523), 1094–1111.（DOI 未核实）
21. Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized quantile regression. *NeurIPS 32*.
22. Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). The limits of distribution-free conditional predictive inference. *Information and Inference*, 10(2), 455–482. DOI: 10.1093/imaiai/iaaa017
23. Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. *Annals of Statistics*, 51(2), 816–845.（DOI 未核实）
24. Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). Distributional conformal prediction. *PNAS*, 118(48), e2107794118. DOI: 10.1073/pnas.2107794118
25. Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS 32*.
26. Gibbs, I., & Candès, E. J. (2021). Adaptive conformal inference under distribution shift. *NeurIPS 34*.

**C-MAPSS / PHM 概率 RUL**

27. Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *PHM 2008*. DOI: 10.1109/PHM.2008.4711414
28. Heimes, F. O. (2008). Recurrent neural networks for remaining useful life estimation. *PHM 2008*. DOI: 10.1109/PHM.2008.4711422（DOI 未核实）
29. de Pater, I., & Mitici, M. (2022). Novel metrics to evaluate probabilistic RUL prognostics with applications to turbofan engines. *PHME*, 7(1), 96–109. DOI: 10.36001/phme.2022.v7i1.3320
30. Javanmardi, A., & Hüllermeier, E. (2023). Conformal prediction intervals for remaining useful lifetime estimation. *IJPHM*, 14(2). DOI: 10.36001/ijphm.2023.v14i2.3417
31. Xu, W., & Zio, E. (2023). A general data-driven framework for RUL estimation with UQ using split conformal prediction. *ICSRS*, 67–74. DOI: 10.1109/ICSRS59833.2023.10381131
32. Mao, S., Li, X., & Zhao, B. (2024). RUL prediction based on time-series features and conformalized quantile regression. *Measurement Science and Technology*, 35(12). DOI: 10.1088/1361-6501/ad762c
33. Zhan, Y., Kong, Z., Wang, Z., Jin, X., & Xu, Z. (2024). RUL prediction with uncertainty quantification based on multi-distribution fusion structure. *RESS*, 251, 110383. DOI: 10.1016/j.ress.2024.110383
34. Wang, R., Zhang, Y., Hu, C., Yang, Z., Li, H., Liu, F., Li, L., & Guo, J. (2024). A parallel prognostic method integrating UQ for probabilistic RUL prediction of aero-engine. *Processes*, 12(12), 2925. DOI: 10.3390/pr12122925
35. Noot, J.-P., Birmele, E., & Rey, F. (2024). LSTM and Transformers based methods for RUL prediction considering censored data. *PHME*, 8(1). DOI: 10.36001/phme.2024.v8i1.3974
36. Rahat, M., & Kharazian, Z. SurvLoss: a new survival loss function for neural networks to process censored data. *PHM Society European Conference*. <https://papers.phmsociety.org/index.php/phme/article/download/4052/2460>（年份未核实）
37. Basora, L., Viens, A., Chao, M. A., & Olive, X. (2025). A benchmark on uncertainty quantification for deep learning prognostics. *RESS*, 253, 110513. DOI: 10.1016/j.ress.2024.110513 ｜ arXiv:2302.04730
38. Zhou, X., Wei, F., & Ma, X. (2025). RUL prediction and UQ with multi-branch deep networks and probabilistic calibration. *IEEE SDPC*. DOI: 10.1109/SDPC68151.2025.11347806
39. Xu, W., Gao, S., Wang, Z., & Zio, E. (2026). A novel uncertainty-aware framework for RUL prediction integrating uncertainty quantification and calibration. *RESS*, 275, 112763. DOI: 10.1016/j.ress.2026.112763
40. Belaunzaran, X., Nappa, A., Artetxe, A., & Sierra, B. (2026). Bifurcated RUL prediction: a hybrid approach for realistic uncertainty characterization. *PHME*, 9(1). DOI: 10.36001/phme.2026.v9i1.4903

---

## 附：三处内部文档与文献的冲突（供作者核对）

| 内部表述 | 出处 | 与文献冲突 |
|---|---|---|
| "条件均值在 125 之外不可辨识" | `CALIBRATION_LAYER_RESULTS.md` §4 | 训练集 T 已知；只有"仅给封顶标签"时才不可辨识（Tobin 1958；Klein & Moeschberger 2003 p.74） |
| "单舰队 dev 100 台不足以认证覆盖率" | §3 | 11 个百分点的落差远超 n=100 的抽样误差（≈3%），归因应为分布偏移而非样本量（Barber et al. 2021） |
| "这不是模型缺陷，是测量学问题"（顶端负偏） | §4 | 负偏来自"未封顶分布 vs 封顶标签"的口径错配，是我们自己引入的，不能归给数据 |
