# NOVELTY_MATRIX — 最近邻工作与可证伪区别（v2，2026-11-05 19:35）

状态：文献元数据已核实（Crossref / Semantic Scholar DOI 解析）；本项目主张已根据
实测结果调整。**标注 [V] 的条目已通过 DOI 核实题名、年份、期刊；[P] 表示仅据摘要、
未取得全文，投稿前必须复核。**

## 0. 本项目的主张（据实测结果）

本项目**不主张**一个新的 RUL 预测器结构。实测表明：在 C-MAPSS FD002/FD004 上，
个体寿命先验的三种使用方式中只有"加性注入"带来小幅收益（FD002 外层验证 3.3%，三种子 3.41%），
而把它当时间基准或目标尺度分别劣化 13% 与 28%；逐观测的自适应修正不可辨识
（无可观测量能预测个体读数何时更优，|ρ|<0.08），尽管逐窗口 oracle 还有 15% 空间。

因此主张落在**测量学**层面：
1. 个体寿命尺度的可观测性随观测时间的定量曲线（corr 0.26→0.94，MAE 35.5→9.9）；
2. 三种使用方式的受控比较与边界条件；
3. 自适应修正不可辨识的系统负证据（含灵敏度系数的正/负两面）；
4. 评价端点与发动机样本决定排序的定量证据（外层验证 3.0% 的融合增益在官方测试上消失）。

## 1. 最近邻工作

### N1. Qu, Yang, Luo, Liu, Zhou, Guo, 2026 — 条件感知时空网络 [V]
- 题名：A Condition-Aware Spatio-Temporal Network for Robust Aero-Engine Remaining
  Useful Life Prediction
- 出处：IEEE Transactions on Instrumentation and Measurement, 2026
- DOI：10.1109/TIM.2026.3742092
- 核心：synergistic feature interaction (SFI) + spatio-temporal hierarchical graph
  attention network (STHGAT)；非线性基函数与 hash-based FFT 捕捉工况调制，
  层次图注意力聚合时空退化轨迹。
- 输入：工况设定 + 多传感器序列。假设：工况与传感器之间存在非线性协同，可由端到端
  网络学习条件相关表示。
- 与本项目区别：N1 把"工况—传感器协同"交给大型端到端网络；本项目显式分离工况处理
  （按工况簇的个体早期参考态）与退化信息提取，主干保持 42.3k 参数不变，并把
  **个体寿命尺度的可观测性**作为独立测量问题研究。
- 区分实验：同一划分与预算下比较端到端条件感知（本项目适配的 CET-MSA 与 PE）
  与本项目的显式条件匹配表示；并用工况支持度分层（MC3）检验低支持工况下的差异。

### N2. Kapoor, Gupta, Agarwal, Uppal, Juneja, Sharma, 2026 — phase-aware normalization [V/P]
- 题名：STARNet: Stacked Transfer-Aware for Robust Remaining Useful Life Prediction
  for C-MAPSS Multi-Regime Engines
- 出处：IEEE Access, 2026
- DOI：10.1109/ACCESS.2026.3663754
- 核心（摘要原文）：multi-scale Conv–BiLSTM + attention pooling、window statistics
  fusion、mixture-of-experts encoding，并引入 phase-aware normalization 与 delta
  features 以适应多工况。
- 假设：按"相位"归一化特征分布可稳定跨工况的输入分布。
- [P] **phase-aware normalization 的确切公式未取得全文核实。** 若其相位由循环数或
  批次统计量定义，则与本项目的 c/μ_T（因果个体寿命估计）不同；若同样使用个体寿命
  估计，则本项目的 NORM 线不构成独立创新——但本项目实测 NORM 在完整预算下劣化
  13%，因此无论公式如何，本项目**不主张**相位归一化的新颖性，只报告其在本设置下
  的失败边界。
- 区分实验：NORM vs NORM_SHUF/NORM_CONST；以及 NORM 在完整预算下的劣化（已有）。

### N3. Piao, Huang, Tsung, 2025 — 保形 RUL 预测 [V]
- 题名：CRULP: Reliable RUL Estimation Inspired by Conformal Prediction
- 出处：IEEE Transactions on Instrumentation and Measurement, vol. 74, 2025
- DOI：10.1109/TIM.2024.3522678
- 核心：深度点预测 + uncertainty-aware conformalized quantile regression +
  cross-conformal，把点预测转成有覆盖保证的区间。
- 假设：校准集与测试残差可交换。
- 与本项目区别：N3 关注区间覆盖；本项目关注点估计的观测证据可辨识性，并用
  发动机聚类 bootstrap 给出描述性区间（不主张覆盖保证）。两者回答不同问题，
  不可混用；本项目的 RMSE 不被表述为仪器不确定度。
- 区分实验：同一组点预测上分别加 CRULP 式保形层与本项目的引擎聚类 bootstrap，
  报告覆盖/宽度与点误差。

### N4. Gupta, 2026 — 工况归一化协议审计（预印本）[V]
- 题名：Auditing Operating-Condition Normalization in C-MAPSS Remaining Useful Life
  Evaluation
- 出处：engrXiv 预印本，2026（DOI 10.31224/8145，未同行评审）
- 核心：1080 次受控网格；多工况子集上按工况归一化在每个匹配单元都降低 RMSE
  （平均有符号差 −4.39 / −7.40）；"最终模型与归一化器在 train+val 上联合重拟合"
  带来 −1.119 至 −0.206 的 RMSE 变化。
- 与本项目关系：支持本项目"按工况归一化 + 只用训练折拟合"的做法，并说明预处理
  协议本身会显著改变结论。本项目不把按工况归一化当作创新。
- 区分实验：本项目所有对照共享同一归一化与划分；把"只用训练折拟合"作为协议前提。

## 2. 与最近邻的实质区别（一句话）

N1/N2 提供**更强的表示或归一化**来提升精度；N3 提供**区间可靠性**；N4 审计**预处理
协议**。本项目提供**个体寿命信息的可观测性与可辨识性边界**，并证明在该边界内
自适应修正不可实现、最佳可辩护做法是固定融合；同时给出**评价端点决定排序**的
定量证据。这是一个测量学贡献，不是预测器结构贡献。

## 3. 可证伪实验与结果

| 编号 | 对照 | 判据 | 结果 |
|---|---|---|---|
| F1 | 新方法 vs 锁定参考 | 平均增益 ≥3% | 未达成（最佳固定融合 2.9%，且不迁移到官方测试） |
| F2 | 机制对照（置换/常数/年龄工况来源） | 增益在对照下消失 | NORM/PHASE/DIFF 均未显示机制性增益 |
| F3 | 最强简单替代（固定融合、仅年龄门控） | 新方法须优于它们 | 自适应门控全部劣于固定融合 |
| F4 | 测量条件（MC1/MC2/MC3） | 报告干净与受影响条件 | 见 MEASUREMENT_CONDITIONS |
