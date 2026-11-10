# TIM 2025–2026 RUL 版面调查（2026-11-10）

> 目的：不再凭想象猜机制，先看 IEEE TIM 这一年**实际发了什么**，按机制归类，
> 找出没被占的位置。所有条目都给 DOI 或可定位链接；未取到 DOI 的标注。

## 0. 结论先行

TIM 的 RUL 版面在 2025–2026 年呈现**高度一致的形态**：新架构/新表示 + 机制级消融 + C-MAPSS 四子集。
九个方向里，我们能想到的"明显空位"（跨域/TTA、不确定性、物理信息、增广）**都已被占**。

**唯一没有被这批论文占住、且我们有真实资产的，是"区间/覆盖的测量与验证"——
但 TIM 的这批论文不做区间，所以那里既没有竞争者，也没有现成的审稿口味可循。**

## 1. 已核实的 TIM 2026 RUL 论文（按机制归类）

| 机制 | 论文 | DOI / 出处 |
|---|---|---|
| 多传感器时序**增广** + 数据不平衡 | Chen, Zhang, Jiao, *Multi-Sensor Time Series Forecasting-Based Augmentation Framework for RUL of Aero Engines* | `10.1109/TIM.2026.3684677`（TIM 75, Apr. 2026, 2509513） |
| **变量依赖**感知的轻量 CNN | Hou et al., *LVDACNN* | `10.1109/TIM.2026.3655912`（TIM 75, 2026） |
| **频谱-时序双通道 + 自适应图** | *iSTformer* | `10.1109/TIM.2026.3659594`（TIM 75, 2026） |
| **PDE 物理信息**时序增强网络（轴承） | *A Physics-Informed Temporal-Enhancing NN* | `10.1109/TIM.2026.3690837`（TIM 75, 2026） |
| **信号分解 + 混合模型**（CEEMDAN-KPCA + BiLSTM-Transformer，轴承） | *A Hybrid Prognostic Method…* | `10.1109/TIM.2026.3676093`（TIM 75, 2026, 3510413） |
| 物理结构 + 热力学**双图张量融合** | STGATN（用户精读四篇之一） | `10.1109/TIM.2025.3550613` |
| **异构动态图** | HDA-GNN | `10.1109/TIM.2025.3569885` |
| **Koopman 隐空间动力学** | ST-KDFormer | `10.1109/TIM.2025.3625335` |
| **工况-传感器交互 + 分层时空图** | SFI-STHGAT | Xplore 11732126（DOI 待补） |

## 2. 跨域 / 测试时自适应（TTA）：也已占满

| 工作 | 出处 |
|---|---|
| *Test-Time Adaptation for Cross-Domain RUL Prediction: When Does It Work and What Should Practitioners Use* | Eksploatacja i Niezawodność 2027;29(1):226057（online 2026-07-29） |
| *Zero-Shot Cross-Domain Transfer for RUL of Aero-Engines via Iterative Diffusion Refinement*（RUL-DiT） | RESS 2026, `S0951832026005971` |
| *Multi-Source Domain Generalization for Machine RUL via Risk Minimization-Based TTA* | IEEE 2024, Xplore 10738213 |
| *Integrated Fuzzy Subdomain and TTA Framework for Cross-Condition Bearing RUL* | Meas. Sci. Technol. 2026, 37(19):195105 |
| *RUL Prediction of Aeroengines via Target-Domain Mutual Information and Fine-Grained Alignment* | IEEE 2026, Xplore 11646913 |

## 3. 由此得到的判断

1. **TIM 的 RUL 审稿口味 = 架构/表示 + 机制消融 + 四子集 RMSE。**
   我们的资产（80 格、逐机覆盖、稳定性、区间审计）**不在这个口味里**。
2. **"明显空位"不存在**：增广、变量依赖、频谱、物理信息、图、跨域、TTA、混合模型，
   在 2025–2026 的 TIM 上都有对应论文。
3. **唯一同时满足"没被这批论文占住"和"我们有真实资产"的位置**是
   **区间/覆盖的测量与验证**——但它在 TIM 的 RUL 版面上没有先例，审稿风险高，
   且统计内核（覆盖依赖校准单元数）已被 2026 年的统计文献占住。

## 4. 因此只剩两条可走的路

- **路线 M（测量方法）**：把"引擎级覆盖不可由边际覆盖推出"做成一个**正向的测量方法**：
  给出"要多少台发动机 × 多少窗口才能把引擎级覆盖估到给定精度"的公式 + 参考实现 +
  在 80 格上的验证，并证明它改变结论。定位是 *measurement methodology*，不是 audit。
  **成本 0 GPU、2–3 周**；风险：审稿人可能认为是"已知容忍界 + RUL 应用"。
- **路线 R（新表示）**：造一个新架构/训练机制。风险最高：需要真正的新想法，
  且要在四子集上打赢 2026 年的 SOTA；我们的算力（20 GPU-h 先导）不足以做大规模调参。

**建议**：走路线 M，并把它写成"方法"而不是"审计"——核心是一个**可执行的样本量公式 + 参考实现**，
不是"别人错了"。
