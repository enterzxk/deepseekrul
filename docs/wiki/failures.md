# 失败与已关闭路线

> 保留失败证据与机制解释，禁止删除。规则见 AGENTS.md §12/§15。

## 已关闭路线

| 路线 | 关闭日期 | 观察到的现象 | 机制解释（当前最佳） | 证据 | 是否可能复活 |
|---|---|---|---|---|---|
| 非对称校准 | 2026-11 | 最优解退化为对称 | 残差分布近对称，偏度不足以支撑非对称分位数 | docs/tables/asymmetric.csv | 否（除非换数据集出现强偏态） |
| 分层校准（mu/sigma 分箱） | 2026-11 | 全部更差 | 分箱内样本过少，校准方差吃掉条件化收益 | docs/tables/mondrian.csv | 低（需更大样本/更粗分箱） |
| 引擎级校准 | 2026-11 | 等价于放宽区间 | 引擎级欠覆盖是边际覆盖的必然代价，非可校准偏差 | docs/tables/engine_vs_inflation.csv | 否 |
| seed 集成 | 2026-11 | n=2 有效，n=4 退化 | 误差相关性随 seed 数上升，方差收益饱和 | docs/tables/ensemble_scaling.csv | 低 |
| 局部化 CQR | 2026-11 | 40 格无收益，FD003 更差 | 端到端 CQR 已吸收局部尺度，再局部化只增方差 | docs/tables/cells_all.csv | 否 |
| 宽度缩放 a·s(x)+q̂ | 2026-11-08 | a=1 最优，验证集选的 a* 更差 | 加性 q̂ 稀释对比度的假设不成立；逐格选择是噪声 | docs/tables/width_scaling.csv | 否（D001 判定门关闭） |

## 失败处理模板

```
## Fxxx · 日期 · 路线名
现象：<可复现的观测>
机制：<为什么会这样，尽量给公式或统计解释>
证据：<文件/实验目录>
判定：refine / replace / abandon
后续：<如果 refine，新假设 ID>
```

## F001 · 2026-11-09 · 新立项第一版「识别边界」框架被文献占位

**现象**：D018 提出的新问题「能否只用校准集信息识别将要失效的对象？识别依赖哪些可检验条件？」
在写成六件套第一版后，经独立文献审计发现其**贡献类型**已被三篇工作占位。

**机制（为什么会被占）**：该问题的抽象形式（无标签目标信息下判定覆盖是否可信）是一个
2026 年已被理论化的通用问题；把它套到 RUL 上不产生新的类型，只产生新的数据集。

**证据**
- `driftguard2026`：定理级可检测界 + minimax 推论 + [0,1] 识别集（label-free 审计）。
- `hacc2026`：算术阈值 $n\ge\lceil1/\alpha\rceil$ + 收缩修复（min-group 0.28→0.72）。
- `zhou2026cpa`：条件覆盖评估作为有监督预测任务 + 收敛率。
- `domingues2026`：C-MAPSS 上「边际达标、条件崩溃」的现象已发表（68–76% / 37–50%）。
- 草案第一版全文见 `docs/PROPOSAL_unit_coverage_identifiability.md` 附录 C（保留追溯）。

**判定**：abandon（第一版框架），replace 为第二版「协议算术 + 后果量化 + ICC 可复现性 + 文献审计」。

**后续**：H005（待 L2 立项门裁决，见 `docs/PROPOSAL_unit_coverage_identifiability.md` §9）。
若 L2 判否决，则本条目升级为「新方向整体否决」，按 D016 第 8 项转内部技术报告/应用稿候选。

## F002 · 2026-11-09 · 第二版立项草案被 L2 否决：计数对象混淆 + 保证对象缺失

**现象**：第二版草案（「协议算术下界 + 后果量化 + ICC + 文献审计」）在 L2 立项门被判 `reject`
（D021）。否决理由不是「贡献不够」这类软判断，而是**内部形式化错误**。

**机制（为什么会写错）**：把三个不同的计数对象当成同一个「9」的约束对象——
(a) 校准**发动机**数（6/16/6/15）；(b) 每台校准发动机的**窗口**数（130–200）；
(c) G1 写的「最大单机校准点数」。而 $n\ge\lceil(1-\alpha)/\alpha\rceil$ 约束的是
**指定保形构造里的校准分数数目**，与上述三者都不是同一件事。更根本的是：草案没有定义
**发动机级分数**，也没有建立「分数保证 → 单台目标机覆盖保证」的推导。

**证据**
- 草案 §3.2 G1 vs §5.1 vs §5.3（三处计数对象不一致，G1 按自身写法会失败）。
- `src/punet/conformal.py::engine_calib_split`（整机分配 → 测试机自身校准标签为 0）。
- `docs/decision_log.md` D021（完整裁决与 5 条缺陷）。

**判定**：abandon（第二版），不再沿 D018 的问题立项。

**后续**
- 保留可复算结果并**限定范围**：逐机覆盖分布（真实最小 0.2814–0.5521）、
  **固定划分下**跨 seed 稳定性（ICC 0.780）。
- 删除三类主张：协议算术不可达、FD002/FD004 可认证、稳定失效无法被收缩改善。
- 按 D016 第 8 项整理内部技术报告（`docs/INTERNAL_REPORT_baseline_audit.md`）。
- 若未来重开认证方向，必须先给出保证事件、随机性来源、α 与 δ、具体分数与假设，
  并推导发动机数如何约束该保证（D021 推翻条件）。

## F003 · 2026-11-09 · H005 证伪：校准侧难度排序不能预测条件尺度是否有用

**现象（预注册判定门，全部在折级 n=20 上）**

| 门 | 统计量 | 阈值 | 结果 |
|---|---|---|---|
| G-A | Spearman(ρ_cal, r_scale) | ≤ −0.35 且 CI 上界 < 0 | **−0.039**，CI [−0.438, +0.419] → **FAIL** |
| G-B | ρ_cal(多工况) − ρ_cal(单工况) | ≥ 0.20 且 ≥4/5 折同号 | +0.216，4/5 → PASS（但只说明诊断量本身有子集差异，不构成机制） |
| G-C | Spearman(shift_sd, 覆盖缺口) | ≤ −0.30 且 CI 上界 < 0 | **−0.221**，CI [−0.449, +0.454] → **FAIL** |
| G-D | go/no-go 规则（τ=0.5，先验固定） | 均值 ≤ −2% 且 FD003 ≥ −2% | 均值 **+2.79%**、FD003 **+13.77%** → **FAIL** |

**机制解释（当前最佳）**：不是"机制被反证"，而是**诊断量在可用校准规模下不可估**。
事后（非预注册）在每格内对校准发动机做 500 次 bootstrap：

| 子集 | 校准发动机数 | ρ_cal 均值 | 格内 bootstrap SD | 95% 半宽 |
|---|---|---|---|---|
| FD001 | 6 | +0.091 | **0.490** | 0.961 |
| FD003 | 6 | +0.486 | **0.414** | 0.811 |
| FD002 | 16 | +0.374 | 0.249 | 0.488 |
| FD004 | 15 | +0.635 | 0.171 | 0.335 |

在 6 台校准发动机下，ρ_cal 的抽样 SD 达 0.41–0.49，95% 区间几乎覆盖 [−1,1]；
此时任何基于它的 go/no-go 规则都无法被可靠校准——这与 G-A/G-D 的失败一致。
**注意**：这**不证明**「尺度有用 ⟺ 难度排序」在期望意义上为假，只证明**该诊断量在此校准规模下不可用**。

**证据**：`scripts/scale_diagnostic.py`、`tmp/freeze/h005.json`、`docs/tables/mcd80_cells.csv`。

**判定**：refine → 机制主张撤回；保留「条件尺度效应高度异质」这一现象事实
（FD002 19/20 胜 −7.05% vs FD003 1/20 胜 +28.53%）。

**后续**：若要把「可诊断性」作为贡献，必须先给出**一个在 n_cal=6 下仍可估的统计量**
（或明确的最低校准规模），并预注册；不得直接复用 ρ_cal。

### F003 更正（2026-11-09，D022 L2 审查发现）

原实现有三处统计缺陷，**修正后结论不变**，但区间与措辞必须用修正版：

| # | 缺陷 | 修正 |
|---|---|---|
| 1 | G-A/G-C 的 bootstrap 只重采样自变量、结果量固定 → 破坏配对，区间无效 | 改为**按折重采样 (x,y) 配对**：G-A CI 由 [−0.438,+0.419] 变为 **[−0.431,+0.317]**；G-C 由 [−0.449,+0.454] 变为 **[−0.712,+0.300]** |
| 2 | 事后「95% 半宽」= 1.96×平均 SD，不是逐格百分位区间 | 改为**逐格 2.5/97.5 百分位**再取均值：FD001 **0.898**、FD002 **0.479**、FD003 **0.747**、FD004 **0.324**（无定义重采样比例 0.0） |
| 3 | G-D 代码写 `FD003 >= -2%`，语义是「FD003 至少好 2%」，与预注册「不得恶化」不符 | 改为 `FD003 <= +2%`；**两种写法下结果都是 FAIL**（FD003 +13.77%） |

**保留原输出**：`tmp/freeze/h005_v1_original.json`（有缺陷）；修正版 `tmp/freeze/h005_v2_corrected.json`。
**不得**把修正后的分析追认成原始预注册结果；预注册阈值未改。

**措辞更正**：H005 的失败应表述为「**未达到预注册支持标准**」，
**不能**写成「机制已被排除」「所有校准侧诊断均不可行」「普遍不可估」。
n_cal=6 下的 bootstrap 波动只支持「**该估计量在本样本中不稳定**」，
不支持「任何规则都无法校准」。

## F004 · 2026-11-09 · MUP/ICST 测量扰动传播路线被关闭

**现象**：`docs/MUP_SPEC_V2.md` 提交 L2 复核，被判 reject（D028）。

**机制（为什么错）**
1. **条件对象混淆**：ICST 把"给定清洁 $x$ 的扰动平均"当成"给定观测 $z$ 的条件误差"。
   反例：$X,\varepsilon\sim N(0,1)$ 独立、$Y=X$、$\mu(t)=t$ ⇒ 真实
   $\mathbb E[(Y-\mu(Z))^2|Z=z]=\tfrac12+\tfrac{z^2}{4}$，而 ICST 锚定为常数 1；无限数据也消不掉。
2. **代数退化**：$r_0^2-2r_0\overline d+\overline{d^2}=\overline{(r_0-d)^2}$，
   恒等式只是平方展开；用无偏 MC 组合时 ICST 等价于"直接带噪残差监督"，无新估计量。
3. **分层保证不成立**：$S_0\sim U(0,2)$、$S_1\sim U(0,1)$、混合 0.1/0.9 时，
   总体 90% 分位数下 $\nu=0$ 组覆盖仅 47.4%；混合分位数不保证最差组。

**证据**：`docs/decision_log.md` D026/D027/D028；`docs/MUP_SPEC.md`；`docs/MUP_SPEC_V2.md`。

**判定**：abandon（当前 A/ICST 立项）；**不**恢复 B、H004 或已关闭的后处理路线。

**后续**：转候选 C（退化状态空间 + 保形层）的两天、0 GPU 设计门。
若未来重开测量误差方向，必须先给出部署目标
$h_\nu(z)=\int[a(x)+(m(x)-\mu(z))^2]p_\nu(x|z)dx$ 的可部署后验积分算法与误差界。

## Public-source compatibility and evidence limits — 2026-11-11

AIDGN unselected-model imports, FFVE missing imageio, and FC-STGNN abstract
array-container construction initially failed. Minimal recorded runtime
repairs completed the main matrix; failed attempts are retained in its
`evidence/failed_attempts/`. CPT's released pretraining contract passes a
scalar output back into a36-dimensional classifier; only that interface
probe is FAILED, no simplified regression substitute was trained.
Historical own-launcher source gaps survive explicitly in
`evidence/EXECUTION_PROVENANCE.json`; original configs/results are intact.
See the independent audit/report for all protocol risks and limitations.
