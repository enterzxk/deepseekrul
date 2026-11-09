# IJPHM 稿 v2 · 结构与主张映射

> 定位：**测量/评估贡献**（D022），不是方法创新。目标刊：IJPHM full-length regular paper。
> 数字一律取自 `docs/manuscripts/ijphm_v2/CLAIM_LEDGER.md` 的 C 编号；不得在正文里出现台账外的数字。
> 状态：**骨架**（2026-11-09）。尚未写正文。

---

## 拟定标题（三选一，投稿前定）

1. *What conformal RUL intervals actually deliver: a multi-seed audit of coverage, unit-level failure, and a conditional epistemic scale on C-MAPSS*
2. *When does an epistemic-uncertainty scale help? A pre-registered audit of conformal RUL intervals*
3. *Marginal coverage is not unit reliability: an 80-cell audit of conformalized quantile regression for RUL*

> 倾向 1：既含主结果（C1–C3），也含两个负向发现（C4–C5、C6–C10），不夸大。

---

## 章节骨架与主张映射

### 1. Introduction（~1.5 页）

- 现象：保形 RUL 论文普遍报边际覆盖；工程使用关心单机可靠性。
- 三个问题：
  Q1 效率收益在多样本重复下稳不稳？（C1–C3）
  Q2 边际覆盖达标时，单机覆盖如何？（C4–C5）
  Q3 换一个条件尺度（认知不确定性），收益何时出现、能否事先诊断？（C6–C10）
- 贡献声明（四条，逐条对应主张）：
  1. 80 格多 seed 的**配对**评估，含同中心对照（C1–C3、C6）。
  2. 单机覆盖下尾的量化与**跨 seed 重复性**（C4–C5）。
  3. **预注册**的校准侧诊断检验，结果为未达支持标准（C9–C10）。
  4. 可复算的冻结协议与审计脚本（`freeze_audit.py` 50/50、`verify_mcd80.csv` 逐位一致）。
- **禁止**：写「首次」「唯一」「系统性高估」「保证被违反」。

### 2. Related work（~1.5 页）

- 2.1 保形预测与 CQR（Romano 2019；Barber 2021 条件覆盖不可能性）。
- 2.2 RUL 上的保形方法：CRULP（TIM 2025）、Robinson（IJPHM 2026）、Diao（Sensors 2026）。
- 2.3 条件覆盖审计：Yan 审计包（Zenodo 2026）、Domingues & Ather（JAIMLNN 2026）、
  HACC（MLWA 2026）、CPA（arXiv 2026）、DriftGuard（Research Square 2026）。
- **差异段**：本稿不是再报一次条件覆盖崩溃，而是（i）同中心对照隔离尺度效应，
  （ii）多 seed 重复，（iii）预注册诊断检验 + 估计稳定性。**每条差异必须能在全文矩阵里指到原文。**

### 3. Protocol（~2 页）

- 3.1 数据与折划分（`OUTER_SEED=20261104`，5 折，引擎级）。
- 3.2 校准/测试划分（`engine_split(seed=42, 0.3)`；校准发动机数 6/16/6/15）。
- 3.3 冻结基线（端到端双分位 + CQR；区间定义）。
- 3.4 四个被比较的臂：常数宽度、CQR、MC-Dropout 尺度、**同中心常数宽度**（关键对照）。
- 3.5 指标：Winkler（α 固定 0.1）、PICP、单机覆盖（min/p05/median）。
- 3.6 统计纪律：折为报告单位；20 折共享训练数据 → 只作描述性；bootstrap 仅描述。
- 3.7 可复现性：`freeze_audit.py` 50/50；`verify_mcd80.py` 重训点预测逐位一致；
  MC-Dropout 两遍抽样噪声底线（0.63/3.20/14.70 pp，方向翻转 0/80）。

### 4. Results（~4 页）

- **4.1 效率收益是稳的**（C1–C3）。表 1：80 格与折级；图 1：逐格相对变化分布。
  必须同段写 C3 的「子集均值匹配、逐格最大 8.84 pp」。
- **4.2 边际达标 ≠ 单机可靠**（C4–C5）。表 2：逐子集单机覆盖 min/p05/median；图 2：逐机覆盖分布。
- **4.3 换尺度：收益高度依赖工况结构**（C6–C8）。表 3：四子集 × 同中心对照；
  图 3：折级散点（多工况 vs 单工况）。
- **4.4 能否事先诊断？**（C9–C10）。预注册四门的结果表；图 4：ρ_cal 的逐格 bootstrap 分布。
  **必须写清**：这是「未达预注册支持标准」，不是「机制不存在」。

### 5. Discussion（~1.5 页）

- 为什么 FD002 有用、FD003 灾难：只给可检验的解释，不宣称已证机制。
- 对报告实践的含义：单 seed + 边际覆盖 + 无同中心对照，会同时高估效率与可靠性。
- 与最近邻的关系：我们复现了他们的现象，并加了对照与预注册检验。

### 6. Limitations（~0.5 页，必须诚实）

- 20 折共享训练数据；80 格共享 5 折划分 → 全部为描述性。
- 单主干（IDA-PE）；4 个子集；FEMTO 仅 6 个全寿命轴承。
- 官方 test 集未参与（刻意的协议选择，不是遗漏）。
- ρ_cal 的事后估计稳定性只在 n_cal=6/15/16 三个规模上测过。

### 7. Conclusion（~0.3 页）

- 一句话：CQR 的效率收益可复现；但边际覆盖不保证单机可靠性，
  而条件尺度的收益既依赖工况结构、又在本协议的校准规模下无法被一个自然诊断事先判据。

### Appendix A：覆盖对齐口径（事后诊断，C14–C15）

- 明确标注「事后、非部署规则」；与主表分开。

### Appendix B：可复算清单

- 脚本 → 表 → 图的映射；`freeze_audit.py` / `paper_evidence.py` / `scale_diagnostic.py` /
  `engine_coverage_stability.py` 的退出码与断言数。

---

## 写作纪律（D022 强制）

1. 每个数字后面在草稿里加 `[C#]` 标记，定稿时删。
2. 名义水平与覆盖对齐**不得同段混写**；对齐只出现在附录 A。
3. 任何「首次 / 唯一 / 系统性 / 保证被违反 / 不可估」一律不写。
4. 未核实文献不得写成确定事实；最近邻主张须指到原文位置。
5. 图与表必须由脚本生成，不手填。

---

## 待办（按序）

- [ ] 最近邻全文差异矩阵（DriftGuard / HACC / Domingues / Yan 论文本体）
- [ ] IJPHM 当前官方要求核验
- [ ] 表 1–3、图 1–4 的生成脚本（`scripts/make_ijphm_figs.py`）
- [ ] 正文写作（按节，逐节对台账）
- [ ] 投稿前：citation-audit + paper-claim-audit（独立审查，不自证）
