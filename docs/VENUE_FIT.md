# 期刊匹配审查（D016 第 8 项）

> 2026-11-09。目的：把现有证据包对上具体期刊的贡献要求，给出可投/不可投的判定。
> **本文件不构成投稿决定**；投稿目标选择属 L2（D016 第 8 项），结论写 `decision_log.md`。

---

## 1. 证据包清单（可复算，全部有脚本与断言）

| # | 证据 | 规模 | 强度 | 入口 |
|---|---|---|---|---|
| E1 | CQR vs 常数宽度，名义水平 | 80 格 = 4 子集 × 5 折 × 4 seed；**75/80 格、20/20 折**、均值 −10.22% | 强（多 seed + 折级单位） | `freeze_audit.py` B1–B6；`mcd80_cells.csv` |
| E2 | 覆盖对齐比较 | 80 格、−9.09%、中位 −10.75% | 强，但**口径需标注**（D005 修正后） | 同上 |
| E3 | 覆盖匹配性 | 单 seed e2e，覆盖差 ≤1.3 pp | 中（单 seed） | `audit_closeout.md` §3 |
| E4 | 跨域 FEMTO | 6 轴承、5/6、−12.21% | 中（6 个全寿命轴承） | `freeze_audit.py` G1–G2 |
| E5 | 条件尺度异质性 | 80 格、同中心对照；FD002 **19/20 胜 −7.05%** vs FD003 **1/20 胜 +28.53%** | 强（同中心对照隔离尺度效应） | `mcd80_cells.csv`、`verify_mcd80.csv` |
| E6 | 校准侧诊断被证伪 | ρ_cal vs 结果 −0.039 [−0.44,+0.42]；G-D 规则 +2.79% | 强（预注册判定门） | `scale_diagnostic.py`、`h005.json` |
| E7 | 诊断量在 n_cal=6 不可估 | 格内 bootstrap SD **0.41–0.49**（n=6）vs 0.17–0.25（n=15–16） | 强（直接测量） | `scale_diagnostic.py` 事后块 |
| E8 | 逐机覆盖下尾 | 18/20 折 min<0.80；真实最小 **0.2814–0.5521** | 强 | `freeze_audit.py` F1–F3 |
| E9 | 逐机失效跨 seed 稳定 | ICC **0.780**、ρ +0.762、Jaccard 0.599 | 强（4 seed） | `engine_coverage_stability.py` |

**统一叙事（唯一有说服力的那条）**：

> 保形 RUL 的**边际覆盖**（0.87–0.93）与**单 seed 基准**容易让人高估它实际交付的东西：
> **在本冻结协议中，接近名义的总体覆盖仍伴随持续的单机低覆盖**（真实最小 0.28，且跨 seed 稳定）；
> 而常用的认知不确定性尺度，其收益高度依赖子集，且**在本协议的校准规模下，一个自然的
> 校准侧诊断未能通过预注册判定门**。CQR 相对常数宽度的效率收益是稳的（20/20 折），
> 但这不等于条件有效性。

> ⚠ 措辞纪律（D022）：不得写「系统性高估实际交付」（超出证据，边际覆盖本不保证单机覆盖）、
> 「不可估」「任何规则都无法校准」「机制已被排除」。只写实测到的范围与预注册结论。

这是**测量/评估**贡献，不是方法贡献。它不需要新方法，只需要把三件事放在一起并诚实量化。

---

## 2. 候选期刊与匹配证据

| 期刊 | 匹配证据（已核实） | 贡献类型匹配 | 风险 |
|---|---|---|---|
| **IJPHM**（Int. J. Prognostics and Health Management，OA，PHM Society） | Robinson 2026, 17(1), DOI `10.36001/ijphm.2026.v17i1.4724`：GBR + CQR + C-MAPSS 四子集 + 风险偏好——**纯应用**即被接收 | ✅ 高：应用 + 基准 + 诚实负结果 | 影响力低于 TIM；需 10–12 页完整稿 |
| **QRE**（Quality and Reliability Engineering International，Wiley） | 2026 有 conformal UQ + RUL 的应用论文（DOI `10.1002/qre.70453`，运放 RUL） | ✅ 中高：可靠性统计 + 应用比较 | 审稿人可能要求更强的统计推断（我们只能描述性） |
| **PHME**（PHM Society European Conference） | 2026 有轴承 RUL + conformal（DOI `10.36001/phme.2026.v9i1.4902`） | ✅ 高：会议短文，快 | 会议影响力有限；适合先占位 |
| **RESS**（Reliability Engineering & System Safety） | 并行线记录 Yan 的审计稿已投 RESS（**我未独立核实**） | ⚠ 中：偏方法/系统可靠性 | 门槛高于 IJPHM/QRE；纯审计稿风险大 |
| **IEEE TIM** | — | ❌ | D016/D021 已两次判定贡献不足；**不重开** |

---

## 3. 判定

**可投**：IJPHM（首选）、QRE（次选）、PHME（快速占位）。
**不可投**：TIM（已判）、RESS（当前证据强度不够，除非补方法层）。

**投稿前必须补的三件事**（都不是新实验）：

1. **把 E1–E9 写成一条论证线**，明确「测量/评估贡献」定位，不冒充方法创新。
2. **口径表**：E1（名义）与 E2（对齐）必须分开列，E3 标单单 seed，E5 标同中心对照。
3. **反例处理**：正面回答「这不就是 benchmark 吗」——答案是 E6/E7（诊断被证伪 + 不可估）
   与 E8/E9（单机失效稳定），这三条不是 benchmark 能给的东西。

**不建议**：为投 RESS/TIM 去补新方法实验——已关闭路线不得重开（D001/D002/D004/D013/D017），
且 D021 已判定「补实验凑贡献」不是路径。

---

## 4. L2 裁决（D022，2026-11-09）

**decision = revise，selected_option = A（IJPHM）**：现有证据足以支持一篇**范围受限的应用评估稿**，
但须先修正统计实现与过强表述。**不恢复 TIM 目标。**

| 项 | 裁决 |
|---|---|
| 目标期刊 | **IJPHM**（QRE 次选，PHME 会议，RESS 否决：现有材料未证明超出相邻审计工作的系统可靠性贡献） |
| 贡献定位 | ✅ 接受「测量/评估贡献」 |
| 差异化核心 | E5–E7 是**当前最清晰的候选组合**，但**不能**认定为「唯一未被占住」；E9（跨 seed 重复性）有辅助价值 |
| 必须先做 | ① 修正 G-A/G-C 配对 bootstrap、逐格百分位区间、G-D 判据符号；② 把 E7 标为事后估计稳定性；③ 删除「95% 区间几乎覆盖 [−1,1]」的统一断言；④ 完成最近邻**全文**差异矩阵；⑤ 核验 IJPHM 官方范围/文章类型/原创性要求（**目前未核实**） |
| 不许做 | 用「所有诊断均不可能」「边际保证被违反」「普遍不可估」等未证明主张去区别最近邻；增加结果数量维持投稿判断 |

**统计修复已完成**（2026-11-09，`scripts/scale_diagnostic.py`，修正版 `tmp/freeze/h005_v2_corrected.json`）：
G-A CI [−0.431,+0.317]、G-C CI [−0.712,+0.300]、逐格半宽 0.898/0.479/0.747/0.324、
G-D 判据改为 `FD003 ≤ +2%`。**四个门的结果全部不变**（G-A/G-C/G-D FAIL，G-B PASS）。

**尚未完成**（投稿前必须）：IJPHM 官方要求核验、最近邻全文矩阵、稿件重写。
> 未独立核实项：RESS/Yan 投稿一事；IJPHM 官方文章类型与原创性要求。

## 5. IJPHM 官方要求（部分核实，2026-11-09）

> 来源：PHM Society 的 IJPHM 作者信息页与征稿页（[Information for Authors](https://phmsociety.org/phm-forum/topic/information-for-authors/)、
> [IJPHM Accepting Submissions](https://phmsociety.org/ijphm-accepting-submissions/)、
> [IJPHM 创刊说明](http://www.phmsociety.org/sites/phmsociety.org/files/phm_submission/2009/ijPHM_10_004.pdf)）。
> ⚠ 这些页面版本为 2009–2021；**当前** `phmsociety.org/journal/for-authors` 页面本轮抓取失败，
> 投稿前必须再核一次。

| 项 | 官方表述（要点） | 对本稿的含义 |
|---|---|---|
| 范围 | 「prognostics, diagnostics, and system health management of complex engineered systems」；「assessing the current status and predicting the future condition of an engineered component and/or system」 | ✅ 直接命中：RUL 区间 + 评估 |
| 原创性 | 「must report original research」；不得已发表或同时投递 | ⚠ 原创性仍是主要风险；**测量/评估**类原创发现是否被接受，取决于审稿人 |
| 文章类型 | full-length regular papers / technical briefs / communications / survey papers | ✅ 走 **full-length regular paper** |
| 正文要求 | 「new and carefully confirmed findings」；「experimental procedures and results … sufficient for others to replicate the work」 | ✅ 我们有多 seed、冻结划分、可复算脚本；这正是我们的强项 |
| 评审 | 单盲；编辑决定 | — |
| 出版 | 开放获取、存档 | — |

**结论**：范围与文章类型匹配成立；**原创性门槛**是唯一未解决的判定点，
且只能由审稿人回答。投稿前须完成：官方当前要求复核 + 最近邻全文矩阵 + 稿件重写。
