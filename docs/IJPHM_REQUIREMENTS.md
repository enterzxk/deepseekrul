# IJPHM 投稿要求核验（2026-11-09）

> 目的：D022 选定 IJPHM 后，`docs/research_state.md` §11 把「核验 IJPHM 官方范围/文章类型/原创性要求」
> 列为投稿前第一步且「不得跳过」。本文件只记录**可核验**的事实，并标出核实等级。
>
> 核实等级：**A** = 直接抓取官方页面正文；**B** = 官方域页面经检索索引取到的正文片段
> （直连被本机网络代理 404）；**C** = 二手来源，需投稿前复核。

## 0. 结论（GO / 条件）

| 项 | 结论 |
|---|---|
| 范围匹配 | ✅ 期刊范围是 PHM（prognostics / diagnostics / system health management），C-MAPSS RUL 区间评估落在范围内 |
| 文章类型 | ✅ 有 full-length regular paper 一档，要求「new and carefully confirmed findings」——我们的测量/评估发现可以走这一档 |
| 费用 | ✅ 官方长期声明无投稿费、无版面费（**投稿前仍应再确认一次**） |
| 原创性 | ✅ 内部技术报告未公开，不构成先前发表；稿件未投他刊 |
| **引用格式** | ⚠ **IJPHM 用 APA 格式，不是 IEEE 数字引用**——当前稿件是 IEEEtran + `\cite`，必须改 |
| 页数上限 | ✅ **官方无硬性页数限制**（"PHM Society does not impose any hard limits on the number of pages in a paper"） |
| 模板 | ✅ 官方 LaTeX（`PHMSociety.cls`）与 MS Word 模板已下载；LaTeX 用 `\PHMbibliography` + **apacite**，两栏，摘要 150–400 词且不得引用文献 |
| 评审方式 | ✅ 单盲（评审人匿名、作者实名） |

**净结论**：范围与类型匹配，可以按 D022 的 IJPHM 路线推进；但**引用格式必须从 IEEE 改成 APA**，
这是一项确定会发生的工作量，应在写作阶段就按 APA 组织，而不是最后再转换。

## 1. 逐项核实

### 1.1 范围（B）

> "The International Journal of Prognostics and Health Management (IJPHM) is the premier online open access
> journal related to multidisciplinary research on **Prognostics, Diagnostics, and System Health Management**."
> —— 期刊自述（多份官方 CFP 与欢迎词一致）

期刊目标包括「为同行评审的 PHM 知识提供汇聚点」「推动 PHM 教育与研究的跨学科协作」
「促进并保证 PHM 实践的专业标准」。
来源：[IJPHM Flyer (2014)](https://www.phmsociety.org/sites/phmsociety.org/files/ijPHM%20Flyer%20january2014.pdf)、
[Welcome to IJPHM (2009)](http://www.phmsociety.org/sites/phmsociety.org/files/phm_submission/2009/ijPHM_10_004.pdf)

**对本稿的含义**：我们做的是「RUL 预测区间在部署条件下的实测表现与失效结构」，
属于 prognostics 的评估/测量工作，**在范围内**。但要注意：期刊定位是 PHM，
所以正文必须把结论翻译成 PHM 决策语言（维护决策、风险），不能只写成机器学习基准报告。

### 1.2 文章类型（B）

> "IJPHM publishes **full-length regular papers, technical briefs, communications, and survey papers**.
> Full-length regular papers should describe **new and carefully confirmed findings**, and experimental
> procedures and results should be given in detail sufficient for others to replicate the work."

来源：[IJPHM Accepting Submissions](https://phmsociety.org/ijphm-accepting-submissions/)

**对本稿的含义**：走 **full-length regular paper**。关键词是 "new and carefully confirmed findings" ——
这正是 D022 定的「测量/评估定位」能对上的表述：我们的发现是**经验发现**（尺度异质性、
边际覆盖掩盖单机失效、覆盖匹配只在子集均值层面成立），不是新算法。
**不要**在摘要里出现 "we propose a novel method" 这类措辞。

### 1.3 费用（C，投稿前需再确认）

> "Publication Fees **NO** / Submission Charges **NO** / Article Processing Fees **NO** /
> Article Access Charges **NO** / Membership Fee **NO** / Other Charges **NO**"

来源：[PHM Society IJPHM 页面（镜像）](https://phmsandbox.com.es/international-journal-of-prognostics-and-health-management-issn-2153-2648/)，
以及官方欢迎词中「free and unrestricted access」的表述。
⚠ 该条来自镜像站，标 C；但 PHM Society 自创刊起即免费开放获取，与官方欢迎词一致。
**投稿（L3 对外动作）前由用户确认一次即可。**

### 1.4 引用格式（B，重要）

> "**IJPHM templates use APA style for citations and references.** For a better understanding of how APA
> style works please refer to slides 15-25 of this Tutorial on Basics of APA Style."
> "PHM Society has a strict policy on manuscript formatting before they can be published …
> Since IJPHM does not have a separate publishing department we rely solely on authors and reviewers
> to make sure papers are ready for publication in the archives."

来源：[IJPHM Submissions（官方 OJS）](https://papers.phmsociety.org/index.php/ijphm/about/submissions)

**对本稿的含义（确定工作量）**：
- 现稿是 IEEEtran + `\cite{}` 数字引用；IJPHM 要 **APA**（作者-年份，正文 `(Piao et al., 2025)`）。
- 需要：换模板/文档类、`\cite` → `\citep`/`\citet`（natbib apalike）或手写作者-年份、
  参考文献表按作者姓氏字母序、期刊名全称、DOI 保留。
- 这一步**不能留到最后**：主张台账里每条引用都要按 APA 重排，越晚改越贵。

### 1.5 评审方式（B）

> "A **single-blind** peer review process is followed where the reviewers are never known to the authors,
> but the authors are always known to the reviewers."

来源：[IJPHM Publication Ethics Code (2013)](http://ftp.phmsociety.org/sites/phmsociety.org/files/IJPHM%20Publication%20Ethics.pdf)

**含义**：不需要匿名化稿件（作者实名），但**不能**假设评审人看不到我们的代码仓库链接。

### 1.6 未核实项（必须补）

| 项 | 状态 | 补法 |
|---|---|---|
| 页数/字数上限 | ✅ **无硬性限制**（官方 submissions 页原文，2026-11-09 直连抓取） | 已解决 |
| 官方模板 | ✅ 已下载 `IJPHM_FullPaper_Latex_Template.zip`（含 `PHMSociety.cls`、样例 .tex/.bib）与 MS Word 模板 | 已解决 |
| 投稿系统流程与所需材料 | ❌ 未核实 | 投稿前在 OJS 走一遍草稿流程 |
| 预印本/技术报告政策 | ⚠ 未核实（我们的内部报告未公开，风险低） | 查 Ethics Code 全文 |

## 2. 先例检索（证明「评估类」在 IJPHM 有位置）

已核实存在的 IJPHM 评估/应用类先例（用于 cover letter 与 Related Work）：

| 论文 | DOI | 类型 |
|---|---|---|
| Robinson, C. (2026). Remaining Useful Life Estimation for Aircraft Engines with Risk-Aware Prediction Intervals via Conformalized Quantile Regression | `10.36001/ijphm.2026.v17i1.4724` | CQR 应用 + 风险评估，C-MAPSS 四子集 |
| Javanmardi, A., & Hüllermeier, E. (2023). Conformal Prediction Intervals for Remaining Useful Lifetime Estimation | `10.36001/ijphm.2023.v14i2.3417` | 保形区间用于 RUL 的方法性论文 |

**判断**：IJPHM 近年确实发表保形/RUL 方向的工作，且 Robinson 2026 就是「方法应用 + 评估」形态，
说明本稿的类型有先例。**但这也意味着**：同一期刊已有同族工作，Related Work 必须逐条区分
（差异表见 `docs/difftable_crulp_ijphm.md`）。

## 3. 对当前写作线的三条硬约束

1. **引用格式按 APA 写**，不要先写 IEEE 再转。
2. **摘要与引言不出现 "novel method"**；用 "we evaluate / we measure / we find"。
3. **结论必须落到 PHM 决策语言**（维护决策、风险、覆盖不足的后果），否则会被判为「机器学习基准报告」。

## 4. 待办（按序）

- [ ] 下载 IJPHM 官方 LaTeX 模板，确认页数与格式细节（未核实项）
- [ ] 把 `references.bib` 的 18+ 条按 APA 重排，正文引用改作者-年份
- [ ] 在 Related Work 中逐条区分 Robinson 2026 / Javanmardi 2023 / CRULP / Yan 审计包
- [ ] cover letter 写明：评估类贡献 + 与同刊先例的差异 + 代码/产物可复现性
- [ ] 投稿前由用户确认费用与最终稿（L3）
