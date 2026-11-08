# Citation Audit 报告

> 生成于 2026-11-08。方法：arXiv API（13 条）+ CrossRef（5 条）逐条核实存在性、元数据、上下文。

## 结论

**18 条文献全部真实存在**（无编造）。**发现并修复 3 处元数据错误**。

## 一、存在性（全部通过）

| 来源 | 条数 | 结果 |
|---|---|---|
| arXiv API | 13 | 13/13 存在 |
| CrossRef DOI | 3 | 2/3 可解析（1 条 DOI 错误，见下） |
| 无 DOI/arXiv（会议） | 2 | 按标题+作者核实存在 |

## 二、已修复的错误

| 条目 | 错误 | 真相 | 修复 |
|---|---|---|---|
| **busico2026** | DOI `10.1016/j.ress.2026.111436` 解析 404 | 真 DOI `...113634`，RESS vol.278，2027 | DOI、卷号、年份已改 |
| **wu2025** | 标为 IEEE TIM 期刊论文 | 实为 **ICPHM 2025 会议论文** | 改为 `@inproceedings`，补 DOI |
| **sensors2026** | 作者写作 `Anonymous` | Diao Runsheng, Zhou Mingzhe, Meng Guanglei, Wang Shanze | 作者已补全 |

## 三、核对后确认无误的疑点

| 条目 | 疑点 | 结论 |
|---|---|---|
| barber2023 | bib 年份 2023，arXiv 2022 | 正确：Annals of Statistics 51(2) 2023，arXiv 是预印本 |
| hou2025 | bib 年份 2025，arXiv 2026-03 | 正确：arXiv journal_ref 明确写 "IEEE TIM (2025)"，是期刊先出、arXiv 后发 |

## 四、未做的一层

**上下文适当性**（被引论文是否真的支持正文那句话）只做了抽样检查，未逐条做。
18 条中，方法类引用（romano2019, barber2023, zhai2026a/b）与数据集引用（saxena2008,
nectoux2012）风险低；风险较高的是 2026 年的"最近邻工作"引用（frizzo2025/2026, ly2025,
hou2025, yang2026, jin2026, lee2026, barry2026, srinivasan2026），因为它们的差异表是
论文新颖性论证的基础。

**建议**：投稿前把 Related Work 里每一条 2026 引用与原文摘要逐句对照一次。
