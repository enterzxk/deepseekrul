# 投稿前清单（IJPHM v2）

> 生成 2026-11-09。**本文件是「证据包是否就绪」的唯一检查表**；每条都指向可验证的产物。
> 状态：**证据包已完成并通过全部机器可查的验收；剩余项全部是人工输入或 L3 授权。**

---

## 1. 交付物

| # | 文件 | 是什么 | 状态 |
|---|---|---|---|
| 1 | `docs/manuscripts/ijphm_v2/submission_draft.pdf` | IJPHM 官方模板编译稿（8 页、3 图、APA） | ✅ |
| 2 | `docs/manuscripts/ijphm_v2/draft.docx` | Word 版（IJPHM 接受 Word/PDF） | ✅ |
| 3 | `docs/manuscripts/ijphm_v2/{00..06}_*.md` | 分节源文件（摘要/引言+相关工作/协议/结果/讨论/附录） | ✅ |
| 4 | `docs/manuscripts/ijphm_v2/references_v2.bib` | 16 条参考文献，DOI 级核实 | ✅ |
| 5 | `docs/manuscripts/ijphm_v2/CLAIM_LEDGER.md` | 主张—证据台账（C1–C22） | ✅ |
| 6 | `docs/manuscripts/ijphm_v2/NEIGHBOR_MATRIX.md` | 最近邻差异矩阵（含核实等级） | ✅ |
| 7 | `docs/figures/ijphm_v2/fig{1,2,3}_*.{pdf,png}` | 三张图（矢量 PDF + 预览 PNG） | ✅ |
| 8 | `docs/BASELINE_FREEZE.md` | 基线冻结清单（A/B/C 可复算分级） | ✅ |
| 9 | `docs/INTERNAL_REPORT_baseline_audit.md` | 内部技术报告 | ✅ |
| 10 | `docs/VENUE_FIT.md`、`docs/IJPHM_REQUIREMENTS.md` | 期刊匹配与官方要求核验 | ✅ |

---

## 2. 可执行验收（全部通过）

| 门 | 命令 | 结果 |
|---|---|---|
| 基线冻结 | `python scripts/freeze_audit.py` | **50/50** |
| 论文数字 | `python scripts/paper_evidence.py --cells docs/tables/mcd80_cells_seeded.csv` | **17/17** |
| 稿件一致性（台账覆盖 / 数字锚点 / 引用键 / 违禁措辞） | `python scripts/check_manuscript_consistency.py` | **PASS**（22 条主张、16 个引用键） |
| 引用真实性 | `python scripts/citation_audit.py` | **16/16**（标题相似度 1.000） |
| C5 独立复算 | `python scripts/independent_c5_check.py` | **PASS**（ICC 0.780 / ρ 0.762 / Jaccard 0.599） |
| C11 独立复算 | `python scripts/independent_c11_check.py` | **PASS**（split 口径 6/6、−16.30%） |
| 编译 | `node scripts/build_ijphm_submission.mjs` | 8 页、3 图、0 未解析引用 |

---

## 3. 必须由用户提供（agent 无法代填）

| # | 项 | 位置 | 说明 |
|---|---|---|---|
| U1 | 作者姓名、单位、邮箱 | `scripts/build_ijphm_submission.mjs` 的 `\author`/`\address` | 目前是 `First Author` 占位 |
| U2 | 第一作者全名（版权脚注） | 同上，`\phmLicenseFootnote{}` | 模板要求真名 |
| U3 | Biographies | `05_discussion.md` 的 `# Biographies` 段 | 目前是占位 |
| U4 | Acknowledgment / 资助 / 利益冲突声明 | `05_discussion.md` 的 `# Acknowledgment` 段 | 目前是占位 |
| U5 | 是否投稿、投哪一版 | — | **L3**：对外提交需明确授权 |

---

## 4. 仍未核实（不阻塞，但投稿前应知情）

| 项 | 现状 | 风险 |
|---|---|---|
| HACC 全文（`10.1016/j.mlwa.2026.101038`） | 只读到摘要 + §1–2.2（ScienceDirect 阻断） | 正文只写到摘要级，已在 `NEIGHBOR_MATRIX.md` 标注 |
| CRULP 全文（`10.1109/TIM.2024.3522678`） | 闭源，只有摘要 + CrossRef 参考文献 | 正文未对其机制下判断，仅作相关工作引用 |
| IJPHM 投稿系统流程 | 未走一遍 OJS 草稿 | 投稿前需实际操作 |

---

## 5. Cover letter 要点（供用户起草）

1. **贡献类型**：测量/评估（不是新方法）——多 seed、同中心对照、预注册诊断检验、逐机覆盖。
2. **与同刊先例的差异**：Robinson (2026) 报边际覆盖 88–92%、无逐机覆盖；本稿报逐机分布下尾（最小 0.12）。
3. **与最近邻的差异**：Domingues (2026) 报现象；本稿加同中心对照 + 4 seed 复现 + 预注册诊断。
4. **可复现性**：全部数字由脚本断言（50/50、17/17），产物随仓库。
5. **不主张**：方法新颖性、单位级保证、对任何最近邻方法的优越性。

---

## 6. 已知的、必须在审稿中如实回答的弱点

1. 20 折共享训练数据；80 格共享一个 5 折划分 → 全部描述性。
2. 单一主干（IDA-PE）、四个子集、FEMTO 仅 6 个全寿命轴承。
3. 未使用官方 test（刻意的协议选择）。
4. ρ_cal 的估计方差只在 6/15/16 三个校准规模上测过。
5. 条件尺度的机制未被识别——只报「方向随工况结构分离」这一观测。

---

## 7. 结论

**证据包已就绪**（第 1、2 节全部通过）。阻塞项只有第 3 节的人工输入与第 5 节的 L3 授权。
在第 3 节补齐前，稿件**不应**被提交。
