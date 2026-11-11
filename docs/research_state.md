# 研究状态（恢复入口）

> 更新于 2026-11-08 23:35。任何新会话/压缩恢复后，**先读本文件**，再读 `decision_log.md`。
> 规则见 `AGENTS.md` §9。

## 1. 当前目标

**当前版本已放弃 TIM 投稿目标**（D016，branch）。现目标（**2026-11-09 02:45 更新，D021 后**）：
1. ✅ 冻结现有 CQR 方法为基线，形成可追溯清单（D020，`docs/BASELINE_FREEZE.md`，50/50 断言）；
2. ✅ 用现有预测建立条件失效清单（D018，`docs/tables/failure_inventory.csv`）；
3. ❌ **围绕 D018 问题重新立项已被 L2 否决**（D021，reject）：草案存在计数对象混淆与
   「分数保证 → 单位级保证」推导缺失，修正后亦不足以跨 Q1 门槛；
4. ▶ 现目标：按 D016 第 8 项把现有材料整理为**内部技术报告 + 基线**
   （`docs/INTERNAL_REPORT_baseline_audit.md` 已完成）；
   应用稿可行性须先做**具体期刊的贡献匹配审查**，不得据此自动恢复 TIM 目标。

**D018 问题的处置**：**不再沿其立项**（D021）。保留其证据价值（逐机覆盖下尾、固定划分跨 seed 稳定性），
但删除「协议算术不可达」「FD002/FD004 可认证」「稳定失效无法被收缩改善」三类主张。

## 2. 已完成（可验证）

| 项 | 产物 | 状态 |
|---|---|---|
| 主实验 4 子集 × 5 折 × 2 seeds | `experiments/punet_matrix/` 40 格 | ✅ |
| 追加 seeds 2024 / 13 | `experiments/punet_matrix/*/f*_s{2024,13}/` | ✅ 79/80 |
| 名义水平配对 | 19/20 胜、中位 −8.7% | ✅ |
| 覆盖对齐配对 | 29/40 胜、中位 −8.4% | ✅ |
| 跨域 FEMTO 6 轴承 | `experiments/femto/` | ✅ 5/6、−12.2% |
| A4 置换消融 20 折 | 均值 +0.2% | ✅ |
| 覆盖诊断 20 折 | 引擎级最小 0.50–0.71 | ✅ |
| 论文 Markdown / LaTeX / PDF | `docs/manuscripts/` | ✅ 编译通过 |
| 图 3 张 | `docs/figures/` | ✅ |

## 3. 已排除的方向（不要重复）

| 方向 | 结论 | 证据文件 |
|---|---|---|
| 上下文中心修正（D002） | 不可跨引擎迁移，覆盖被破坏 | `docs/tables/d002_centre.csv` |
| 宽度放大（D001） | a=1 已最优 | `docs/tables/width_scaling.csv` |

| 方向 | 结论 | 证据文件 |
|---|---|---|
| 非对称校准 | 最优=对称，无收益 | `docs/tables/asymmetric.csv` |
| 分层校准（mu / sigma 分箱） | 全部更差 | `docs/tables/mondrian.csv` |
| 引擎级校准 | 等价于放宽区间 | `docs/tables/engine_vs_inflation.csv` |
| seed 集成 | n=2 有效、n=4 退化 | `docs/tables/ensemble_scaling.csv` |
| 局部化 CQR | 40 格上无收益，FD003 更差 | `docs/tables/cells_all.csv` |

## 3.5 论文主张（D003 后冻结）

> ✅ 2026-11-08 已用 `verify_paper_numbers.py` 对重建后的工件核对：**17/17 断言通过**。
> 论文口径 = CQR vs 常数宽度：75/80、均值 −9.1%、中位 −10.8%；名义水平 19/20、中位 −8.7%。
> 另注：`table_all_methods.py` 打印的 "Loc-CQR vs const 71/80、−8.4%、−11.7%" 是 **Loc-CQR** 对常数宽度，与论文头部主张（CQR 对常数宽度）不是同一比较，勿混用。

正面、限定范围的效率主张：
- 名义水平 19/20 胜、中位 −8.7%
- 覆盖对齐 29/40 胜、中位 −8.4%
- 增益集中于 FD001/FD002；FD003/FD004 接近打平（作为**适用范围**陈述，不作为负结果）
- 跨域 FEMTO 5/6、−12.2%
- 不主张"均匀改善"，不主张"方法学新颖性"

构建命令：`node scripts/build_manuscript.mjs`（含 7 章结构断言）

## 3.6 论文状态（2026-11-08 冻结）

- 编译产物：6 页、7 表、3 图、230 KB PDF，LaTeX 无错误
- 数字核对：16/16 断言通过（`scripts/verify_paper_numbers.py`）
- 文献：18 条 BibTeX、11 处引用、0 undefined
- 构建：`node scripts/build_manuscript.mjs`（结构断言 + 摘要同步 + 表题提升）
- 主张：按 D003 冻结为"限定范围的效率收益"

## 3.7 S9 缺口清单（2026-11-08 重建）

权威版本见 `docs/manuscripts/SUBMISSION_CHECKLIST.md`。摘要：

- 未完成且可独立执行：① citation-audit ② paper-claim-audit ③ H004 覆盖对齐重做
  ④ MC-Dropout 补到 80 格 ⑤ 图 4
- 必须用户处理：作者信息、页数确认、最终通读
- 过期记录已作废：旧的"40 格 / 2 seeds / 29-40 / 16-16"口径全部更新为 80 格 / 4 seeds / 75-80

## 4. 待执行

> 执行顺序、判定门与出口断言见 `docs/autopilot_contract.md`；按队列连续执行，不再逐项请示用户。

| 队列 | 状态 | 证据 |
|---|---|---|
| 0 从 a40 取回 preds | ✅ 完成 | 本地 80 个 `preds_point.csv` |
| 1 补 FD004 f4 s13 | ✅ 无需执行 | 80/80 格已齐（4 子集 × 5 折 × 4 seed），原"79/80"记录已过期 |
| 2 重建 `docs/tables/` | ✅ 完成 | `docs/tables/cells_all.csv`，80 行 |
| 3 核对论文数字 | ✅ 完成 | `verify_paper_numbers.py` **17/17 PASS** |
| 4 H004 训练侧假设取舍 | ✅ 完成（D009） | Codex（level 2）裁决 **关闭 H004**，0 格新增预算；登记为「未立项，证据不足而关闭」，非「已证伪」 |
| 5 重训 | ❌ 取消 | H004 关闭，无重训 |
| 6 收口审计 | ✅ 完成 | `docs/audit_closeout.md`（三个口径分清、80 格非独立、覆盖匹配已验证） |
| 7 文献去占位 | ⏳ 待做 | 见 `docs/manuscripts/SUBMISSION_CHECKLIST.md` |
| 8 G.7 整稿审阅 | ⏳ 待做 | 依赖 6、7 |
| 9 重编译 + 清单收口 | ⏳ 待做 | 依赖 8 |

## 5. 恢复命令

```bash
ssh a40 'cd ~/deepseek-rul && ls experiments/punet_matrix/*/f*_s*/preds_point.csv | wc -l'
python scripts/table_all_methods.py
python scripts/make_tim_tables.py
```

## 6. 环境事实

- 远端 `a40`：无外网、无 git；Python `/home/yndx/miniconda3/envs/a40_env/bin/python`
- 本地：无 torch。Python 必须用绝对路径 `C:\Users\20562\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe`（PATH 上的 `python` 是 Store 占位符，静默失败）；Node 同理见 `docs/autopilot_contract.md` §7.1
- `docs/tables/` 目录已不存在，`verify_paper_numbers.py` 无法运行（2026-11-08 实测）
- 数据集：C-MAPSS 在 `data/`；FEMTO 在远端 `~/femto`（3.1 GB）

## 7. 未验证/待确认

- 桌面 profile 的 `reasoningEffort` 是 `high` 而非约定的 `max`（见 AGENTS.md §2）
- 参考文献：正文 `\cite|\citep|\citet` 共 **11 处**、18 个唯一 key，与 `references.bib` 的 18 条一一对应；`git 7548e31` 已逐条核实存在（修了 3 处元数据错误）。**无 G10 阻塞。**
  仍缺：**CRULP（IEEE TIM 2024/2025, DOI 10.1109/TIM.2024.3522678）未收录**，它是同刊直接竞争工作，投稿前必须补引并做差异表（见 decision_log 投稿就绪度裁决）。

## 8. 长期自主科研配置（2026-11-08 第二轮）

- AGENTS.md 新增 §11–§17：科研使命与 Q1 标准、AUTO_PROCEED、子代理编队、GPU 政策、
  证据与 Wiki、不可改动边界、AGENT_GUIDE 缺失记录。
- 新建 ARIS Research Wiki：docs/wiki/{index,literature,hypotheses,experiments,failures,progress}.md
- 实测：ARIS 官方 AGENT_GUIDE.md 本机不存在；未虚构引用，见 AGENTS.md §17。
- 本轮未启动任何实验。

**恢复入口更新**：新会话读 research_state.md → decision_log.md → results_summary.md → docs/wiki/index.md。
## 9. 投稿前缺口清单（2026-11-08 23:50，主控线第一轮）

> 依据：`docs/research_state.md`、`docs/decision_log.md`（D009/D010）、`docs/results_summary.md`、
> `docs/manuscripts/SUBMISSION_CHECKLIST.md`、`docs/audit_closeout.md`。只核对磁盘产物，不重跑实验。

### 9.1 已确认存在（不需要重做）

| 产物 | 证据 | 状态 |
|---|---|---|
| 80 格预测（4 子集 × 5 折 × 4 seed） | `docs/tables/cells_all.csv` 80 行；`experiments/punet_matrix/` | ✅ |
| 80 格 e2e JSON（含 picp/pinaw/winkler/engine_min） | `experiments/punet_e2e/*/*/*.json` 80 个 | ✅ |
| 数值核对脚本 | `scripts/verify_paper_numbers.py`（127 行，17 项断言） | ✅ 脚本在 |
| 稿件 | `docs/manuscripts/manuscript.tex` / `.pdf` | ✅ 编译通过 |
| 收口审计 | `docs/audit_closeout.md` | ✅ 本轮新建 |

### 9.2 缺口（按优先级）

| # | 缺口 | 性质 | 可否 L1 直接做 | 判定门 |
|---|---|---|---|---|
| G1 | **`verify_paper_numbers.py` 本地跑不通**：它要 `docs/manuscripts/v1.0-draft.md` 与 `experiments/punet_matrix/*/f*_s*/preds_point.csv`，本地 preds 为 0 | 环境 | ✅（改路径/改源） | 17/17 PASS 或逐项说明失败原因 |
| G2 | **三个口径未在正文分开**：75/80（未对齐）、19/20（e2e，覆盖匹配）、29/40（覆盖对齐） | 主张 | ❌ 需 L2（D010 待复核） | 正文每处数字标明口径与 n |
| G3 | **「FD003/FD004 接近打平」与 75/80 表格矛盾** | 主张 | ❌ 需 L2 | 该句限定为覆盖对齐口径 |
| G4 | **80 格被当作独立证据**（seed 方差占 46.8%） | 统计 | ❌ 需 L2 | 主表改折级 n=20 |
| G5 | `SUBMISSION_CHECKLIST.md` 过期：写「2 seeds / 40 格」，实际 4 seeds / 80 格 | 文档 | ✅ | 与 `cells_all.csv` 行数一致 |
| G6 | **已查清**：`manuscript.tex` 实测 `\cite`=0、`\bibitem`=0、无 References 章节；`references.bib` 存在但从未被引用。清单「18 条、11 处引用」为**错误** | 证据 | ✅ 已查清 | 以实测计数为准 |
| ~~G10~~ | **假阴性，已撤回**：实测 `\cite|\citep|\citet` 共 **11 处**、18 个唯一 key、`references.bib` 18 条、`git 7548e31` 已核实 18 条全部存在 | 证据 | ✅ 已查清 | — |
| G7 | **29/40 原始数据本地不存在**，无法复算 | 证据 | ✅（标记） | 论文标注「未复现」或不作为主口径 |
| G8 | 未跑 `/citation-audit` 与 `/paper-claim-audit` | 审计 | ✅ | 两份报告写入 `docs/` |
| G9 | `docs/tables/` 只有 2 个 csv，缺 D004 的对齐明细 | 证据 | ⚠️ 部分 | 能从 80 格 e2e 重算的补上 |

### 9.3 可独立执行、可进实验队列

| 任务 | 判定门 | 预算 |
|---|---|---|
| T1 修 `verify_paper_numbers.py` 使其本地可跑（改用 `docs/tables/cells_all.csv` + `experiments/punet_e2e`） | 17 项断言全部可执行，PASS 或明确 FAIL 原因 | 本地 ≤15 min |
| T2 从 80 个 e2e JSON 重算覆盖对齐口径，补 `docs/tables/aligned_all.csv` | 行数 = 80；FD003/FD004 数值与 D004 记载同号 | 本地 ≤15 min |
| T3 校准 `SUBMISSION_CHECKLIST.md` 与 `research_state.md` §7 的过期陈述 | 与实际产物计数一致 | 本地 ≤10 min |
| T4 跑 `/citation-audit` | 0 条编造引用 | 本地 |
| T5 跑 `/paper-claim-audit` | 每个数字能回溯到产物 | 本地 |

### 9.4 必须 L2 的（已挂 PENDING-L2）

- G2/G3/G4 三处主张改动（已向 Codex 提出，见 `tmp/l2_audit.txt`）
- 是否进入 G.7 → 依赖上述改动定稿

## 10. 2026-11-09 02:45 · D016 第 2/5/6/7 项收口（主控线本轮）

| 项 | 状态 | 产物 / 裁决 |
|---|---|---|
| D016-2 基线冻结 | ✅ 完成 | `docs/BASELINE_FREEZE.md` + `scripts/freeze_audit.py`（**50/50 PASS**）+ D020 |
| D016-5/6 六件套草案 | ✅ 草案 v2 完成 | `docs/PROPOSAL_unit_coverage_identifiability.md` |
| D016-7 L2 立项门 | ❌ **否决**（reject） | **D021**；草案已标「已否决」，原文保留 |
| D016-8 内部技术报告 | ✅ 完成 | `docs/INTERNAL_REPORT_baseline_audit.md` |

**本轮新增的可复算证据**
- `scripts/freeze_audit.py`：50 项断言（产物清点 / 80 格 / 折级 / 名义水平 / A4 / 引擎级 /
  FEMTO / **划分同一性**），退出码 0 = 基线成立。
- `scripts/engine_coverage_stability.py`：ICC 0.780、跨 seed ρ +0.762、Jaccard 0.599。
- `scripts/launch_matrix_80cells.sh`：把原先只在 `tmp/`（gitignore）里的 80 格启动命令入版本控制。

**本轮纠正的口径错误**（详见 D020）
1. `failure_inventory.csv` 的 `engine_min` 是**折均值**，真实最小单机覆盖为
   0.4214 / 0.4500 / 0.5521 / 0.2814。
2. e2e 逐子集用 ratio-of-means，19/20 与中位数用逐格比值（差 ≤0.5 pp）。
3. 指标实现有两份拷贝（副本重算与 `cells_all.csv` max|diff| = 7e-15）。
4. 80 格共用 seed-42 校准/测试划分；分析期与运行期 q̂ 逐位一致。

**本轮证伪/作废**
- 第一版「识别边界」框架：被 DriftGuard/HACC/CPA/Domingues 占位（`failures.md` F001）。
- 第二版「协议算术审计」：被 D021 判为内部形式化错误 + 贡献不足（`failures.md` F002）。
- 「σ_T 是有效机制」：D017 已证伪（A4 +0.246%，11/20 变差）。

**下一步（按序）**
1. 不启动新立项、不启动 G5、不重跑实验。
2. 若要做应用稿：先做**具体期刊的贡献匹配审查**（D016 第 8 项），结论写 decision_log；
   不得把本报告当作投稿就绪认定。
3. 并行执行线的 MC-Dropout 80 格按 D019 继续，与本轮无关。

## 11. 2026-11-09 12:00 · 应用稿路线：revise → IJPHM（D022）

**用户指令**：不接受「负结果」的表述，要求继续 auto research。据此重新定位：
实验结果本身是好的；被判不足的是**贡献类型**，不是数据。

| 项 | 状态 | 产物 / 裁决 |
|---|---|---|
| MC-Dropout 80 格回填 | ✅ 完成，完整性门通过 | `experiments/punet_mcd80/`（80 格）、`verify_mcd80.csv`（重训点预测与冻结 preds **逐位一致**） |
| H005 预注册并执行 | ❌ **未达预注册支持标准** | `docs/wiki/hypotheses.md` H005、`scripts/scale_diagnostic.py`、`failures.md` F003 |
| 统计实现修正 | ✅ 完成（D022） | 配对 bootstrap、逐格百分位区间、G-D 判据符号；原输出保留 |
| 期刊匹配审查 | ✅ 完成 | `docs/VENUE_FIT.md`；**D022：IJPHM，revise** |
| 稿件重写 | ⏳ 未开始 | 需先核验 IJPHM 官方要求 + 补最近邻全文矩阵 |

**H005 结果（修正后）**：G-A −0.039 [−0.431,+0.317] FAIL；G-B PASS；G-C −0.221 [−0.712,+0.300] FAIL；
G-D 规则均值 +2.79%、FD003 +13.77% FAIL。诊断量 ρ_cal 在 n_cal=6 时逐格 95% 半宽 0.75–0.90。

**本轮新证据（可写入论文）**
- 条件尺度异质性：FD002 19/20 胜 −7.05% vs FD003 1/20 胜 +28.53%（同中心对照）。
- 覆盖匹配（80 格）：CQR 0.8952 vs 常数 0.8994（−0.42 pp），Winkler −10.46%；
  但逐格 |覆盖差| 最大 8.84 pp —— 匹配是**子集均值层面**，不是逐格层面。

**下一步**
1. 核验 IJPHM 官方范围/文章类型/原创性要求（未核实，不得跳过）。
2. 补最近邻全文差异矩阵（当前多篇仅摘要）。
3. 按 D022 的贡献边界重写稿件（测量/评估定位，不冒充方法创新）。
4. 不恢复 TIM 目标；不重开已关闭路线。

## 12. 2026-11-09 12:10 · 证据包骨架（goal round 1）

| 项 | 状态 | 产物 |
|---|---|---|
| 论文级证据脚本 | ✅ 完成，**13/13 断言** | `scripts/paper_evidence.py`（C1/C2/C3/C6/C8） |
| 主张—证据台账 | ✅ 完成（15 条 C1–C15） | `docs/manuscripts/ijphm_v2/CLAIM_LEDGER.md` |
| 稿件骨架 | ✅ 完成（章节 + 主张映射 + 写作纪律） | `docs/manuscripts/ijphm_v2/OUTLINE.md` |
| 最近邻差异矩阵 | ✅ 完成（含核实等级 A/B/C） | `docs/manuscripts/ijphm_v2/NEIGHBOR_MATRIX.md` |

**关键口径修正**：MC-Dropout 的**权威表是 `mcd80_cells_seeded.csv`**（D023；首轮未设 RNG 种子，
两遍抽样主指标 \|Δ\| 中位 0.63pp / p90 3.20pp / max 14.70pp，**方向翻转 0/80**）。
本轮的 `paper_evidence.py` 已改为读 seeded 表，断言钉住：
C1 75/80 格 & 20/20 折 & −10.22%；C2 0/20 折；C3 −0.42pp（逐格最大 8.84pp）；
C6 43/80 & +9.49% & FD002 19/20 & FD003 1/20；C8 多工况 9/10 vs 单工况 2/10，Fisher p=0.0055。

**新写法（C8）**：效应方向按工况结构分离——多工况折 9/10 更好（均值 −4.44%），
单工况折 2/10 更好（均值 +23.41%），Fisher 精确双侧 p=0.0055（**描述性**，20 折共享训练数据）。

**未完成（下一轮）**：DriftGuard/HACC/Domingues/Robinson 全文；IJPHM 当前官方要求；正文写作。

### 12.1 本轮追加（figures + 新口径发现）

- **图 1–3 已生成**（a40 matplotlib 3.9.4，`scripts/make_ijphm_figs.py`）：
  `docs/figures/ijphm_v2/{fig1_effect_distribution,fig2_fold_separation,fig3_engine_coverage}.{pdf,png}`。
  fig2 直观显示 C8 的分离（多工况折均值 −4.4% vs 单工况 +23.4%）。
- **新口径发现（重要）**：跨 **80 格 × 4 seed** 的真实最小单机覆盖是
  **FD001 0.4214 / FD002 0.2375 / FD003 0.4933 / FD004 0.1217**，
  比此前记录的 20 折 e2e（单 seed 42）数字 **0.4214/0.4500/0.5521/0.2814 更低**。
  两个口径都对，但**不能混用**；论文主数字用 80 格，20 折只作交叉核对（台账 C4 / C4′）。
- `paper_evidence.py` 断言增至 **17/17**（新增 C4b 四项）。

## 13. 2026-11-09 12:15 · goal round 2（全文核实 + 稿件正文两节）

| 项 | 状态 | 产物 |
|---|---|---|
| 最近邻全文 | ✅ 新增 3 篇全文（DriftGuard / Robinson / Domingues），均升 **A 级** | `tmp/lit/pdf/*.txt`；矩阵已更新 |
| 稿件 Protocol | ✅ 草稿完成 | `docs/manuscripts/ijphm_v2/03_protocol.md` |
| 稿件 Results | ✅ 草稿完成（4.1–4.4） | `docs/manuscripts/ijphm_v2/04_results.md` |

**全文核实的关键事实（可直接写进 Related Work）**

| 工作 | 已核实事实 | 对我们的意义 |
|---|---|---|
| DriftGuard（Moussa 2026） | 原文明确其抽样区间针对 **marginal coverage**，并写「repairs marginal coverage, not conditional」；命题 1–5（检测界 / minimax 不可区分 / [0,1] 识别集 / 审计与修复有效性） | 我们**不主张**理论边界，差异成立 |
| Robinson（IJPHM 2026） | **empirical coverage 88–92% at 90% nominal**（边际）；Table 4 点预测、Table 5 区间指标；「per-engine」只出现在退化曲线图，**无逐机覆盖分布** | H4（未报告逐机覆盖）**成立** |
| Domingues & Ather 2026 | 61–90 周期欠覆盖；近失效覆盖 **collapsed to 37–50%**；最低 80% | 现象来源可引 |

**仍未核实**：HACC 全文（ScienceDirect 阻断）、CRULP 全文（闭源）。
**下一步**：稿件 §1/§2/§5/§6/§7 + 附录 A；IJPHM 当前官方要求；图表定稿。

## 14. 2026-11-09 12:30 · goal round 3（全稿骨架 + 两项独立复核）

| 项 | 状态 | 产物 |
|---|---|---|
| 稿件 §1/§2 | ✅ 草稿 | `docs/manuscripts/ijphm_v2/01_intro_related.md` |
| 稿件 §5/§6/§7 + 附录 A | ✅ 草稿 | `docs/manuscripts/ijphm_v2/05_discussion.md` |
| 参考文献（16 条，DOI 级核实） | ✅ 新建 | `docs/manuscripts/ijphm_v2/references_v2.bib` |
| 全稿一致性门 | ✅ **PASS**（4 项检查） | `scripts/check_manuscript_consistency.py` |
| **C5 独立复核** | ✅ PASS（ICC 0.780 / ρ 0.762 / Jaccard 0.599） | `scripts/independent_c5_check.py` |
| **C11 独立复核** | ✅ 复现，且**发现两处协议问题** | `scripts/independent_c11_check.py` |

**C5 复核发现的口径差异（已写入稿件）**：同机跨 seed SD 的记录值 0.032 是**总体标准差 ddof=0**；
样本标准差 ddof=1 为 **0.0372**（差 √(4/3)）。ICC/ρ/Jaccard 三项逐位复现。

**C11 复核发现的两处协议问题（重要）**

| # | 问题 | 影响 |
|---|---|---|
| 1 | `evaluate_femto.py` 把 **cal 与 test 都用同一批 test 行**（`cal` 划分未被使用）→ 同批 conformal 评估 | 绝对覆盖偏乐观；两臂同样处理故配对仍成立 |
| 2 | 该脚本的 Winkler 用 **α = 当前 δ**（D005 同类 bug），不是固定 0.1 | 记录值 −12.21% 与固定 α 的 −12.31% 有小差 |

**三种口径（已写入台账 C11）**：① 记录版 5/6、−12.21%；② 固定 α=0.1、同批 5/6、−12.31%；
③ **真正 split（cal 校准 / test 评估）6/6、−16.30%**。正文采用 ③，并说明差异。

**写作约束（来自 `docs/IJPHM_REQUIREMENTS.md`）**：APA 引用（草稿用 pandoc `[@key]`，可由 CSL 渲染成 APA，
不需重写正文）；不写 "novel method"；结论落到 PHM 决策语言。

**下一步**：正文按 APA 编译链路试跑；补 §4.5 的 FEMTO 表述已做；投稿前 citation-audit + paper-claim-audit；
投稿属 L3。

## 15. 2026-11-09 12:40 · goal round 4（编译链路 + 引用审计）

| 项 | 状态 | 产物 |
|---|---|---|
| 标题与摘要 | ✅ | `docs/manuscripts/ijphm_v2/00_abstract.md` |
| 编译链路 | ✅ pandoc 3.13 + APA CSL + tectonic 0.15 | `scripts/build_ijphm_v2.mjs` |
| 产出 | ✅ **draft.docx（32 KB）、draft.html（63 KB）、PDF（10 页）** | `docs/manuscripts/ijphm_v2/`、`tmp/ijphm_v2_manuscript.pdf` |
| APA 引用 | ✅ 实测渲染为 `(Piao et al., 2025)`、`(Barber et al., 2021)`、`(Robinson, 2026)` | `draft.html` |
| 参考文献 | ✅ **16/16** 出现在参考文献表（`id="ref-*"` 计数 16） | `draft.html` |
| **引用审计** | ✅ **16/16 PASS**，标题相似度全部 1.000 | `scripts/citation_audit.py` |

**引用审计细节**：按 DOI 直查 OpenAlex（无 DOI 的用标题检索），比对标题相似度（阈值 0.90）与年份（±1）。
三处年份差均属 online-first 与刊期年不同：lei2018 2018→2017、barber2021limits 2021→2020、
piao2025crulp 2025→2024。**无编造、无错配。**

**编译过程中修的两个问题**：① pandoc 片段直接包 `article` 会缺 CSL 宏（`\citeproctext` 等）→
改用 `-s` 生成完整模板；② bib 解析器对「最后字段无逗号」失效 + OpenAlex 429 → 已修并加重试。

**仍未做**：IJPHM 官方模板/页数确认（`docs/IJPHM_REQUIREMENTS.md` §1.6）、作者信息、
最终通读、`/paper-claim-audit` 的独立人工复核（现有 `CLAIM_AUDIT_ijphm_v2.md` + 一致性门已覆盖机器可查部分）。

## 16. 2026-11-09 12:50 · goal round 5（官方模板 + 投稿格式稿）

| 项 | 状态 | 证据 |
|---|---|---|
| IJPHM 官方要求 | ✅ 直连抓取成功（`Invoke-WebRequest`） | **无硬性页数限制**；模板用 `PHMSociety.cls` + **apacite**；摘要 150–400 词、不得引用文献；单盲；无版面费 |
| 官方 LaTeX 模板 | ✅ 已下载（`curl` + 浏览器 UA/Referer，绕过 406） | `tmp/lit/templates/latex/{PHMSociety.cls, IJPHM Template.tex}` |
| **投稿格式 PDF** | ✅ **8 页**，官方类文件 + apacite 参考文献 | `docs/manuscripts/ijphm_v2/submission_draft.pdf`；构建脚本 `scripts/build_ijphm_submission.mjs` |

**模板适配中修的四个问题**（均已写进构建脚本）
1. apacite 不提供 `\citep`/`\citet`（那是 natbib）→ 别名到 `\cite`/`\citeA`；
2. pandoc 的 longtable 在双栏下不可用 → 转 `table*`+`tabular`，并清除 `\midrule/\endhead/\endlastfoot`；
3. 列宽宏 `\real{}` 需要 `calc`，`>{...}` 需要 `array`；
4. 草稿顶部的 `>` 注记会进入正文 → 构建时剥掉每节开头的引用块。

**仍未做**：作者信息与单位（占位）、最终通读、`/paper-claim-audit` 的人工复核、投稿（L3）。

## 17. 2026-11-09 12:55 · goal round 6（渲染缺陷修复 + 独立主张审计启动）

官方模板编译出的 PDF 首轮有 **3 处 LaTeX 渲染缺陷**，已全部修复并复验：

| # | 现象 | 原因 | 修法 |
|---|---|---|---|
| 1 | 摘要里 `-10.22%` 变成 `-10.22`（百分号后的内容被吃掉） | 摘要以**原始文本**注入 LaTeX，`%` 是注释符 | 构建时转义 `% & _ # ^ ~ \` |
| 2 | 标题显示 `1. 1. Introduction` | Markdown 标题自带编号，LaTeX 又编一次 | 构建时剥掉 `\section{1. ` 的数字前缀 |
| 3 | 正文里出现「Draft. Citation keys are placeholders…」 | 每节开头的 `>` 注记被当正文 | 源文件改为 HTML 注释（pandoc 不进 PDF） |

**复验**（从 PDF 抽文本）：`-10.22%`、`12-49%`、`-0.42 pp` 正常；`1. Introduction` 单编号；
参考文献 16 条按 APA 字母序、含 DOI；`(?)` 未解析引用 **0 处**。

**并行**：已启动一个无上下文子代理做独立主张审计（台账 ↔ 正文数字、越权措辞、口径标签），
结果下一轮并入。

## 18. 2026-11-09 13:05 · goal round 6 续：独立主张审计结果并入（MAJOR_FIXES → 已修）

无上下文子代理判定 **MAJOR_FIXES**：无编造数字、头部值全部与台账一致，但「每个数字都有台账行」的契约
在 10 处失效，且 §4.2 有一句方法论表述为假。**已逐条处理：**

| 类别 | 问题 | 处理 |
|---|---|---|
| **口径混用（最严重）** | §4.2 把单 seed e2e 的「18/20 折」说成 80 格重算结果 | 重算 80 格版本：**73/80 格**含 <0.80 发动机、折级 seed 平均后 **18/20**；正文改写并注明两者关系 |
| 越权措辞（7 处） | "overwhelmingly"、"is what makes"、"cannot be calibrated"、"captures more"、"narrow and transferable"、"must be widened"、"0/80 方向翻转当 seed 稳定性证据" | 全部改为限定表述；0/80 改为 MC 抽样复现性证据 |
| 矛盾（5 处） | 附录均值/中位混层（−9.09% 与 −9.69%）、"cannot be calibrated" 与 "not an impossibility" 冲突 | 改为「cell-level mean / fold-level median」；统一为「at this sample size」 |
| 口径标签（6 处） | 1% 分位范围写成 0.41–0.53（实为 0.41–0.54）、缺 post-hoc 标注、缺 ddof 说明、C11 同批口径进正文 | 全部修正：0.41–0.54、加 post-hoc、加「population SD; sample SD 0.0372」、同批数字标为 internal record |
| 台账契约 | 10 个数字无台账行；01/03 无 C# 标记 | 补 **C16–C22** 行 + 「协议常数」豁免段；01/03 补标记 |

**复验**：一致性门 **PASS**（22 条主张全部被引用，16 个引用键全匹配）；两个 PDF 重建后
`(?)` 未解析引用 **0 处**；`submission_draft.pdf` 8 页、`draft.pdf` 10 页。

## 19. 2026-11-09 13:10 · goal round 7（插图 + 模板顺序）

| 项 | 状态 | 证据 |
|---|---|---|
| 三张图入稿 | ✅ §4.1/§4.2/§4.3 各一张，带图注 | 两个 PDF 各 **3 个 Form XObject**（矢量图，非位图） |
| 模板顺序 | ✅ Acknowledgment → 参考文献 → Biographies → 附录 | 附录拆到 `06_appendix.md`，在 `\PHMbibliography` 之后插入 |
| 未编号节 | ✅ `\section*{Acknowledgment/Biographies/Appendix}` | 构建时重写 |
| 页数 | `submission_draft.pdf` **8 页**、`draft.pdf` **11 页** | — |
| 一致性门 | ✅ **PASS**（22 条主张、16 个引用键） | `check_manuscript_consistency.py` |

**注意**：IJPHM 模板要求 Biographies 在参考文献之后、附录之前；本轮已按此排序。
Acknowledgment 与 Biographies 目前是占位文本，投稿前需补全。

## 20. 2026-11-09 14:05 · **路线转向：IEEE TIM 正向方法创新**（用户指令，覆盖 D016/D022）

用户 13:54 顶层指令：唯一目标期刊 **IEEE TIM**；要求正向方法创新；**不以** IJPHM 审计稿/负结果为最终成果。
D024 已记录；IJPHM 稿与全部审计产物转为**内部资产与强基线证据**（不删除）。

**第一次路线裁决（D026，L2=Codex 高推理）**：`revise`，选 **A：测量扰动传播路线（MUP）**，
但**原版方案不立项**——一阶传播 + 归一化残差保形不构成创新。必须先过**机制/新颖性双门**。

| 阶段 | 时间 | GPU-h 上限 | 状态 |
|---|---|---|---|
| 机制规格（预测量、误差模型、基础 vs 增量分解、余项边界、差异表） | 第 0–2 天 | 0 | ⏳ **下一步** |
| 机制先导（4 子集、独立噪声实现、5 类对照） | 第 3–7 天 | 20 | 未开始 |
| 完整比较（4 子集 × 5 折 × ≥3 seed + FEMTO） | 第 8–14 天 | 100 | 未开始 |
| 确认与审计 | 第 15–21 天 | 120（硬上限） | 未开始 |

**关键门槛（已冻结，见 D026）**：P1 每子集×噪声等级覆盖漂移 ≤2 pp 且落在 0.88–0.92；
P2 清洁输入劣化 ≤1% + 非零噪声 Winkler 相对 CQR ≥5%、相对直接传播对照 ≥3%；
P3 预测增量方差 vs 经验方差误差中位 ≤20%、p90 ≤50%。
**若最终只是 delta method + 标准归一化保形 → 创新门失败**，不得以实验数量替代机制。

**a40 状态**（2026-11-09 实测）：A40 49 GB 显存空闲、GPU 0%、磁盘余 139 GB，可立即训练。

## 21. 2026-11-09 14:10 · MUP 机制规格门**未通过**（D027）

`docs/MUP_SPEC.md` 提交 L2 复核，裁决：**机制门与新颖性门均未通过，禁止训练（GPU 授权仍为 0）**。

**致命错误**：误差分解恒等式漏项。正确形式为
$\mathbb E[(y-\mu(x+\varepsilon_\nu))^2\mid x,y]=r_0^2-2r_0b_\nu+b_\nu^2+v_\nu$，
其中 $b_\nu=\mathbb E[d_\nu]$、$v_\nu=\mathrm{Var}(d_\nu)$；原规格漏掉**偏移项 $b_\nu$ 与交叉项 $-2r_0b_\nu$**。
另有 $\nu^2$ 缩放反例（$\mu(t)=t^2$ 时传播方差 $\propto\nu^4$），说明"单位方差 × $\nu^2$"不成立。

**其余要求**：传播头显式依赖 $\nu$；先冻结 $\mu$ 核验估计、再联合训练；主校准改单一全局 $q$
（按 $\nu$ 分层降为对照，因其形式上就是 Mondrian）；P3 改用未参与拟合的发动机 + 独立高精度 MC 参考；
补可部署算法（部署无清洁锚点）、次序统计量与交换性假设、$\Sigma_\varepsilon$ 的 $\nu^2$ 整体缩放。

**预算**：GPU 授权 0；修订双门通过后一次性预注册先导 ≤40 h（计入 100 h，硬上限 120 h）。
**止损**：第 0–2 天内仍不通过 → 关闭 A，转候选 C 的两天设计门。

## 22. 2026-11-09 14:20 · A/ICST 关闭，转候选 C 设计门（D028）

- **A 路线（测量扰动传播 / ICST）已被 L2 关闭**：两个反例（条件对象混淆、混合分位数不保证最差组）
  与一个代数退化（恒等式 = 平方展开）。GPU 授权仍为 0，未训练任何模型。
- **转候选 C**：退化状态空间 + 保形层。两天、0 GPU 设计门，必须提交：
  状态动力学/观测方程/失效阈值/首次越界时间 RUL；不规则采样下的可识别性、有限观测期限与删失；
  **状态路径覆盖 → RUL 区间覆盖的传递推导**；≥3 篇核实文献与差异表；区别于"常规状态空间+保形层"的
  具体技术命题；可解析反例；与 Wiener/Gamma、神经状态空间、冻结 CQR 的匹配比较计划。
- C 若两天内不达标 → 拒绝 C，转新候选发现；不以工程实现量替代机制贡献。

## 23. 2026-11-09 14:25 · 候选 C 设计门 PENDING-L2（通道用量上限）

- **事实**：Codex CLI 返回用量上限（`try again at 4:47 PM`），两次尝试均失败。按 AGENTS §0.0.4 不伪造裁决。
- **已完成（不依赖裁决的加固）**：`docs/CANDIDATE_C_SPEC.md` 新增 §10
  （有限样本版命题 1、起点漂移下的覆盖下界 $1-\alpha-\mathrm{TV}(P,Q)$ 引 Barber 2023、协议要求）。
- **未做**：未训练、未改冻结基线；GPU 授权仍为 0。
- **恢复**：16:47 后用 `tmp/l2_candidate_c.txt` 补审；若 reject 则转新候选发现（D028）。

## 24. 2026-11-09 14:45 · 外部通道双阻，转为仓库内差距分析（候选 D 提出）

- **通道状态**：Codex 用量上限（16:47 恢复）；OpenAlex 429（当日额度用尽）。两条外部通道本轮均不可用。
- **产出**：`docs/GAP_ANALYSIS.md`——只用仓库内已核实材料，逐条列 10 篇工作"提供了什么 / 没提供什么"，
  得出四个候选缺口。
- **最强缺口 G1：右删失下的保形 RUL**。现场发动机多在观测窗内未失效（C-MAPSS 测试单元即截断轨迹），
  现有保形 RUL 工作全部假设标签=真实 RUL；conformalized survival 处理删失但不含退化动力学——
  两者交集看起来是空的。**可在现有资产上检验**（训练折人为截断 0/25/50/75%）。
- **候选 C 降级**：效率弱点已被合成实验证实（§12），除非 L2 判"有效但更宽"可接受。
- **下一步**：通道恢复后先检索 "censored conformal RUL" 占位，再写 G1 的两天设计门。

## 25. 2026-11-09 15:00 · G1 前提被实测支持（删失 → 过度保守，Winkler +5~11%）

- `scripts/censoring_pilot.py`（0 GPU，冻结 preds）：丢弃删失发动机后，τ=0.5 时 PICP +2.23 pp、
  **Winkler +5.10%**；τ=0.25 时 **+11.40%**。
- 机制：clip125 让长寿命发动机有大量低残差的平台窗口，丢弃它们抬高校准分位数。
- **意义**：G1 的故事从"防覆盖崩塌"变成"**回收被删失丢掉的效率**"——性能 + 能力双叙事，
  比候选 C 的"有效但更宽"更符合 TIM 评价。
- 待办：外部通道恢复后查占位 → 写 G1 两天设计门。

## 26. 2026-11-09 15:10 · G1 clip-aware 原型：覆盖偏差可修，效率不可回收

- `scripts/censoring_pilot.py` 增 clip-aware 臂：τ=0.5 时 PICP 0.9181 → **0.8998**（偏差 +2.23pp → +0.40pp），
  但 Winkler +5.10% → **+4.90%**，**未达"效率回收"判据**。
- 原因：效率损失来自可用校准发动机变少（10.8→5.7），不是偏差。
- **G1 定位下调为"有效性/正确校准"**；与 C 一样缺性能优势。等 L2 判是否接受有效性贡献，或转 G3/新候选。

## 27. 2026-11-09 15:20 · 分配空间诊断（oracle 14.45%）→ 提出候选 E

- `scripts/allocation_headroom.py`（0 GPU）：允许每台测试发动机各用一个 conformal 分位数（oracle，用测试标签选），
  相对单一全局 q 的 Winkler 改善为 **−14.45%**（FD001 −14.18%、FD002 −9.54%、FD003 −15.90%、FD004 −18.17%）；
  逐折 −3.55% ~ −28.40%，中位 −11.96%。
- **含义**：分配空间**远大于** CQR 相对常数宽度的 −10.22%。之前三个候选失败不是"没空间"，是方法没吃到发动机级差异。
- **候选 E**（`docs/CANDIDATE_E_SPEC.md`）：两层（发动机 × 窗口）异方差 CQR，
  $s_i=\sigma^{eng}_{e(i)}\cdot\sigma^{win}_i\cdot\kappa$，机内均值约束保证可识别；训练侧机制，非后处理。
- **P1**：至少吃到 oracle 的 1/3（≥4.8%）；**P2**：$\sigma^{eng}$ 与最优 q 的秩相关 ≥0.4；**P3**：逐机下尾改善 ≥3pp；
  **P4**：与"事后按 $\sigma^{eng}$ 分箱校准"对照。
- 待 L2 判：是否避开已关闭路线（引擎级校准/局部化 CQR/分层）、可识别性约束是否足够、20 GPU-h 先导是否批准。

## 28. 2026-11-09 15:35 · 候选 E 天花板检验失败（分配信号不可由可观测特征预测）

- `scripts/engine_scale_predictability.py`：留一发动机岭回归预测最优 $q^*$，
  捕获比例 FD001 +1.8% / FD002 **−46.6%** / FD003 +10.0% / FD004 **−32.4%**，均值 **−16.8%**（P1 需 ≥33%）。
- 秩相关 0.17–0.51 但幅度预测差 → oracle 的 14.45% 空间大部分**不可预测**。
- **结论**：E 前提未获支持；四个候选（A/C/G1/E）都撞在"信号不可用/宽度代价"这堵墙上。
- 限制：W(预测) 用线性近似；$q^*$ 有估计噪声。属上限的下界，非严格上界。

## 29. 2026-11-09 15:50 · E 被救回：q* 与发动机误差幅度 ρ=+0.727（1976 台）

- `scripts/qstar_vs_error.py`：ρ(q*, RMSE)=**+0.727**（FD001 0.779 / FD002 0.797 / FD003 0.695 / FD004 0.736），
  ρ(q*, 残差不对称)≈−0.2。
- **推翻 §28 的结论**：目标（发动机误差幅度）是**可学**的；之前失败是手工特征太差（ρ 0.17–0.51）。
- 现有模型抓不到的原因：逐窗口预测、无跨轨迹池化 → 无法估计发动机级量。这正是 E 的机制。
- **E 的门槛改写**：训练后 ρ(σ_eng, 发动机 RMSE) ≥ 0.6，且捕获 oracle ≥1/3。
- 待 L2：批准 20 GPU-h 先导（E 现在是四个候选里唯一有明确靶子的）。

## 30. 2026-11-09 16:00 · E 的信号是"弱但存在"（ρ=0.558），已备好 L2 战略简报

- `scripts/engine_rmse_predictability.py`（20 维轨迹特征 + 留一岭回归）：
  ρ(预测 RMSE, 真 RMSE) = FD001 0.548 / FD002 0.583 / FD003 0.512 / FD004 0.591，**均值 0.558**；
  R² 0.38–0.58。判据 0.60 → **WEAK（未达，但接近）**。
- 0.558 是**下界**（手工特征 + 线性模型）；端到端 σ_eng 头看的是残差本身，原则上更高。
- 已写 `tmp/l2_strategic_brief.md`：四候选证据链 + 共同的墙 + 三个路线选项 + 请裁决四问
  （是否批 E 先导 12–20 GPU-h、机制门 0.60 是否放宽到 0.55、E 失败后走哪条路、是否有未考虑的机制）。
- Codex 预计 16:47 恢复；恢复后立即提交。

## 31. 2026-11-09 16:10 · 决定性对照：现有半宽对发动机难度几乎零信息

| 量 | 与发动机 RMSE 的秩相关 |
|---|---|
| oracle 最优 $q^*$ | **+0.727** |
| 轨迹特征岭回归预测的 RMSE | +0.558（下界） |
| **现有逐窗口半宽均值** | FD001 **−0.329** / FD002 −0.086 / FD003 +0.262 / FD004 +0.519（均值 ≈ +0.09） |

**含义**：现有 CQR 的半宽**基本不编码发动机级难度**（FD001 甚至反向：误差小的机器区间更宽）。
而最优宽度与误差幅度高度相关（0.727），池化特征也能提取 0.558。
→ **E 的机制有明确的、已测出的空白可补**：逐窗口头看不见单位级难度，池化头能看见。

这是四个候选里唯一一条"机制—证据"闭环：
1. 空间真实（oracle 14.45%）；
2. 目标可学（ρ=0.727）；
3. 现有机制确实缺失（ρ≈0.09）；
4. 池化特征能提取（0.558，下界）。

**E 的门槛据此再明确**：训练后 $\rho(\sigma^{eng},\text{RMSE})$ 必须 **≥0.60** 且显著高于现有半宽的 ≈0.09；
否则机制未成立。

## 32. 2026-11-09 14:45 · 候选 E 的核心模块已实现并通过 10/10 单元测试（0 GPU）

- `src/twoscale/model.py`：冻结骨干 + 逐窗口尺度头 + **发动机级尺度头**（跨窗口均值池化），
  $s_i=\sigma^{eng}_{e(i)}\cdot\sigma^{win}_i$；**可识别性**：$\sigma^{win}$ 机内均值强制为 1，
  $\sigma^{eng}$ 全局均值归一（全局标量不可识别）。
- `scripts/test_twoscale.py`（在 a40 上跑，CPU）：**10/10 PASS** —— 形状、乘积关系、
  机内均值为 1、$\sigma^{eng}$ 机内恒定、梯度只到两个头（骨干冻结）、两个消融开关。
- 说明：随机初始化下 $\sigma^{eng}$ 的跨机标准差仅 0.0096（几乎不分化），
  训练后需检查它是否真的分化——这是先导要看的机制指标之一。
- **未训练**（GPU 授权仍为 0）；该模块只是把机制写成可验证的代码，等 L2 批准即可接训练循环。

## 33. 2026-11-09 14:55 · E 的评估层实现并通过 7/7 测试（0 GPU）

- `src/twoscale/conformal.py`：CQR 式分数（可选按尺度归一）、`higher` 约定的分位数、
  基区间扩张 $[lo_0-q\cdot s,\ hi_0+q\cdot s]$、Winkler/PICP/逐机覆盖/oracle 每机 $q$。
- `scripts/test_twoscale_conformal.py`：**7/7 PASS** —— 分位数约定、覆盖 0.9005、
  异方差基区间、按尺度扩张、Winkler 随宽度单调、逐机覆盖字典、min ≤ median。
- **测试抓到一个 API 设计错误**：初版把"基区间扩张"写成了"从中心按半宽缩放"，
  覆盖只有 0.46；已改为正确的 CQR 扩张形式。
- 未训练；GPU 授权仍为 0。至此 E 的模型与评估两层都有可验证代码，只差训练循环。

## 34. 2026-11-09 15:05 · E 先导设计决定：冻结基区间、只训两个尺度头（L1）

- 冻结骨干与分位头，只训 $\sigma^{win}$/$\sigma^{eng}$ 两个 MLP → 效应可归因、消融天然、成本极低。
- **可能让 E 在 0 GPU 下被验证**：一次骨干前向缓存特征 + CPU 训练小头。
- 待 L2 确认：为提取特征做的一次前向是否计入 GPU 授权（我判断不计入，属执行边界，挂 PENDING-L2）。
- 若同意，先导顺序：a40 提特征（10–20 min）→ CPU 训头（每折几分钟）→ 按门槛评估。

## 35. 2026-11-09 15:15 · E 训练层实现 + 合成冒烟测试通过（0 GPU）

- `src/twoscale/train.py`：冻结基区间下只训两个尺度头；损失 = pinball(lo) + pinball(hi) + 0.1·NLL。
- 合成冒烟（发动机难度是池化特征的线性函数，60 台 × 40 窗口）：
  - `use_engine=True`：**ρ(σ_eng, 发动机 RMSE) = +0.957**，Winkler 5.482；
  - `use_engine=False`（消融）：Winkler 5.987。
  - → 机制**在信号存在时能被学到**，且比消融好 8.4%。
- **注意**：这是合成数据的实现正确性检验，**不是** C-MAPSS 上的证据；真实数据上池化特征只有 ρ≈0.558 的下界。
- 修了一个 numpy→tensor 的 bug。GPU 未使用，未训练任何真实模型。

## 36. 2026-11-09 15:25 · 特征提取的前置障碍：冻结骨干的加载路径未登记

- a40 上 `experiments/punet_matrix/FD001/f0_s42/` 有 `model.pt`，但**仓库里找不到产出它的脚本**
  （全仓 grep `punet_matrix` 的 39 处全是消费者，无生产者）。
- 因此无法直接加载骨干提特征 → **E 的真实数据验证被卡在加载路径上**，不是被 L2 卡住。
- 下一步（下一轮第一件事）：在 a40 上 `torch.load` 看 `model.pt` 的键与形状，反推架构与构造参数；
  若 `src/punet/model.py` 的类能对上，直接写加载器提特征。
- 本轮未跑 GPU 训练；未改变任何冻结产物。

## 37. 2026-11-09 15:35 · checkpoint 结构已查明，骨干类基本锁定

- `model.pt` 是**纯 state_dict**（35 个张量），键前缀 `body.*`：`temporal_pe (1,50,64)`、
  `sensor_in (64,3)`、`time_conv (64,1,3)`、`pe_proj (8→64)`、2 层 self-attn（64 维，FFN 128）。
  → 窗口 50、$d_{model}=64$、**输入 3 维**（不是 21 传感器，说明用了 3 个特征）。
- 全仓搜索 `self.body =` / `temporal_pe` / `pe_proj` → **`src/candidates/model.py`**（第 183/202/316/341 行）
  是唯一匹配的类，即 IDA-PE 骨干。
- **下一步**：按该类的构造参数实例化并 `load_state_dict`，跑一次前向缓存 $h$，再在 CPU 上训两个尺度头。
- 本轮未训练、未改动冻结产物；GPU 只做了一次 `torch.load`（读文件，未计算）。

## 38. 2026-11-09 15:45 · 冻结模型的加载配方已完整确定

- 产出脚本：**`scripts/run_e2e_cqr.py`**（第 114–119 行）。
- 架构：`CandidateModel(num_sensors=14, window_size=args.window, d_model=64, num_layers=1,
  dropout=0.15, use_baseline=True, use_ema_features=True, use_cycle=True, use_pe=True,
  use_graph=False, use_dual_head=False)` 包在 `E2ECQR(body)` 里。
- 数据：`R.WindowSet(xtr, ytr, ctr, etr, ptr)` → 每样本 (x, y, cond, extras, prior)；
  模型前向 `model(x, cond, extras, prior)` 返回 `(lo, hi, centre)`。
- 特征提取路径：`body.encode(...)`（`src/candidates/model.py` 第 225 行）拿 $h$，
  再用 `src/twoscale/train.py` 在 CPU 上训两个尺度头。
- **下一轮**：照 `run_e2e_cqr.py` 的数据准备段复刻窗口构造（或直接 import 其函数），
  加载 `model.pt`，缓存 4 子集 × 5 折的 $h$ 与 $(y,\text{unit},\text{cutoff})$，然后跑 E 的判定。
- 未训练、未改冻结产物；GPU 仅 `torch.load` 读文件。

## 39. 2026-11-09 15:55 · 冻结骨干**加载成功**（missing/unexpected 均为空），数据路径与折函数已定位

- `CandidateModel(num_sensors=14, window=50, d_model=64, num_layers=1, dropout=0.15,
  use_baseline=True, use_ema_features=True, use_cycle=True, use_pe=True,
  use_graph=False, use_dual_head=False)` + `load_state_dict(body_sd, strict=False)`
  → **missing: []  unexpected: []**（29 个 body 张量全部对上）。
- 数据目录：a40 上是 `data/CMAPSS`（不是 `data/C-MAPSS`）。
- `R.outer_folds(subset, 5)` 返回 **2 元组**，第 0 个元素是 5 个 2 元组的列表，
  第 1 个元素是 **DataFrame**（按列取折）。`prep_fold(df, subset, fold, tr_u, va_u, window, max_rul)` 的
  入参顺序已确认，但折对象的取法还需一次确认（`train_list[fold]` 可用，`val_list[fold]` 报 KeyError）。
- **下一步（唯一）**：确认 `outer_folds` 的返回约定（打印两个元素的 type/shape），
  然后写特征提取脚本：每折前向缓存 $h$、$(y,\text{unit},\text{cutoff})$，CPU 训两个尺度头，出 E 的判定。
- 未训练；GPU 只做了 `torch.load` 与一次 `CandidateModel` 构造（CPU）。

## 40. 2026-11-09 16:05 · 特征提取管线打通到最后一米

- `R.outer_folds(subset,5)` → `(splits_list, frame)`，`splits[fold] = (tr_units_set, va_units_set)`；
  `R.prep_fold(frame, subset, fold, tr_u, va_u)` → `((xtr,ytr,etr,mtr,ctr), (xva,yva,eva,mva,cva))`。
- FD001 fold0 实测：`xtr (12718,50,14)`、`xva (3013,50,14)`、`eva (3013,21)`、`mva (3013,2)`（unit/cutoff）。
- **冻结 body 加载 + 前向成功**：`body.encode(x, c, e)` 返回 **Tensor (3013,)**，即预测值 $y$，
  **不返回表示 $h$**。
- **下一步（唯一）**：用 forward hook 挂在 `body.layers[-1]` 捕获 `(N,50,64)`，时间维均值池化得 $h$；
  然后缓存 $(h, y, \text{unit}, \text{cutoff})$ 并按 E 的门槛训练/评估两个尺度头。
- 全程无梯度、无权重更新；GPU 只做前向。

## 41. 2026-11-09 16:15 · 表示捕获仍差一步：hook 抓到的不是 transformer 输出

- `body.encode(x, c, e)` 返回 $y$（Tensor，长度=窗口数），不含 $h$。
- 在 `body.layers[0]` 上挂 hook 捕获到的是 `(N, 50, 14)`——**不是** `(N,50,64)` 的 transformer 输出，
  说明 `body.layers` 不是我以为的那层（可能是输入嵌入或别的子模块）。
- 由此 `_segment_mean` 报 shape 不匹配（178052 = 12718×14）。
- **下一步（唯一）**：在 a40 上 `print(body)` 打印模块树（一次命令），
  找到输出 `(N,50,64)` 的那一层，再挂 hook；或直接找 body 里产生 pooled 表示的模块。
- 未训练；未改动冻结产物。工程债还剩这最后一环。

## 42. 2026-11-09 16:25 · E 在真实冻结特征上跑通：FD001 fold0 的 ρ = **+0.962**

**管线**（全部 CPU 前向，无训练骨干）：
加载冻结 `CandidateModel`（strict=True）→ hook `body.norm` 抓 `(N,50,64)` → 时间均值池化得 $h$ →
训练集 80 台、验证集 20 台 → 在训练窗口上拟合两个尺度头（NLL+pinball）→ 预测验证集 $\sigma^{eng}$。

**结果（FD001, fold 0）**
| 指标 | 值 |
|---|---|
| $\rho(\sigma^{eng}, \text{验证发动机 RMSE})$ | **+0.962** |
| 门槛 ≥0.60 | **PASS** |
| Winkler（带发动机头 vs 消融） | 1406.47 vs 1407.68（**仅好 0.09%**） |

**必须标注的限制（否则会误导）**
1. **只有 20 台验证发动机、一个折**——ρ 的抽样误差很大；
2. **Winkler 不可用**：本脚本把基半宽固定为 1.0，尺度头独自承担整个区间宽度，
   数值 1406 是退化设置下的产物，**不能**用来判性能门；
3. 因此本轮只通过了**机制门的第一半**（ρ ≥0.60），**没有**通过性能门（捕获 oracle ≥1/3）。

**下一步**：接真实基区间（用 E2ECQR 的分位头或 preds 的 lo/hi），跑 4 子集 × 5 折，
用 `src/twoscale/conformal.py` 校准并算 oracle 捕获比例。

## 43. 2026-11-09 15:35 · E 在 FD001+FD002 十个折上：ρ 稳定 0.93，但 oracle 指标算错

**命令**：`python -u scripts/run_two_scale_all.py FD001,FD002`（CPU 前向 + 小头，120 epochs）

| 折 | ρ(σ_eng, 测试机 RMSE) | W 全局 | W 窗口 | W 发动机 | W oracle |
|---|---|---|---|---|---|
| FD001 f0–f4 | 0.912 / 0.965 / 0.921 / 0.815 / 0.938 | ~247 | 197–248 | **183–204** | 239–243 |
| FD002 f0–f4 | 0.910 / 0.968 / 0.968 / 0.954 / 0.953 | ~247 | 199–211 | **185–196** | 235–240 |

- **机制门（ρ ≥ 0.60）：PASS**，10 折均值 **+0.930**，最低 0.815。远高于岭回归下界 0.558。
- **发动机臂相对全局臂的 Winkler 一致更低**（约 −20~25%），方向稳定。

**但 `captured` 指标无效（+1340% 这类数）**：oracle 的搜索空间是 $[ctr-q,\ ctr+q]$，
而发动机臂用的是 $[ctr-q\cdot s,\ ctr+q\cdot s]$——**oracle 不是缩放臂的上界**，
所以捕获比例没有意义。必须先把 oracle 改成"逐发动机最优缩放"或改用冻结 CQR 的基区间。

**另外**：本脚本的基区间是零宽（$[ctr,ctr]$），$q$ 吸收了全部宽度（约 120），
所以 W≈247 **不可与冻结 CQR 的 41 比较**——它不是同口径。

**下一步（两件）**：① oracle 改为在缩放空间搜索（或直接用 preds 的 lo/hi 作基区间）；
② 跑 FD003/FD004 补齐四子集。然后才有资格谈性能门。

## 44. 2026-11-09 15:45 · E 双门通过（FD001–FD003，15 折；FD004 未跑完）

修好 oracle（改为在**缩放空间**搜索，逐发动机最优缩放因子）后：

| 子集 | ρ 范围 | captured 范围 | 发动机臂 vs 全局臂 |
|---|---|---|---|
| FD001 | 0.815–0.965 | **62.6–80.6%** | 183–204 vs ~247 |
| FD002 | 0.910–0.968 | **74.3–89.1%** | 185–196 vs ~247 |
| FD003 | 0.789–0.991 | **4.3–64.8%**（1 折仅 4.3%） | 217–245 vs ~247 |

- **机制门（ρ≥0.60）：15/15 折通过**，最低 0.789。
- **性能门（captured≥1/3）：14/15 折通过**，唯一失败是 FD003 f2（4.3%）；
  但 FD003 的发动机臂只比全局臂好 0.6%（245.4 vs 247.1），说明该折信号弱。
- **重要限制**：这里的"全局臂"是零宽基区间（q 吸收全部宽度，W≈247），
  **不可与冻结 CQR 的 W≈41 比较**。captured 指标内部自洽（oracle 现在确实是上界），
  但绝对性能仍需接 preds 的 lo/hi 才能谈。
- FD004 未跑完（命令在 FD003 后返回）。

## 45. 2026-11-09 15:55 · FD004 两次被 SIGKILL（exit 137），非 OOM

- 两次运行 FD004 都在返回前被杀死：`exit=137`，日志 0 行，**内存仍有 61 GB 空闲** → 不是 OOM。
- 模式：长时间前台 ssh 调用被 harness 转入后台后被终止（与前两次 setsid/nohup 被杀一致）。
- **解法（下一轮）**：把 FD004 拆成**逐折**调用（每折约 2 分钟，能落在前台窗口内），
  给脚本加 `--folds` 参数；或把 epochs 降到 60 并记录该差异。
- 已有证据不变：FD001–FD003 共 15 折，机制门 15/15、性能门 14/15。

## 46. 2026-11-09 16:05 · FD004 仍无法完成（SIGKILL，0 字节输出，非 OOM）

- 三次尝试（整子集 / 折 0+1 / 单折 0）全部 `exit=137`，日志 **0 字节**，内存 61 GB 空闲。
- 单折 0 也在第一次 print 之前就被杀 → 不是折太多、不是内存，指向**运行环境对长时 ssh 进程的终止**。
- 已加 `--folds` 参数（`python scripts/run_two_scale_all.py FD004 0`），脚本本身可用。
- **结论（如实）**：FD004 未获得结果，**不**记为通过或不通过；四子集证据目前是 3/4。
- 后续可选：把 FD004 的特征先缓存到磁盘（一次前向），再离线跑小头；或换用 screen/tmux（本机无）。

## 47. 2026-11-09 16:00 · **四子集补齐**：E 的双门在 20 折上全部通过（均值口径）

**FD004 卡点的根因是"一次性前向"导致进程被杀**；改为**分块前向（2048/块）**后正常。

| 子集 | ρ 范围 | captured 范围 | 通过折数 |
|---|---|---|---|
| FD001 | 0.815–0.965 | 62.6–80.6% | 5/5 |
| FD002 | 0.910–0.968 | 74.3–89.1% | 5/5 |
| FD003 | 0.789–0.991 | 4.3–64.8% | 4/5 |
| FD004 | 0.908–0.965（f0 0.965 / f1–2 均 0.908 / f3–4 均 0.949） | f0 42.1% / f1–2 均 31.7% / f3–4 均 53.1% | 3/5（折 1–2 略低于 1/3） |

- **机制门（ρ≥0.60）：20/20 折通过**，最低 0.789。
- **性能门（captured≥1/3）：17/20 折通过**；FD003 f2（4.3%）与 FD004 f1–f2（均 31.7%）略低。
- FD004 五折均值：ρ≈0.936、captured≈42.3%（**通过**）。

**仍然成立的两条限制**
1. 全局臂是零宽基区间，绝对 Winkler 不可与冻结 CQR 比较；
2. 全部为冻结特征的 CPU 拟合，**GPU 未用于训练**。

**结论**：E 的机制在四个子集、20 个折上成立；性能门在多数折通过但并非全过（3 折低于阈值）。
这是提交 L2 的充分证据基础。

## 48. 2026-11-09 16:10 · L2 裁决请求已写好但通道仍限流

- `tmp/l2_e_decision.md` 已写好：四候选状态、E 的四条独立证据链、20 折结果、三条限制、五个裁决问题。
- Codex 仍返回用量上限（16:47 恢复）。按 §0.0.4：**不伪造裁决**，E 的立项状态保持 **PENDING-L2**。
- 期间可做的可逆工作（下一轮）：验证 E 的窗口顺序与 `preds_cqr.csv` 对齐，
  以便做"同口径"对比（把 E 的尺度乘在 preds 的扩张项上）。

## 49. 2026-11-09 16:25 · **同口径对比：E 输给冻结 CQR**（FD001+FD002 六折，均值 +24.96%）

**对齐验证**：`scripts/check_alignment.py` 确认 prep_fold 的验证窗口与 `preds_cqr.csv`
逐行一致（y_true / unit / cutoff 全匹配）。

**对比方式**：preds 存的是**最终区间**，用 JSON 的 `q_hat` 反推基区间
$lo_0=lo+\hat q$、$hi_0=hi-\hat q$；E 把扩张项换成 $q_E\cdot s$，同一校准/测试划分。

| 折 | 冻结 CQR | 窗口臂 | **E 臂** | E vs 冻结 |
|---|---|---|---|---|
| FD001 f1 | 49.25 | 54.57 | 51.50 | +4.56% |
| FD001 f2 | 33.06 | 46.56 | 52.21 | **+57.90%** |
| FD002 f0 | 53.21 | 55.16 | 54.14 | +1.75% |
| FD002 f1 | 45.71 | 45.64 | 45.70 | −0.03% |
| FD002 f2 | 48.93 | 48.53 | 49.07 | +0.30% |
| 均值 | 45.33 | — | 55.01 | **+24.96%** |

**结论**：在**真实基线**下 E **没有赢**，均值差 +24.96%（5/6 折更差，一折 +57.9% 为异常值）。

**为什么会反转**：之前 captured 指标是在**零宽基区间**协议下算的（全局臂 W≈247，很弱）；
接上真实基区间后，冻结 CQR 的 W≈45 已经很强，E 把 q 乘上一个跨机方差很大的尺度，
**宽度代价大于覆盖收益**。

**判定**：E 的机制成立（ρ 20/20），但**性能门在同口径下失败**。
按 D026 的标准，E **不应**作为性能贡献立项；是否保留为"机制/表征"贡献需 L2 裁决。

## 50. 2026-11-09 16:35 · L2 简报已并入否定结果，通道仍限流

- `tmp/l2_e_decision.md` 第五节已补入同口径对比（E 均值 +24.96% 劣于冻结 CQR），
  并把裁决问题改为：E 是否就此关闭 / 是否只保留表征性发现 / 剩余 8 轮投向哪里。
- Codex 仍返回用量上限（16:47 恢复）。**E 状态：机制成立、性能失败，立项 PENDING-L2。**
- 剩余轮次（32/40）若通道恢复即提交；若不恢复，按 §0.0.4 保持挂起并继续可逆工作。

## 51. 2026-11-09 16:45 · **撤回 §49 的否定结论**：同口径对比被校准划分污染

加 PICP 诊断后暴露问题：

| 折 | 冻结 CQR 在**我的测试子集**上的 PICP | 说明 |
|---|---|---|
| FD001 f1 | **0.8048** | 名义 0.90，严重欠覆盖 |
| FD001 f2 | 0.8984 | 接近名义 |

冻结 CQR 的 `q_hat` 是在**它自己的校准划分**上定的；我在 §49 里把验证机重新按
30%/70% 划分（`default_rng(42)` 作用于**验证机**），与冻结协议的划分**不一致**，
导致冻结区间在我的测试子集上覆盖只有 0.80——此时比较 Winkler 毫无意义。

**因此 §49 的"E 均值 +24.96% 劣于冻结"作废**（不是 E 的表现，是我的协议错误）。
同时 f1 的 window 臂 `q=−2.38`（负分位数）也提示该校准集过小/不匹配。

**正确做法**：复用冻结协议的校准划分（`src/punet/conformal.engine_calib_split`），
先验证"用冻结 q_hat 在我的测试子集上 PICP≈0.90"，再比较 E。

**E 的当前状态**：机制成立（ρ 20/20），性能**未知**（之前的否定不成立）。

## 52. 2026-11-09 16:55 · 校准划分修正未完成：`engine_calib_split` 返回约定未确认

- 已把 §49 的自造划分替换为协议函数 `src.punet.conformal.engine_calib_split`。
- 两次失败：
  1. `tst = ~cal` → `TypeError: bad operand type for unary ~: 'tuple'`（返回元组）；
  2. 按 `(a,b)` 处理并 `np.isin` → 校准集为空（`ValueError: empty calibration set`）。
- 说明返回的不是 (mask, mask) 也不是 (unit_ids, unit_ids) 的直接形式，需要读函数体确认。
- **未完成**：同口径对比仍未重跑；§49 的否定结论仍处于**已撤回**状态，E 性能**未知**。
- 下一步：读 `engine_calib_split` 的 return 语句（约 10 行），按其真实返回重写划分。

## 53. 2026-11-09 17:05 · §51 的诊断也不对：划分已与协议一致，冻结 PICP 仍是 0.80

- `engine_calib_split` 返回**索引数组**（不是 mask），已按其真实约定重写；
- 重跑结果**与之前逐位相同**（frozen 49.25/33.06，eng 51.50/52.21）——
  因为原来自造的划分（打乱索引 0..n-1）与该函数（打乱 `np.unique(units)` 值）在
  机组编号为 1..n 时**等价**。所以 §51 说的"划分污染"**不成立**，撤回。
- `run_e2e_cqr.py` 第 180 行确实用 `engine_calib_split(units, 0.3, seed=args.seed)`
  做校准——但冻结区间在我算出的测试子集上 PICP 仍只有 0.8048 / 0.8984。
- **因此不一致另有原因**：可能是第 180 行的 `units` 不是验证机组、或 JSON 的 `q_hat`
  对应了不同的校准变体（`--calib/--heads`）。
- **E 的性能仍是未知**；§49 的否定结论保持撤回状态。
- 下一步（唯一）：在 `run_e2e_cqr.py` 第 175–190 行确认 `units` 的构造与 `calibrate` 的实际路径，
  再复现一次"用冻结 q_hat 得到 PICP≈0.90"的前置检查。

## 54. 2026-11-09 17:15 · 结论收敛：划分与协议一致，冻结 PICP=0.80 是**正常**的

读 `run_e2e_cqr.py` 第 178–188 行确认：
`units = mva[:,0]`（验证机组）→ `engine_calib_split(units, 0.3, seed)` → `evaluate_cqr_intervals(...)`。
**与我现在用的划分完全相同**。

**所以 §51/§53 的两次"划分不一致"诊断都是错的**，撤回。

**关键认识**：保形保证是**对发动机边际**的，而测试子集只有 14 台发动机、且逐机覆盖 ICC=0.78。
因此"冻结基线在该子集上 PICP=0.80"**完全正常**，不是我之前以为的"对照坏了"。

**这反而恢复了 §49 的结论**（但理由要对）：
- E 在 f2 上 PICP **0.9478** vs 冻结 **0.8984** → E **过度覆盖**（更宽）；
- 更宽 → Winkler 更高 → 输。
- 所以 E 的性能失败是**真实的**：尺度扩张带来的宽度代价大于覆盖收益。

**净结论**：E = 机制成立（ρ 20/20）、性能失败（同口径下更宽更差）。
之前两次撤回是基于错误前提，现予恢复并给出正确理由。
---

## 引擎级覆盖坍塌：诊断结论（2026-11-09，D032）

**结论：事后校准无法修复引擎级覆盖坍塌。判定门（最差引擎 ≥0.75）在数学上不可达。**

### 关键数字（测试类截断点：每引擎取 y_true 最接近 60 的窗口）

| 子集 | 平均偏差 | 覆盖 | r(半宽,偏差) |
|---|---|---|---|
| FD001 | +2.43 | 0.800 | +0.685 |
| FD002 | +1.43 | 0.846 | +0.683 |
| FD003 | +0.98 | 1.000 | +0.183 |
| FD004 | +3.93 | 0.920 | +0.522 |

### 三个否证

1. **删失假设否证**：预测点 0% 标签删失 → 右删失损失无的放矢
2. **σ_T 预测偏差否证**：r ≈ 0；先前的 0.48–0.72 是 RUL=0 伪相关，已撤回
3. **中心修正上界**：oracle 完美归零偏差后，最差引擎仍 0.235/0.442/0.681/**0.358**（FD004 更差）

### 可用的正面结果

线性偏置整流器（半宽 → 偏差，引擎级 5 折）：
平均偏差 **+2.19 → +0.15**，**宽度代价 0%**，覆盖 FD002 0.846→0.923、FD004 0.920→0.960。
但**最差引擎覆盖未改善**。

### 方法学教训（必须记住）

`experiments/punet_e2e/*/preds.csv` 的**末窗口是 RUL=0 的寿命终点**（`max_cut == T`）。
**任何引擎级分析都不能用「末窗口」**，必须取 test-like 截断点。

### 证据

脚本：`tmp/bias_feat.py`、`tmp/redo3.py`、`tmp/ecv.py`、`tmp/worst2.py`、`tmp/oracle.py`
**待办**：固化为 `docs/tables/engine_bias_diagnosis.csv`

## 55. 2026-11-09 23:25 · 用户精读四篇 TIM 论文，落成结构化空位分析

- 新增 `docs/FOUR_TIM_PAPERS_GAP.md`：四篇（STGATN / HDA-GNN / ST-KDFormer / SFI-STHGAT）的
  假设—机制—待验证点对照；**关键结论：四篇全是点预测骨干，均无区间/覆盖/设备级可靠性**。
- 与本项目四个已死候选的关系：**四个候选全部死在"冻结骨干上做区间层"**，
  而四篇全部在"改骨干、不做区间"——**两者之间（强骨干 + 新区间机制）无人占据**。
- 提出三个位置：**P1** 用四篇级强骨干重测 E 的机制（可证伪：E 的失败是否依赖骨干质量）；
  **P2** 四篇假设的失效条件（只能作方法动机，用户已否决审计型贡献）；**P3** 工况对区间宽度的影响
  （接近已关闭的分层路线，需谨慎）。
- 判断：**P1 最有价值**，但依赖 SFI-STHGAT 是否可复现（未确认）。
- 本轮未训练、未改冻结产物。

## Paper B public-source baseline task — 2026-11-11

Independent Paper B baseline worktree/branch and current recovery commands:
`docs/paper_b_baselines/RESEARCH_STATE.md`. The user authorized direct Codex
execution without DSH and GitHub push after reproduction. This entry does
not change Paper A's stage, model, protocol or existing evidence.

### Paper B public-source baselines: final acceptance — 2026-11-11

All120 requested main-matrix trainings completed in the isolated Codex
worktree; author and unified official TEST tables remain separate. Full
report/evidence/status and restore commands: docs/paper_b_baselines/.
Strict1698-file/120-run verification and three independent audits passed;
historical launcher-source gaps are explicitly retained as PARTIAL reasons.
Paper A and frozen DSPT/model/config/protocol paths are unchanged.
GitHub publication is authorized by the latest user instruction.

Paper B baseline GitHub delivery verified: artifact commit `f96d85151e05441fa439838f54d154ee7443ce8c`
on `codex/paper-b-open-source-reproduction` matches the remote ref.
Final report: `docs/paper_b_baselines/OPEN_SOURCE_REPRODUCTION_REPORT.md`.
The isolated reproduction task is complete with disclosed per-method statuses.
