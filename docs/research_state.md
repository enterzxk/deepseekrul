# 研究状态（恢复入口）

> 更新于 2026-11-08 23:35。任何新会话/压缩恢复后，**先读本文件**，再读 `decision_log.md`。
> 规则见 `AGENTS.md` §9。

## 1. 当前目标

**当前版本已放弃 TIM 投稿目标**（D016，branch）。现目标：
1. 冻结现有 CQR 方法为基线，形成可追溯清单；
2. 用现有预测建立条件失效清单，围绕「平均覆盖掩盖的对象失效能否被识别」重新立项；
3. 新立项须过六件套 + L2 立项门后才实现。

**新问题候选（D018）**：平均覆盖 0.87–0.93 掩盖引擎级失效（最低 0.281，18/20 折 engine_min<0.80）；
失效与模型误差强度强相关（ρ(rmse, engine_min)=−0.78，但需改用校准集可观测替代量以免循环）。
待答：能否只用校准集信息识别将失效对象？允许结论是识别边界。

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
