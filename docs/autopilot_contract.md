# 自动科研执行契约（Autopilot Contract）

> 规则来源：`AGENTS.md` §0.0（停止权收口到 L3）。本文件只做**可执行**的落地：当前状态、下一步队列、出口断言、工具链。
> 新建/恢复会话：先读 `docs/research_state.md` → `docs/decision_log.md` → 本文件，然后**直接执行下一步队列**，不请示用户。
> 2026-11-08 经独立审查（L2-DEGRADED）修订：队列重排、补取数前置、收紧第 4 级边界。

## 1. 当前状态（2026-11-08）

| 阶段 | 状态 | 依据 |
|---|---|---|
| A–E | 完成 | `docs/wiki/progress.md` |
| F 迭代 | 进行中；远端 80 格，**本地 0 格** | `ssh a40` 实测 80 个 `preds_point.csv`；本地 `experiments/` 下 0 个 csv |
| G 论文 | 进行中，PDF 已编译；数字口径待重核 | `docs/manuscripts/` |

## 2. 下一步队列（按序自动执行，无需请示）

| # | 动作 | 前置 | 判定门 / 验收 | 预算 |
|---|---|---|---|---|
| 0 | 从 a40 取回 `experiments/punet_matrix/**/preds_*.csv` | 无 | 本地 preds 文件数 = 远端（实测 80） | ≤5 min |
| 1 | 补 FD004 f4 s13 | 队列 0 | 矩阵 40/40 齐；单格日志无 NaN、有 `preds_point.csv` | a40 单格 ≈20 min |
| 2 | 重建 `docs/tables/`（`table_all_methods.py`） | 队列 0–1 | `cells_all.csv` 等表生成，行数 = 格数 | 本地 ≤10 min |
| 3 | 核对论文数字并同步 `research_state.md` §3.5/§3.6/§7 | 队列 2 | `verify_paper_numbers.py` 全通过；三处口径与 D005 一致 | 本地 ≤10 min |
| 4 | 训练侧"尺度-残差耦合"假设 H004 取舍 | 队列 3 | L2 通道给出取舍；不可用则 `PENDING-L2` + 保留双分支 | 1 次决策 |
| 5 | 若 H004 立项：重训 40 格 × 1 seed | 队列 4 判"做" | 判定门：覆盖对齐胜率 ≥ 75/80 **且**中位改善不劣于 −9.1%；不达即回退、不改主线 | a40 ≈2 h；失败分支写 `failures.md` |
| 6 | 参考文献去占位 + BibTeX 核实 | 队列 3 | 0 undefined citation，逐条有 DOI/arXiv | 本地 |
| 7 | G.7 整稿审阅 | 队列 3、6 | 审阅报告存在，问题项有处理结论 | 本地 |
| 8 | G.8 翻译（如需英文稿） | 队列 7 | 英文稿与中文稿数字逐项一致 | 本地 |
| 9 | 重编译 + 结构断言 + `SUBMISSION_CHECKLIST` | 队列 7–8 | `node scripts/build_manuscript.mjs` 7 章断言通过；清单缺口清零 | 本地 |

### 2.2 进度（2026-11-08 实测）

- 第 0 步 ✅：本地 preds 80 个（15 MB，经 scp 取回）
- 第 1 步 ✅ 无需执行：80/80 格已齐，原 "79/80" 记录过期
- 第 2 步 ✅：`docs/tables/cells_all.csv` 80 行
- 第 3 步 ✅：`verify_paper_numbers.py` **17/17 PASS**（论文数字与工件一致）
- 下一步：第 4 步 H004 取舍（`L2-DEGRADED`：独立 subagent）

### 2.1 顺序理由（不要随手调换）

- **0 必须在最前**：本地 `experiments/` 下 0 个 csv（`d0ad488` 清理所致），数据只在无 git 的 a40 上。
  不先取数，第 2 步会在 10 分钟内失败并白白升级一轮。
- **1 先于 3**：补第 40 格会改变矩阵基数，所有聚合数字随之变化；先核对再补格等于核对废数据。
- **5 有判定门才算实验**：无判定门的"重训看看"违反 §14，禁止启动。

## 3. 阶段出口断言（不通过就地返工，不提问）

| 阶段 | 断言 |
|---|---|
| C | `idea_report.md` 含 "## Part 3"，且实验大纲有判定门与预算 |
| D | `implementation.md` 存在，校验脚本通过，编码清单逐项有结论 |
| E | `docs/dev_log.md` 存在，`src/` 可导入，最小 smoke 通过 |
| F | 每个实验格有配置 + 判定门 + 结果文件；失败机制写入 `docs/wiki/failures.md` |
| G.0–G.6 | 章节正文非空，引用可解析，数字与 `results_summary.md` 一致 |
| G.7 | 审阅报告存在，问题项有处理结论 |

## 4. 停机白名单

仅 `AGENTS.md` §0.0.1 五类（含可操作判据与"归类不明→继续"兜底）。其余一律继续，并在阶段总结中说明取舍。

## 5. 留痕

- 决策 → `docs/decision_log.md`（含 `L2-DEGRADED` / `PENDING-L2` 标注）
- 真实停机 → `docs/stop_log.md`（一行一条）
- 阶段推进 → `docs/research_state.md` + `docs/wiki/progress.md`

## 6. 变更记录

- 2026-11-08 v1：建立（对应 D006）。
- 2026-11-08 v2：独立审查后修订——队列加第 0 步取数、1 前置 3、`build_manuscript.ps1`→`.mjs`、
  补 G.7/G.8/清单收口、重训补判定门与卡时预算（对应 D007）。

## 7. 本地工具链（实测，2026-11-08）

### 7.1 必须用绝对路径的运行时

PATH 上的 `python` 是 Microsoft Store 占位符（`...\WindowsApps\python.exe`），静默退出码 1；
`py`、`node` 均不在 PATH。**不要用裸 `python` / `node`**，用：

| 用途 | 绝对路径 |
|---|---|
| Python 3.12.14 | `C:\Users\20562\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe` |
| Node | `C:\Users\20562\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe` |

（由 `load_workspace_dependencies` 返回；路径变化时重新调用该工具获取，不得猜测。）
该 Python 自带 numpy / pandas / openpyxl / python-docx / python-pptx / Pillow / lxml，无 torch。

### 7.2 数据只在 a40（已核实，非推测）

- 本地 `experiments/`：**0 个 csv**；`docs/tables/` 不存在。
- `git show d0ad488`：3579 files changed，−535,963 行（清理提交）。
- a40 实测可达（`ssh -o BatchMode=yes` 成功），`~/deepseek-rul` 下 80 个 `preds_point.csv`。
- 因此 `verify_paper_numbers.py` 现在必然 `FileNotFoundError`；"论文数字 16/16 通过"在重建表之前**不可复现**。

### 7.3 长任务

远端 `a40`：无外网、无 git，Python 在 `/home/yndx/miniconda3/envs/a40_env/bin/python`。
训练/评测一律后台作业 + `job_output(wait)` 收口。

### 7.4 L2 通道实测（2026-11-08）

| 级别 | 通道 | 实测结果 |
|---|---|---|
| 1 | `subagent_codex` | ❌ 激活成功但 turn 阶段 product-error（累计 3 次） |
| 2 | `codex` CLI |  ✅ **2026-11-08 21:50 恢复** —— 路径 `%LOCALAPPDATA%\OpenAI\Codex\bin\f71a721e0e6193a9\codex.exe`，经 `tools/l2_ask.ps1` 调用（UTF-8 stdin，`exec --skip-git-repo-check -C <cwd> -`）。实测约 90 s（含 5 次 WebSocket→HTTPS 回退重连）。**当前首选 L2 通道** |
| 3 | 独立 `subagent` | ✅ 可用 —— **当前实际 L2 通道**，结论标 `L2-DEGRADED` |
| 4 | DSH 自决 | 仅限可逆且可验证的取舍；路线/新颖性/阶段门一律 `PENDING-L2` |

结论（2026-11-08 21:50 更新）：level 2 已恢复，L2 决策走 `tools/l2_ask.ps1`，**不再标 L2-DEGRADED**。
level 1 仍不可用（`subagent_codex` 的 app-server 在首次可重试网络错误时被判为终局失败，
根因见 `docs/decision_log.md` D008）。
