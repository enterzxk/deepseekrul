# 交接文档 V2 —— 新会话从这里开始

> 生成 2026-11-09 01:35。**本文件取代 `docs/HANDOFF.md`（2026-11-08 版，已过期）。**
> 适用模式：Auto Research OS（按 AGENTS.md §0.0 自动推进，只有 L3 五类才停）。
> 旧文件保留不删，但其中「当前处于有界投稿就绪修订阶段」与「75/80 为主证据」两条**已作废**。

---

## 0. 一句话状态

**当前版本的 IEEE TIM 投稿目标已被 L2 裁决放弃（D016，decision=branch）。**
现有 CQR 方法冻结为基线；新方向是「平均覆盖掩盖的引擎级失效能否被识别」。
不存在「补几个实验就能投 TIM」的路径。

---

## 1. 恢复顺序（严格按序，不要跳）

1. `AGENTS.md`（尤其 §0.0 停止权 / §0.5 L1-L2-L3 / §16 不可改动边界）
2. `docs/research_state.md` ← **权威状态**，§1 目标、§4 队列、§9 缺口清单
3. `docs/decision_log.md` ← **D001–D018**，已决事项不得重开
4. `docs/results_summary.md` ← 权威实验记录（含已撤回条目）
5. `docs/audit_closeout.md` ← 口径审计
6. 本文件

> `research/agent-state/state.json` 只提供任务队列与幂等账本，**不复制 docs 里的数字**。

---

## 2. 环境事实（2026-11-09 实测，别猜）

### 2.1 两台远端，用途不同

| 别名 | hostname | 用途 | 实测 |
|---|---|---|---|
| `a40`（亦作 `yndx-a40`） | `YNDXTEA0034` | **论文 A 全部实验产物**：`~/deepseek-rul/experiments/punet_matrix` 80 格 preds、`punet_e2e`、`femto` | ✅ 80 个 `preds_point.csv` |
| `lab3090`（亦作 `3090`） | `user-X11DPi-N-T` | **论文 B（点预测/span pooling）**：`/home/user/rulb` | ✅ 60 个 `TESTCACHE` csv |

**本地仓库不存实验产物**，只在远端。

### 2.2 a40 调用模板（缺 LD_LIBRARY_PATH 会 GLIBCXX 报错）

```bash
ssh -o BatchMode=yes a40
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python <script>
```
- a40 **无外网、无 git**

### 2.3 本地 Windows 绝对路径（裸 `python`/`node` 是 Store 占位符，静默失败）

| 用途 | 路径 |
|---|---|
| Python | `C:\Users\20562\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe` |
| Node | `C:\Users\20562\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe` |
| pandoc | `C:\Users\20562\AppData\Local\Pandoc\pandoc.exe` |
| Codex CLI | `%LOCALAPPDATA%\OpenAI\Codex\bin\f71a721e0e6193a9\codex.exe` |

### 2.4 L2 通道（已修复，别再走降级）

`subagent_codex` **坏的**：它 spawn `codex.js app-server --stdio`，而 app-server 在首次**可重试**网络错误
（`{"message":"Reconnecting... 2/5","willRetry":true}`）时被插件判为终局失败（`stage: turn`）。

**可用通道**：`tools/l2_ask.ps1`
```powershell
powershell -NoProfile -File .\tools\l2_ask.ps1 -PromptFile .\tmp\q.txt -WorkDir <cwd> -TimeoutSec 900
```
要点：prompt 必须以 **UTF-8 原始字节**写 stdin（PowerShell 默认编码会报 `input is not valid UTF-8`）。
实测单次 90–200 s，**必须后台跑**（`run_in_background`）+ `job_output` 收口。
也可用 `mcp__codex__codex`（走本机 CLI 0.167，读 `~/.codex/config.toml`，模型 `gpt-6.1-sol`）。

---

## 3. 已确立的事实（不要再验证）

### 3.1 主结果——**注意单位**

| 口径 | n | 结果 |
|---|---|---|
| **折级（推荐主表）** | **20 单元**（4 子集 × 5 折，seed 先平均） | **19/20 胜、均值 −9.09%、中位 −9.69%** |
| 格级（附录明细） | 80 格 | 75/80、−9.09%、−10.75% |

**80 格不是 80 个独立证据**：4 seed 共用同一 5 折划分。折内 seed 方差占 **46.8%**，
折内极差中位 6.13 pp / 最大 36.63 pp，仅 **16/20** 折四 seed 同向。

**进一步限制（Codex D015）**：20 折也**不是** 20 个独立样本（交叉验证训练集重叠），
只能描述性陈述，**不得套独立样本置信区间或显著性检验**。

### 3.2 覆盖是匹配的（这条救了 19/20 主张）

单 seed e2e，CQR vs 点基线：

| 子集 | CQR picp | 点基线 picp | 差 | CQR pinaw | 点基线 pinaw |
|---|---|---|---|---|---|
| FD001 | 0.9078 | 0.9061 | +0.17pp | 0.26 | 0.25 |
| FD002 | 0.8721 | 0.8757 | −0.36pp | 0.27 | 0.30 |
| FD003 | 0.9262 | 0.9389 | −1.27pp | 0.25 | 0.29 |
| FD004 | 0.8763 | 0.8739 | +0.24pp | 0.24 | 0.27 |

差 ≤1.3 pp，CQR 宽度 3/4 子集更窄。**该口径不需要对齐**，可直接部署。
但**不能**把这个覆盖匹配外推到四 seed 的 19/20 结果。

### 3.3 三个口径必须分清（这是全项目最大的坑）

| 口径 | n | 比较对象 | FD003 | FD004 | 状态 |
|---|---|---|---|---|---|
| 未对齐（格级） | 80 | CQR vs 常数宽度 | −6.49% | −8.22% | ✅ 可用 |
| 未对齐（e2e） | 20 | CQR vs 点基线 | −8.83% | −6.98% | ✅ 可用 |
| **覆盖对齐** | 40 | CQR vs 常数宽度 | **−1.6%** | **−0.8%** | ❌ **已从论文证据体系删除**（D015） |

**「FD003/FD004 接近打平」只在第三个口径成立，而该口径已删除。** 正文不得再出现这句话。

### 3.4 σ_T 是冗余的，不是有效机制

置换 σ（A4）**重算结果：均值 +0.25%，中位 +0.93%，11/20 折变差**（D017）。
FD004 甚至变好 3.02%。

`results_summary.md` §1 曾写「20 折全部退化 +40%/+43%/+50%/+10% → σ 携带信息，机制成立」——
**该行已划线撤回**，正确结论是 **σ_T 冗余**。§24 的「+1.1%、62% 折」同样不可复现。

### 3.5 真正的失效在引擎级（这是新方向的入口）

产物 `docs/tables/failure_inventory.csv`（20 行）：

| 子集 | 平均覆盖 | 引擎级最小 | 引擎 5% 分位 |
|---|---|---|---|
| FD001 | 0.908 | **0.690** | 0.706 |
| FD002 | 0.872 | **0.529** | 0.660 |
| FD003 | 0.926 | **0.714** | 0.734 |
| FD004 | 0.876 | **0.501** | 0.647 |

- **18/20 折存在 engine_min < 0.80**
- 最坏 **FD004 f2：折均值 0.829，最差引擎 0.281**
- `ρ(picp, engine_min) = +0.725`，`ρ(rmse, engine_min) = −0.783`
- ⚠️ **循环性警告**：rmse 与 engine_min 都是测试集导出量，该相关**不能**当作「可用 rmse 预测失效」

### 3.6 跨域

FEMTO 轴承 **5/6 胜、均值 −12.2%**（唯一败绩 Bearing2_2：0.75 vs 0.69）。

---

## 4. 已关闭路线（有判定门与证据，**不要重试**）

宽度放大（a=1 最优）、上下文中心修正（不可跨引擎迁移）、非对称校准（最优=对称）、
分层/Mondrian（全更差）、引擎级校准（等价于放宽区间）、seed 集成（n=4 退化）、
局部化 CQR（80 格无收益，FD003 +11.7%）、**H004 引擎-阶段覆盖正则（未过判定门，D013 关闭）**、
十种 σ 变换/分层/加权方案。

证据：`docs/tables/{width_scaling,d002_centre,asymmetric,mondrian,engine_vs_inflation,ensemble_scaling,cells_all}.csv`

---

## 5. 当前 L2 决定：D016（branch）

> **放弃当前版本的 IEEE TIM 投稿目标，将现有方法冻结为基线，重新立项具有明确测量学贡献的研究问题。**

**否决的三条**：A 保留 TIM（不能以目标期刊倒推当前工作有贡献）、
B 降级期刊（收窄主张不能自动满足他刊贡献要求；材料留作应用稿候选）、
D 停止投稿（仍有基线价值，无依据永久停止）。

**D016 的 8 项下一步，完成度：**

| # | 内容 | 状态 |
|---|---|---|
| 1 | 记录决策、关闭当前版本 TIM 路线 | ✅ D016 |
| 2 | 冻结代码/配置/划分/seed/预测/指标对应关系 | ⏳ **待做** |
| 3 | 审计两组置换 σ 记录 | ✅ D017 |
| 4 | 用现有预测建条件失效清单 | ✅ D018 |
| 5 | 围绕新问题形成立项 | ⏳ **待做** |
| 6 | 核实 ≥3 篇最近邻 + 六件套 | ⏳ **待做** |
| 7 | 独立新颖性/方法/实验审查 → 过 L2 立项门 | ⏳ 待做（依赖 5、6） |
| 8 | 现有材料留作内部技术报告与基线 | ✅ 已在做 |

---

## 6. 新问题候选（D018，待立项）

> **平均覆盖 0.87–0.93 掩盖了引擎级失效（最低 0.281，18/20 折 engine_min<0.80）。
> 能否只用校准集信息（不含测试标签）识别出将要失效的对象？识别依赖哪些可检验条件？**

- **允许的结论是识别边界**（如「n_calib < X 或轨迹长度 < Y 时不可识别」），不预设能修好
- **与已关闭方向的区别**：不是再做一次校准（引擎级校准/Mondrian/宽度缩放全已证伪），
  而是**先判定可识别性**；不可识别本身是可发表的边界结论
- 六件套要求见 AGENTS.md §11

---

## 7. 我这轮发现并纠正的错误（新会话不要重犯）

| # | 错误 | 教训 |
|---|---|---|
| 1 | **G10 假阴性**：我用 `\\cite\{` 正则，匹配不到 `\\citep{`，误报「正文 0 处引用」 | 实际 **11 处 `\\cite|\\citep|\\citet`、18 个唯一 key、`references.bib` 18 条**，bibtex 跑通（`.blg` 显示 IEEEtran.bst，`.log` 无 undefined）。**正则要覆盖 natbib 变体** |
| 2 | **decision_log 编号撞车**：两条并行会话同时写，出现 `D001×2`、`D008×2`、`D009×2` | 我的四条已重编为 **D012–D015**（内容未改）。**写决策前先 grep 现有编号** |
| 3 | **results_summary §1 的 A4 数字是错的**（+40~50%），与 §25/§26 的 +0.2% 矛盾 | 重算 +0.25%。**同一文件内同一实验出现多个数字时，以能从原始 JSON 重算的为准** |
| 4 | 把 **80 格当 80 个独立证据** | 折内 seed 方差占 46.8%。主表用 20 单元 |
| 5 | 一度把「FD003/FD004 接近打平」写进结论 | 那是**已删除的覆盖对齐口径**；未对齐口径下它们改善 6–9% |
| 6 | 早期在**官方 test** 上判定 wshift / instnorm 两个方向成败 | 已违反口径纪律；这两个方向已停止。**不得再用 test 做方法选择** |

> 通用教训（沿用旧 HANDOFF §4）：**自己写的验证脚本会继承自己写分析脚本时的错误。独立审计不可省。**

---

## 8. 不要做的事

1. 不要重跑 80 格主实验（产物在 a40）
2. 不要重试 §4 的已关闭路线
3. 不要用单点 Winkler 比较不同覆盖的方法（必须固定 α）
4. 不要把 H004 写成机制贡献
5. 不要在窗口级报告相关性（引擎才是独立单位）
6. 不要声称普遍优于 MC-Dropout（仅 10 格，FD002 上更差）
7. **不要把「覆盖对齐接近打平」写回正文**（D015 已删）
8. **不要把 80 格当独立样本做显著性检验**（D015）
9. **不要宣称 σ_T 是有效机制**（D017 已证伪）
10. 不要 force-push 或改写历史；结论被推翻只追加「已撤回」

---

## 9. 文献缺口（投稿前必须补，但当前版本不投）

- **CRULP**：IEEE TIM，Piao/Huang/Tsung，DOI `10.1109/TIM.2024.3522678`，OpenAlex 79 引。
  DL RUL + **UACQR**（集成 CQR + cross-conformal），基于**认知不确定性**。**`references.bib` 未收录。**
- **IJPHM 2026**：有把 CQR 用于全部四个 C-MAPSS 子集的工作（`papers.phmsociety.org` article 4724）——
  并行线提到但**未独立核实**，引用前必须查证。

---

## 10. 分支与提交

- 分支 `punet-interval`
- 本轮起点 `27dba0d`（旧 handoff），本轮新增 D012–D018 + `docs/audit_closeout.md` + `docs/tables/failure_inventory.csv` + `tools/l2_ask.ps1`
- 远端已同步（本地 `git status` 干净后 push）

---

## 11. 给新会话的第一条指令建议

> 读 `AGENTS.md` → `docs/research_state.md` → `docs/decision_log.md`（重点 D012–D018）→ 本文件。
> 然后执行 D016 第 **2** 项：冻结基线清单（代码/配置/划分/seed/预测文件/指标实现的对应关系），
> 产出 `docs/BASELINE_FREEZE.md`，逐项标注「本地可复算 / 仅远端 / 不可复算」。
> 接着做第 **5、6** 项：把 D018 的新问题写成六件套立项草案，文献部分用
> `semantic-scholar` / `openalex` 核实最近邻。
> 卡住走 L2：`tools/l2_ask.ps1`（见 §2.4），**不要用 `subagent_codex`**。
