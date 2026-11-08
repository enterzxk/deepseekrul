# 交接文档：C-MAPSS RUL 保形区间论文

> 生成于 2026-11-09。新会话请按顺序读完本文件，再读 `docs/research_state.md` 与 `docs/decision_log.md`。
> **不要重跑已完成实验，不要重开已决事项。**

---

## 0. 一句话状态

论文技术上完整（6 页、7 表 3 图、16/16 数字断言通过、18 条文献全部核实），
但 **Codex（L2）裁决为 revise，不投**：贡献与同刊前人工作 CRULP（IEEE TIM 2024）差异不足。
当前处于「有界投稿就绪修订」阶段，5 项修订已完成 0 项。

---

## 1. 已确立的事实（不要再验证，直接用）

### 1.1 主结果（80 格 = 4 子集 × 5 折 × 4 seeds）

| 指标 | 值 |
|---|---|
| CQR vs 常数宽度（覆盖对齐 0.90，α 固定 0.1） | **75/80（94%）** |
| 均值 / 中位 | **−9.1% / −10.8%** |
| bootstrap 95% CI | **[−10.6%, −7.4%]** |
| 逐子集 | FD001 −7.6%、FD002 −14.1%、FD003 −6.5%、FD004 −8.2% |
| 名义水平（20 折） | 19/20，中位 −8.7% |
| 跨域 FEMTO | 5/6，均值 −12.2% |

### 1.2 关键诊断

- 边际覆盖 ≈0.90，但**每引擎最小覆盖 0.50–0.71**；80–100% 的折里最差发动机 <0.80
- ρ(半宽, |误差|)：**窗口级 +0.335（80/80 为正）**，**引擎级 +0.086（48/80 为正）**
  → 宽度能追踪同引擎内的时间变化，**几乎不能区分哪台机器更难**
- MC-Dropout 只在 **10/80 格**：整体 57.8 vs 常数 44.4；FD003 灾难性（83.0 vs 40.9）；
  FD002 反而更好（56.7 vs 60.9）

### 1.3 已关闭的路线（有预注册判定门与证据文件，**不要重试**）

| 路线 | 结论 | 证据 |
|---|---|---|
| 宽度放大 | a=1 最优 | `docs/tables/width_scaling.csv` |
| 上下文中心修正 | 不可跨引擎迁移 | `docs/tables/d002_centre.csv` |
| 非对称校准 | 最优=对称 | `docs/tables/asymmetric.csv` |
| 分层/Mondrian | 全部更差 | `docs/tables/mondrian.csv` |
| 引擎级校准 | 等价于放宽区间 | `docs/tables/engine_vs_inflation.csv` |
| seed 集成 | n=2 有效，n=4 退化 | `docs/tables/ensemble_scaling.csv` |
| 局部化 CQR | 80 格上无收益 | `docs/tables/cells_all.csv` |
| **H004 引擎-阶段覆盖正则** | **未过判定门**（Winkler 劣化 9–21%，分组惩罚无增量） | `docs/wiki/experiments.md` §E-H004 |

---

## 2. 现在要做的 5 件事（Codex 修订清单，按依赖排序）

### ① 补引 CRULP 并做差异表（最高优先级）

**CRULP: Reliable RUL Estimation Inspired by Conformal Prediction**
- IEEE TIM，Piao / Huang / Tsung，DOI `10.1109/TIM.2024.3522678`，OpenAlex 被引 79
- 摘要要点：DL RUL 预测器 + **UACQR**（集成 CQR + cross-conformal），
  基于**认知不确定性**构造区间，两个航发数据集
- **当前 `references.bib` 没有这一条** —— 漏引同刊直接竞争工作是致命伤

另需查证：Codex 提到 2026 年 IJPHM 有把 CQR 用于全部四个 C-MAPSS 子集的工作
（`papers.phmsociety.org` article 4724），我未独立核实，**新会话必须先核实再引用**。

### ② 把 MC-Dropout 结果重新定位为差异化贡献

现有叙事是「我们比常数宽度好」。更强的是：
**「同刊主流路线（CRULP）用认知不确定性构造区间；我们检验了这个尺度假设，发现它在
FD001/FD003 上比完全不条件化还差，并给出替代方案（学到的条件半宽）」**

这不是"我们也用了 CQR"，而是"我们证伪了前人依赖的尺度选择"。

### ③ MC-Dropout 补到 80 格（约 1 小时 GPU）

现有 10 格不足以支撑②。补完才能把"认知不确定性不是有效尺度"写成结论而非观察。

### ④ 独立 paper-claim-audit（非自证）

**这是科学阻塞项，不是机械检查。** 理由见 §4 的评分 bug 教训：
我写的 16/16 断言与出 bug 的分析脚本同源，能证明一致性但会保留同一错误。

可用技能：`paper-claim-audit`（已装）。

### ⑤ 关闭 H004 表述，写入 `docs/wiki/failures.md`

H004 的分组惩罚项无增量贡献，增益来自重新训练宽度头本身，且 Winkler 劣化 9–21%。
按 §12 记录失败机制，不升级为机制主张。

---

## 3. 环境与命令（实测，别猜）

### 3.1 远端 a40（实验都在这里）

```bash
ssh -o BatchMode=yes a40
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH   # 必须，否则 sklearn 报 GLIBCXX
/home/yndx/miniconda3/envs/a40_env/bin/python <script>
```
- 无外网、无 git
- 实验产物在 `~/deepseek-rul/experiments/`（`punet_matrix` 80 格、`punet_e2e`、`femto`）
- **本地仓库没有这些产物**（清理时删掉了旧代次，当前证据只在 a40）

### 3.2 本地 Windows

| 用途 | 绝对路径 |
|---|---|
| Python | `C:\Users\20562\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe` |
| Node | `C:\Users\20562\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\node\bin\node.exe` |
| pandoc | `C:\Users\20562\AppData\Local\Pandoc\pandoc.exe` |
| tectonic | `C:\Users\20562\AppData\Local\Temp\tectonic\tectonic.exe` |

**不要用裸 `python` / `node`**（PATH 上是 Store 占位符，静默失败）。

### 3.3 两条关键命令

```powershell
# 编译（含 7 章结构断言、摘要同步、表题提升）
& "<node>" scripts\build_manuscript.mjs

# 数字核对（必须 16/16）
ssh a40 'cd ~/deepseek-rul && export LD_LIBRARY_PATH=... && python scripts/verify_paper_numbers.py'
```

---

## 4. 我犯过并已修正的错误（新会话不要重犯）

| # | 错误 | 教训 |
|---|---|---|
| 1 | **覆盖对齐比较里 Winkler 的 α 跟着 δ 漂移**，导致不同罚则的分数被插值到同一条前沿 | 修好后 59/80 变 75/80，"FD003/FD004 打平"的结论消失。**区间比较必须固定评分规则** |
| 2 | 相关性在窗口级算，忽略同引擎内窗口不独立 | 引擎级 ρ 从 +0.335 掉到 +0.086 |
| 3 | 论文摘要存在双份真相（Markdown 改了、LaTeX 前导码是旧的） | 已改为 `abstract.tex` 单一来源，构建时注入 |
| 4 | 7 张表全部没有表题，正文从不引用图表编号 | 已补 7 条 caption + 逐表引用 |
| 5 | 交付的寿命先验表 σ_T 恒为 0 | 已用 bootstrap 重估 |
| 6 | 局部化 CQR 在 14 格上看着好，是 FD001 占 7/14 的采样偏差 | 80 格上优势消失 |
| 7 | FEMTO 用 cap=125 截断标签，95% 标签=125，模型学成常数 | 改用相对寿命 |
| 8 | PowerShell 写构建脚本反复因引号/正则失败 | 已改用 Node（`build_manuscript.mjs`） |
| 9 | `git add -A` 误把 `tmp/`（679 文件）提交 | 已撤回并加入 `.gitignore` |
| 10 | 讨论/结论被索引切片截断丢失两次 | 已加 7 章结构断言 |

**通用教训**：我自己写的验证脚本会继承我自己写分析脚本时的错误。**独立审计不可省。**

---

## 5. 文件地图

```
docs/
  research_state.md            ← 恢复入口（先读这个）
  decision_log.md              ← D001–D009，已决事项不要重开
  autopilot_contract.md        ← 执行队列与出口断言
  reproduce.md                 ← 复现指南
  citation_audit_20261108.md   ← 文献审计报告（3 处错误已修）
  wiki/{hypotheses,experiments,failures,progress,literature}.md
  manuscripts/
    v1.0-draft.md              ← 正文唯一来源（Markdown）
    abstract.tex               ← 摘要唯一来源
    references.bib             ← 18 条，全部核实存在
    manuscript.tex / .pdf      ← 构建产物（不要手改 tex）
    SUBMISSION_CHECKLIST.md    ← 缺口清单
scripts/
  build_manuscript.mjs         ← 构建（结构断言 + 摘要同步 + 表题提升）
  verify_paper_numbers.py      ← 16 项数字断言
  table_all_methods.py         ← 覆盖对齐对比（α 已修正为 0.1）
  audit_protocol.py            ← 协议审计（窗口 vs 引擎加权、相关性）
  h004_width_allocation.py     ← H004 实验（未过判定门）
src/punet/                     ← 模型与 CQR 实现
```

---

## 6. 不要做的事

1. **不要重跑 80 格主实验**（已完成，产物在 a40）
2. **不要重试 §1.3 的 8 条已关闭路线**
3. **不要用单点 Winkler 比较不同覆盖的方法**（必须覆盖对齐 + 固定 α）
4. **不要把 H004 写成机制贡献**（未过判定门）
5. **不要在窗口级报告相关性**（引擎才是独立单位）
6. **不要声称普遍优于 MC-Dropout**（只有 10 格，且 FD002 上更差）
7. **不要 force-push 或改写历史**（结论被推翻只追加"已撤回"）

---

## 7. 分支与提交

- 工作分支：`punet-interval`
- 最新提交：`7548e31`（citation-audit 修复）
- 远端已同步

---

## 8. 给新会话的第一条指令建议

> 读 `docs/HANDOFF.md` → `docs/research_state.md` → `docs/decision_log.md` D009，
> 然后执行修订清单第 ③ 步（MC-Dropout 补到 80 格，a40 后台跑），
> 期间并行做第 ① 步（核实 CRULP 与 IJPHM 2026，写差异表）。
> 卡住的地方走 L2（`mcp__codex__codex`）。
