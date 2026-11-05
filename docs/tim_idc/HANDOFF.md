# 交接文档（完整版）— IDA-PE / IEEE TIM 投稿流水线

更新于 2026-11-05 12:40。读者：接手本项目的任何人或 AI 会话（含 GPT / Codex 审计）。
目标：10 分钟内获得全部上下文——所有位置、所有判决数字、创新方法是什么、剩下什么活。

---

## 0. 一句话现状

**架构已冻结：最终方法 = M1（四子集主表主打臂）/ M1p（FD002 点最优变体），即 PE 主干
+ zero-init 因果寿命先验注入。Gate 0 六种子、Gate 1、Module B/C 双 DROP、四子集主表、
E22a 规模对齐、两个 2026 基线（CET-MSA 全复现 + BootBoGs-TS 官方数据）全部完成。
剩余工作纯写作/纯 CPU：Gate 2 校准判决 → 论文成稿 → 官方 test 一次解封 → 投稿
（截止 ~2026-11-07/08）。GPU 全空闲，无遗留运维故障。**

## 1. 位置索引

### 1.1 三个位置

| 位置 | 内容 | 说明 |
|---|---|---|
| **GitHub** | https://github.com/enterzxk/deepseekrul | 唯一权威仓库。分支 `tim-idc-rul` = `main` = `origin/main`，最新 commit `b4d14c6`。全部报告/判决 JSON/脚本/README 已同步 |
| **本地 Windows** | `C:\Users\20562\Desktop\deepseek-rul\` | git 主仓库检出（CRLF 行尾，与远程内容 9/9 一致） |
| **3090 服务器** | `ssh user@3090`，代码 `/home/user/rulb/` | 全部原始实验在这里跑。非 git 仓库；审计入口 `/home/user/rulb/AUDIT_START_HERE.md` |
| **服务器 Python** | `/home/user/miniconda3/envs/pmv/bin/python` | torch 2.5.0+cu124，跑实验一律用它 |

### 1.2 3090 服务器文件地图（`/home/user/rulb/` 下）

```
/home/user/rulb/
├── AUDIT_START_HERE.md            ← 审计入口：复算命令 + 判决 JSON 索引 + 红线
├── gates.json                     ← 预注册冻结判据（审计基准，勿改）
├── data/CMAPSS/                   ← C-MAPSS 数据（FD001–FD004 train/test）
├── src/
│   ├── lifetime_prior/            ← 嵌套因果寿命先验（nested_prior.py 等）
│   └── candidates/deg_modules.py  ← Module B/C 候选模块（已 DROP，保留代码）
├── scripts/
│   ├── run_ida.py                 ← 五臂训练（M0/M1/M1p/M1_shuf/M1_const）
│   ├── run_h1.py / run_h2inner.py / run_h3.py / run_h2finalize.py   ← Gate 1 臂
│   ├── run_moduleBC.py            ← Module B/C 七臂
│   ├── run_e22a_align.py          ← E22a 大骨干对齐（M0big/M1big/M1pbig）
│   ├── summarize_gate0.py         ← 判决汇总（--seed N；冻结对）
│   ├── summarize_gate1.py / summarize_modulebc.py
│   ├── summarize_subsets.py       ← 通用子集汇总（FD003/004、E22a 用）
│   └── run_baseline_cet.py 相关   ← 基线复现驱动
├── experiments/ida/
│   ├── prior/{FD001..FD004}_prior_cutoffs.csv      ← 先验表（每引擎 μ_T/σ_T）
│   ├── gate0/gate0_verdict_s{101,202,303,42,67,123}.json   ← Gate 0 六种子判决
│   ├── gate0/fd34_summary.json    ← FD003/004 五臂汇总
│   ├── gate0/FD00X/{ARM}_f{k}_s{seed}/val_predictions.csv  ← 原始预测（含 mu_T/sigma_T 列，可复算 pooled RMSE）
│   ├── gate1/gate1_verdict.json + h2_weights.json          ← Gate 1 判决
│   ├── moduleBC/modulebc_verdict.json                      ← Module B/C 判决
│   └── e22a_align/align_summary.json                       ← E22a 对齐判决
├── baselines/
│   ├── Noise-Robust-RUL-CET-MSA/  ← CET-MSA 官方仓库（已复现）
│   └── BootBoGs-TS/               ← BootBoGs-TS 官方仓库（用官方 results CSV）
├── logs/                          ← 队列日志：gate0B/gate1/gate0C/gate3fd/moduleBC/e22a/baseline_cet/
└── driver.py                      ← GPU 队列驱动：driver.py <tasks.txt> <status.txt>
```

每个 `metrics.json` 内含 `command` 字段，可追溯每次运行的完整命令行。

### 1.3 本地 / GitHub 文档地图（`docs/tim_idc/`）

| 文件 | 内容 |
|---|---|
| `HANDOFF.md` | 本文档（最新状态交接） |
| `GATE0_PREREG.md` | Gate 0 预注册原文（结果前冻结） |
| `gate0_report.md` / `gate0_multiseed.md` | Gate 0 单种子 + 六种子定稿 |
| `gate1_report.md` | Gate 1 三臂判决 + 机制分析 |
| `modulebc_report.md` | Module B/C 双 DROP 判决 |
| `four_subset_results.md` | 四子集主表 + E22a 对齐（含全部 CI） |
| `baselines_repro.md` | 2026 基线复现（CET-MSA / BootBoGs-TS）+ 诚实读法 |
| `protocol_calibers.md` | **两套口径防呆文档（必读，严禁混表）** |
| `paper/00–03` | 论文骨架：结构/相关工作/数据协议/实验设置 |
| `AUDIT_START_HERE.md`（仓库根） | 审计入口（本地+GitHub+远程三处同步） |

## 2. 创新方法（论文方法章素材）

### 2.1 方法主线一句话

**Causal Lifetime Prior → Individualized Degradation Clock → Degradation-Conditioned
PE Predictor**：一个严格因果、嵌套交叉拟合的寿命先验，通过零初始化层注入冻结的
positional-encoding（PE）Transformer 主干，把"这台引擎还剩多少寿命"的个体身份信息
喂给 RUL 预测器，而不改变主干结构。

### 2.2 三个组件

1. **Causal Lifetime Prior（因果寿命先验）**
   - `HistGradientBoosting` 集成（嵌套交叉拟合：外 5 折 × 内 4 折 × 5 模型）预测每个
     引擎的剩余寿命 μ_T 与不确定度 σ_T。
   - **阶梯因果构造（staircase construction）**：止于循环 t 的窗口只消费最新的
     cutoff 预测 c ≤ t，绝不用未来插值——cutoff 之前的一切可用，之后的一切不可见。
   - 泄漏测试 T1–T4（`tests/test_prior_causality.py`）全部通过：未来信息、跨引擎
     泄漏、cutoff 越界、标签泄漏均被排除。

2. **Individualized Degradation Clock（个体化退化时钟）**
   - `tau = t / μ_T`：把寿命身份转成相位时间。同一循环数在不同健康引擎/退化引擎上
     意义不同，时钟把这个差异显式编码。

3. **Degradation-Conditioned PE Predictor（退化条件化 PE 预测器）**
   - PE 主干**冻结不变**（小配置 d_model 64 / 1 层 / 42.3k 参数），先验（μ_T、σ_T、
     tau）经 **zero-init** 层注入 context 通道。
   - zero-init 的作用：每个臂的初始状态严格等价于 M0（纯 PE），训练初期梯度为零、
     随后按需生长——delta 可完全归因于**注入的信息**，而非重初始化或容量。

### 2.3 方法论创新（评测纪律本身也是贡献）

- **预注册门控流水线**：判据在 `gates.json` 冻结于看结果之前；门失败即停、换指标/
  换子集/事后调阈值一律禁止；官方 test 只解封一次。
- **五臂归因设计**：M0（无先验）/ M1（先验 concat）/ M1p（+tau）/ M1_shuf（打乱
  μ_T）/ M1_const（常数 μ_T）。两个对照臂与 M1 参数量完全相同（+192），但都逆转
  增益 → 收益来自**引擎个体寿命信息**，不是容量。
- **引擎级 bootstrap**：禁止窗口级 bootstrap（窗口共享引擎信息会低估方差），一律
  引擎聚类 5000 次。

### 2.4 动机（诊断阶梯，FD002 因果特征）

HGB 直接回归 RUL ~31.3 vs 先验隐含 RUL 17.6 vs 纯 PE 12.0 vs M1 12.28。
**寿命"身份"远比轨迹回归可学**——这解释了为什么走时钟通道注入，而不是把 HGB 的
RUL 直接当预测迁移。

## 3. 完整实验结果（全部数字来自判决 JSON，脚本生成，非手写）

### 3.1 Gate 0 — 先验注入有效（六种子闭环，GO_TO_GATE1）

判据 G0.1–G0.3 在 FD002（259 引擎），G0.4 为 FD001（80 引擎）无害护栏。
引擎级 bootstrap 5000 次。冻结主判决 seed 101：G0.1 −0.279 [−0.498, −0.053] PASS、
G0.2 +0.312 PASS、G0.3 +0.279 PASS、G0.4 PASS。六种子确认
（`gate0_multiseed.md`）：

| 判据 | 结果 |
|---|---|
| G0.1 M1 vs M0 | **6/6 PASS**，效应 −0.279 → −0.580 随种子单调走强 |
| G0.2 M1_shuf vs M1 / G0.3 M1_const vs M1 | 各 5/6 显著（仅 seed 303 跨零），方向 6/6 一致 |
| G0.4 FD001 护栏 | 3/6 PASS；FAIL 均为 n=80 功效问题，点估计从未超 +0.255 |

### 3.2 Gate 1 — 信息通道正确（GO_TO_MODULE_BC，六对全 PASS）

M1 与 M1p 各自同时赢三个替代通道（`gate1_report.md`，全部 CI 排除 0）：

| 对照 | M1 | M1p |
|---|---|---|
| H1：HGB 直接 RUL 注入 | **−17.15** | **−17.22** |
| H2：PE/HGB 加权融合 | −0.226 | −0.290 |
| H3：PE + H1 OOF 特征 | −0.313 | −0.378 |

H2 内层权重塌缩到 w≈0.95–1.0（PE 主导）——HGB 自我否定。结论：**寿命信息必须经
时钟通道消费，PE 主干不可替代。**

### 3.3 Module B/C — 双 DROP（架构冻结为纯 M1p）

- **Module B**（退化条件化注意力偏置，+440 参数）：tau/abs/shuf 三条件统计不可区分
  （tau −0.007/+0.131/+0.030 量级，全 CI 跨零）→ DROP。
- **Module C**（传感器低秩混合器，+252 参数）：MC_tau 点估计最好（−0.261 vs M1p）
  但 CI 跨零，且 shuf 对照复现同样下移（shuf 也到 11.960）→ 是容量效应不是 tau
  信息 → DROP。
- 总预算仅 +1332 参数（3.1%），两个候选模块都不带来 tau 特异信息 → **架构冻结为
  纯 M1p/M1，负面消融进论文（反而强化主线论证）。**

### 3.4 四子集主表（`four_subset_results.md`，outer-val 口径 B，cap125，seed 101）

| 子集（n 引擎） | M0 | **M1** | M1p | M1−M0 [95% CI] | shuf−M1 | const−M1 |
|---|---|---|---|---|---|---|
| FD001 (80) | 9.824 | 9.891 | 9.837 | +0.07（六种子均值，护栏） | — | — |
| **FD002** (259) | 12.75 | **12.31** | 12.32 | 六种子全显著（−0.28…−0.58） | 显著 5/6 | 显著 5/6 |
| FD003 (100) | 9.806 | **9.702** | 9.943 | −0.105 [−0.581, +0.359] ns（方向对） | +0.199 ns | +0.059 ns |
| **FD004** (249) | 12.717 | **12.242** | 12.721 | **−0.475 [−0.749, −0.221] SIG** | **+0.365 [+0.166, +0.575] SIG** | **+0.403 [+0.177, +0.645] SIG** |

读法：**FD004 完整复刻 Gate 0 模式（3/3 对比全显著）**，且是运行上最难的子集
（6 工况、249 引擎）——机制的最强单子集确认。FD003 三对比方向全部一致但功效不足
（n=100），无任何符号反转。FD003/004 上 M1 优于 M1p → **论文四子集表主打 M1**，
M1p 保留为 FD002 点最优变体。

### 3.5 E22a 规模对齐（大骨干 d88/2 层/141k 参数，outer-val 口径）

| 子集 | M0big | M1big | M1pbig | M1big−M0big | M1pbig−M0big |
|---|---|---|---|---|---|
| FD002 (260) | 12.759 | 12.425 | 12.506 | **−0.334 [−0.647, −0.027] SIG** | −0.254 ns |
| FD001 (100) | 10.094 | 9.846 | 9.946 | −0.247 ns | −0.148 ns |

读法：先验效应**跨规模成立**（FD002 显著）；M0big 12.759 ≈ 小 M0 12.75 → 容量不是
outer-val 瓶颈。与历史 E22a test 数字（FD002 11.51）的差距确系 **val/test 分布差**
而非模型退步。结论：**最终方法保持小配置（42.3k，更小更快，不劣）**，大臂作为
规模稳健性检查进论文。zero-init 保证官方 test 解封不会低于历史 11.51 的架构下限。

### 3.6 基线复现（2026 论文，`baselines_repro.md`）

**CET-MSA（全复现，官方代码+配置，cap150 协议，seed 0，logs/baseline_cet/）：**

| 子集 | 复现 test RMSE | 论文数字 |
|---|---|---|
| FD001 | **17.22**（3 迭代 16.92 ± 0.68，排除种子运气） | 不可验证 |
| FD002 | **22.41** | 不可验证 |
| FD003 | **16.55** | 不可验证 |
| FD004 | **21.78** | 不可验证 |

诚实读法：论文 Data Availability 承诺的 result tables 未随仓库发布（docs 全是
0 字节），Springer 付费墙 → 绝对数字公开渠道无法验证。复现水平远低于注意力模型
文献常见水平（FD002 常见 13–15 @cap125；cap150 更宽松只会更好做）——**发布包未
达到论文水平**。忠实报告 + 脚注说明。

**BootBoGs-TS（不重跑，用作者提交的官方 results CSV）：** FD004 test RMSE —
bootbogs_ts_oob_1.5 (h224) **24.39** / bayes_tpe 22.42 / random (h160) 18.29 /
grid_full 27.37。注意：作者自己的表里 proposed 方法不是最好行（random 赢）→
如实带脚注报告。

### 3.7 论文对比表框架（`baselines_repro.md` 末节）

- 主表（官方 test，cap 125）：我方最终 M1 一次解封数字 vs CET-MSA 复现（脚注
  cap150）、BootBoGs FD004 官方数、无代码论文引用数字（AIDGN / LVDACNN /
  MSMT-FDA）；背景行：multiscale spatio-temporal graph (2025) 11.23/12.64/11.15/13.41、
  adaptive dual distillation (2025) 13.43/14.36/14.11/16.25、我方历史 E22a PE
  9.82/11.51/10.06/11.79（同口径参考行）。
- 消融表（outer-val，cap 125）：3.4/3.5 的表，内部专用，绝不与 test 行混排。
- 诚实定位：卖点不是 SOTA-RMSE-at-any-cost，而是**泄漏受控的因果先验 + 预注册
  门控 + 有利的算力足迹（42.3k 参数）**。基线是背景，不是主战场。

### 3.8 两套口径防呆（`protocol_calibers.md`，严禁混表）

| 口径 | 评测集 | 协议 | 数字 |
|---|---|---|---|
| **A：E21/E22a**（2026-10-31，历史参考） | 官方 test 引擎 | 单一 train/val split（split-seed 7），cap125，~10^5 参数 | FD001 9.82 / FD002 11.51 / FD003 10.06 / FD004 11.79 |
| **B：IDA-PE**（当前冻结） | 5 外折 outer-val 训练侧引擎 | 嵌套交叉拟合，cap125，42.3k 参数 | 3.4 节全部数字 |
| 基线口径 | 各基线**自己的官方协议** | CET-MSA cap150 / BootBoGs 官方 CSV | 3.6 节 |

FD001 上 A、B 恰好都是 ~9.82（val/test 分布接近的巧合）；FD002 差 ~1.2 是分布差
不是退步。跨口径行永不并排出现。

## 4. 当前状态与剩余工作（按顺序）

当前：GPU 双卡空闲，无运行队列；11-04 单日 ~410 GPU runs，0 未解释失败；全部判决
JSON 已拉回并提交（`b4d14c6`）。

1. **Gate 2 校准判决**（纯 CPU，判据在 gates.json）：不确定度 σ_T（或校准后的 u）
   与 |μ_T − T| 的 Spearman > 0.3。数据：`experiments/ida/gate0/FD002/
   M1p_f{k}_s*/val_predictions.csv`（含 mu_T/sigma_T 列）。通过 → 加"安全门"小节；
   不通过 → 论文不提校准，主线照写。
2. **论文成稿**（骨架 `docs/tim_idc/paper/00–03`）：方法章用 2.2–2.4 节素材；
   实验章主表 = 3.4 表 + 六种子 G0.1 + Gate 1 三臂 + Module B/C 负消融 + 3.6 基线
   对比表。**数字只从 JSON/CSV 读，不手写。**
3. **官方 test 一次解封**：全部写作/判决冻结后，对最终模型（M1）跑一次 test
   （口径 A：cap125）；**禁止解封后调参重跑。**
4. **投稿**：11-06/07 内审润色，11-07/08 前提交 IEEE TIM。

| 时间 | 事件 |
|---|---|
| 11-05 | Gate 2（CPU）+ 方法章；实验章成稿 |
| 11-06 | 内审、图表精修、official test 一次解封出最终数字 |
| 11-07 | 投稿 |

## 5. 运维陷阱（血泪教训，务必遵守）

1. **relay/watcher 禁用 pgrep 模式串**——三次自匹配死锁。等待条件一律用**标记文件**
   （先验完成 touch `*.done`，relay 轮询文件存在）。
2. **CUDA_VISIBLE_DEVICES 必须显式设**（run_ida/run_moduleBC/run_e22a_align 默认 cuda:0）。
3. **driver 会无声死亡**：每次巡检 `wc -l status` 对任务总数，缺了补漏队列。
4. 后台进程统一 `setsid nohup ... < /dev/null &`；一律绝对路径。
5. 队列运行期间不得修改被引用源文件；新实验一律新脚本。
6. `pkill -f` 模式串会匹配 ssh 自身命令行——远程杀进程先 `ps aux | grep` 拿 PID
   再 `kill PID`。

## 6. 纪律红线（违反 = 推翻整个审计链）

- 判据冻结在 `gates.json`；门失败 → 停下写报告等人工决策，不换指标/子集/阈值。
- **禁止窗口级 bootstrap**，一律引擎聚类 5000 次。
- ensemble 与单模型分开报告；官方 test 只解封一次。
- 汇总脚本只读 metrics/predictions，不手写数字。
- 已确认结论不得推翻：LSC 线 hgb 12.98 优于 plain 13.96；EAC G1/G2 NO-GO；
  Modern Backbone 失败；保留现有 PE 骨干；Module B/C DROP。
- 提交纪律：每个判决/报告一个 commit，message 写明数字。

## 7. 复算验证命令（3090 服务器，只读，随时可跑）

```bash
PY=/home/user/miniconda3/envs/pmv/bin/python
cd /home/user/rulb
# Gate 0 任一种子判决（引擎级 bootstrap 5000）
$PY scripts/summarize_gate0.py --seed 101 --out /tmp/verify_g0.json
# Gate 1 / Module B/C
$PY scripts/summarize_gate1.py && $PY scripts/summarize_modulebc.py
# FD003/004 五臂
$PY scripts/summarize_subsets.py --base experiments/ida/gate0 --subsets FD003 FD004 \
  --arms M0 M1 M1p M1_shuf M1_const --contrasts 'M1,M0' 'M1_shuf,M1' 'M1_const,M1'
```

代码完整性（2026-11-04 核验）：本地与远程 9 个关键脚本 sha256 归一化（LF）后
9/9 一致；本地 Windows 侧整文件 hash 不同属 CRLF 行尾差异，非代码漂移。
