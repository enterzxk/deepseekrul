# 交接文档 — IDA-PE / IEEE TIM 投稿流水线

写于 2026-11-04 23:16。读者：接手本项目的下一个 AI 会话（含 GPT）。目标：5 分钟内获得全部上下文，不做任何已完成的工作，不违反任何冻结纪律。

---

## 0. 一句话现状

**架构已冻结：最终方法 = M1p（PE 主干 + zero-init 先验 context 注入）。Gate 0 六种子、Gate 1、Module B/C 判决、四子集主表、E22a 对齐全部完成，全部 GPU 实验收官。剩余工作纯写作/纯 CPU：Gate 2 校准判决 → 论文成稿 → 基线数字 → 官方 test 一次解封。**

## 1. 项目与目标

- 仓库：本地 `C:\Users\20562\Desktop\deepseek-rul`，分支 `tim-idc-rul`（= main = origin/main），GitHub: enterzxk/deepseekrul
- 算力：3090 服务器 `ssh user@3090`，代码 `/home/user/rulb/`，Python 用 `/home/user/miniconda3/envs/pmv/bin/python`
- 截止：5 天内投 IEEE TIM（2026-11-03 起算，即 ~11-07/08）
- 方法主线：**Causal Lifetime Prior（嵌套交叉拟合预测每引擎寿命 μ_T）→ 个体化退化时钟 tau = t/μ_T → 退化条件化 PE 预测器**
- 全部判据预注册冻结在 `gates.json`，**结果出来后不得修改判据、不得换指标、门失败即停**

## 2. 已完成的判决（不要重跑，全部有 JSON/报告背书）

### ① Gate 0 — 先验注入有效（冻结主判决 seed 101：GO_TO_GATE1）

六种子确认闭环（`docs/tim_idc/gate0_multiseed.md`，binding 呈现规则在内）：

| 判据 | 结果 |
|---|---|
| G0.1 M1 vs M0 (FD002) | **6/6 PASS**，效应 -0.279→-0.580 随种子单调走强 |
| G0.2 shuf vs M1 / G0.3 const vs M1 | 各 5/6 显著（仅 seed 303 跨零），方向 6/6 一致 |
| G0.4 FD001 护栏 | 3/6 PASS；FAIL 均为功效问题（点估计从未超 +0.255） |

### ② Gate 1 — 信息通道正确（GO_TO_MODULE_BC，六对全 PASS）

`docs/tim_idc/gate1_report.md`：M1/M1p 各自同时赢 H1（HGB 直接 RUL，-17.2 RMSE！）、H2（加权融合，内层权重塌缩 w≈1.0）、H3（PE+H1 特征）。**寿命信息必须经时钟通道消费，PE 主干不可替代。**

### ③ Module B/C — 双 DROP（`docs/tim_idc/modulebc_report.md`）

- Module B（退化条件化注意力）：tau/abs/shuf 三条件统计不可区分，全 CI 跨零 → 删
- Module C（传感器混合器）：MC_tau 点估计最好（-0.261 vs M1p）但 CI 跨零，且 shuf 对照复现同样下移（容量效应非 tau 信息）→ 删
- **架构冻结为纯 M1p。** 负面消融进论文（一段话+一张表），反而强化主线论证

### ④ 四子集主表（`docs/tim_idc/four_subset_results.md`，outer-val 口径）

| 子集 | M0 | **M1** | M1p | M1−M0 [CI] | shuf/const−M1 |
|---|---|---|---|---|---|
| FD002 (259) | 12.75 | **12.31** | 12.32 | 六种子全显著 | 显著 5/6 |
| **FD004 (249)** | 12.717 | **12.242** | 12.721 | **−0.475 SIG** | **+0.365/+0.403 SIG（3/3）** |
| FD003 (100) | 9.806 | **9.702** | 9.943 | −0.105 ns（方向对） | 方向对，ns |
| FD001 (80) | 9.824 | 9.891 | 9.837 | 护栏（+0.07，功效 caveat） | — |

要点：FD004 最强单子集确认；FD003/004 上 **M1 优于 M1p** → 四子集表主打 M1；M1p 仅作 FD002 点最优变体。

### ⑤ E22a 对齐（`align_summary.json` + `docs/tim_idc/protocol_calibers.md`）

- E22a 大配置（141k 参数）下 M1big−M0big = **−0.334 SIG**（FD002）→ 先验效应跨规模成立
- M0big 12.759 ≈ 小 M0 12.75 → 容量不是瓶颈；与历史 E22a test 数字（FD002 11.51）的差距确系 **val/test 分布差**，不是退步
- **两套数字口径严禁混表**：E21/E22a = 官方 test（9.82/11.51/10.06/11.79）；IDA = outer-val。详见 `protocol_calibers.md`

## 3. 当前状态

- **GPU 全空闲**，无运行中队列，无遗留运维故障
- 今日（11-04）总计 ~410 GPU runs，0 未解释失败
- 所有判决/汇总 JSON 已拉回本地并提交（最新 `f8c59f1`）

## 4. 你的下一步（按顺序）

1. **Gate 2 校准判决**（纯 CPU，用现有 M1p 预测即可）：
   - 判据（gates.json）：不确定度 σ_T（或校准后的 u）与 |μ_T − T|（真寿命误差）的 Spearman > 0.3
   - 数据：`experiments/ida/gate0/FD002/M1p_f{k}_s*/val_predictions.csv`（含 mu_T/sigma_T 列）
   - 通过 → 加"安全门"小节；不通过 → 论文不提校准，主线照写
2. **论文成稿**（骨架已有 `docs/tim_idc/paper/00-03`）：
   - 方法章：先验→时钟→context（素材：gate0_report + gate1_report 的机制节）
   - 实验章主表：four_subset_results.md 表 1 + 六种子 G0.1 + Gate 1 三臂对照 + Module B/C 负消融表
   - 数字纪律：只从 JSON/CSV 读，不手写
3. **基线复现**（可并行挂 GPU）：CET-MSA（suosimiaoqianli/Noise-Robust-RUL-CET-MSA）、BootBoGs-TS（Gyouness/BootBoGs-TS）已 clone 到远程 `baselines/`，环境未建；AIDGN/LVDACNN/MSMT-FDA 无代码（引用数字）
4. **官方 test 一次解封**：全部写作/判决冻结后，对最终模型（M1）跑一次 test 集（口径 A：cap125）；**禁止解封后调参重跑**
5. **投稿**：11-06/07 内审润色

## 5. 关键路径速查

| 内容 | 路径 |
|---|---|
| 冻结判据 | `gates.json`（勿改） |
| 先验表 | 远程 `experiments/ida/prior/{FD001..FD004}_prior_cutoffs.csv` |
| 五臂预测（四子集） | 远程 `experiments/ida/gate0/FD00X/{ARM}_f{k}_s{seed}/val_predictions.csv` |
| Gate 1 三臂 | 远程 `experiments/ida/gate1/` + `h2_weights.json` |
| Module B/C | `experiments/ida/moduleBC/modulebc_verdict.json`、`src/candidates/deg_modules.py` |
| E22a 对齐 | `scripts/run_e22a_align.py`、`experiments/ida/e22a_align/align_summary.json` |
| 汇总脚本 | `summarize_gate0.py --seed N`（冻结对）、`summarize_gate1.py`、`summarize_subsets.py`（通用） |
| 训练脚本 | `run_ida.py`（五臂）、`run_h1/h2inner/h3/h2finalize.py`、`run_moduleBC.py` |
| 报告 | `docs/tim_idc/`：gate0_report / gate0_multiseed / gate1_report / modulebc_report / four_subset_results / protocol_calibers / GATE0_PREREG |
| 工作日志 | `.workbuddy/memory/2026-11-03.md`、`2026-11-04.md` |
| 队列驱动 | 远程 `driver.py <tasks.txt> <status.txt>`；任务格式 `logname\tcommand` |

## 6. 运维陷阱（血泪教训，务必遵守）

1. **relay/watcher 禁用 pgrep 模式串**——三次自匹配死锁。**用标记文件**：先验完成 touch `*.done`，relay 轮询文件存在
2. **CUDA_VISIBLE_DEVICES 必须显式设**（run_ida/run_moduleBC/run_e22a_align 默认 cuda:0）
3. **driver 会无声死亡**：每次巡检 `wc -l status` 对任务总数，缺了补漏队列
4. 后台进程统一 `setsid nohup ... < /dev/null &`；一律绝对路径
5. 队列运行期间不得修改被引用源文件；新实验一律新脚本（run_ida.py 已被四子集引用过，改它前确认无队列）
6. `pkill -f` 模式串会匹配 ssh 自身命令行——远程杀进程先 `ps aux | grep` 拿 PID 再 `kill PID`

## 7. 纪律红线（违反 = 推翻整个审计链）

- 判据冻结在 gates.json；门失败 → 停下写报告等人工，不换指标/子集/阈值
- **禁止窗口级 bootstrap**，一律 engine 聚类（5000 次）
- ensemble 与单模型分开报告；官方 test 只解封一次
- 汇总脚本只读 metrics/predictions，不手写数字
- 已确认结论不得推翻：LSC 线 hgb 12.98 优于 plain 13.96；EAC G1/G2 NO-GO；Modern Backbone 失败；保留现有 PE 骨干；Module B/C DROP
- 提交纪律：每个判决/报告一个 commit，message 写明数字

## 8. 时间线预估（从 23:16 起）

| 时间 | 事件 |
|---|---|
| 11-05 上午 | Gate 2 校准判决（CPU）+ 论文方法章 |
| 11-05 下午 | 实验章成稿（主表/消融/机制节）；基线复现挂 GPU |
| 11-06 | 内审、图表精修、official test 解封出最终数字 |
| 11-07 | 投稿 |
