# 交接文档 — IDA-PE / IEEE TIM 投稿流水线

写于 2026-11-04 18:32。读者：接手本项目的下一个 AI 会话。目标：5 分钟内获得全部上下文，不做任何已完成的工作，不违反任何冻结纪律。

---

## 0. 一句话现状

**Gate 0（六种子）与 Gate 1 已通过，当前在跑 Module C 验证队列（预计 20:00 完成），跑完后做 modulebc 判决 → 决定保留或删除两个附加模块 → Gate 2（校准）→ 官方 test 一次解封 → 论文。**

## 1. 项目与目标

- 仓库：本地 `C:\Users\20562\Desktop\deepseek-rul`，分支 `tim-idc-rul`（= main = origin/main = `549e39a`），GitHub: enterzxk/deepseekrul
- 算力：3090 服务器 `ssh user@3090`，代码 `/home/user/rulb/`，Python 用 `/home/user/miniconda3/envs/pmv/bin/python`
- 截止：5 天内投 IEEE TIM（2026-11-03 起算，即 ~11-07/08）
- 方法主线：**Causal Lifetime Prior（嵌套交叉拟合预测每引擎寿命 μ_T）→ 个体化退化时钟 tau = t/μ_T → 退化条件化 PE 预测器（M1p）**
- 全部判据预注册冻结在 `gates.json`（本地+远程同文件），**结果出来后不得修改判据、不得换指标、门失败即停**

## 2. 已完成的判决（不要重跑）

### Gate 0 — 先验注入有效（冻结主判决 seed 101：GO_TO_GATE1，4/4 PASS）

六种子确认闭环（`docs/tim_idc/gate0_multiseed.md`，定稿）：

| 判据 | 结果 |
|---|---|
| G0.1 M1 vs M0 (FD002) | **6/6 PASS**，效应 -0.279→-0.347→-0.448→-0.462→-0.540→-0.580 随种子单调走强 |
| G0.2 M1_shuf vs M1 | 5/6 显著（仅 seed 303 跨零），方向 6/6 一致 |
| G0.3 M1_const vs M1 | 5/6 显著（仅 303 跨零），方向 6/6 一致 |
| G0.4 FD001 护栏 | 3/6 PASS（seed 123 点估计 -0.126 为负）；303/42/67 为功效性 FAIL（FD001 n=80，点估计从未超 +0.255） |

归因闭环：shuffled/const 对照（同参数量 +192）使收益反向 → 增益来自真实个体寿命信息。论文呈现规则在 gate0_multiseed.md（binding）：报全六种子、G0.1 可称 6/6、G0.2/3 称"方向一致 5/6 显著"、G0.4 带功效 caveat。

### Gate 1 — 信息通道正确（GO_TO_MODULE_BC，六对比较全 PASS）

`docs/tim_idc/gate1_report.md` + `experiments/ida/gate1/gate1_verdict.json`：

- M1/M1p 各自同时赢三个替代臂：vs H1（HGB 直接 RUL）**-17.15/-17.22 RMSE**；vs H2（PE/HGB 加权）-0.226/-0.290（CI 上界仅 -0.003，报告须给精确区间）；vs H3（PE+H1 特征）-0.313/-0.378
- 机制洞察：H1 崩溃（29.45）→ 寿命信息必须经时钟通道消费，PE 主干不可替代；H2 内层权重塌缩 w≈0.95–1.0（HGB 自己否定自己）；H3 噪声化
- M1p 12.217 < M1 12.280（seed 101 FD002 点估计偏好）

## 3. 正在运行（18:32 时刻）

- **Module C 队列**：20 runs（MC_tau/abs/shuf/static × 5 折），GPU0，18:27:11 启动，单 run ~276s，**预计 ~20:00 完成**
- 查完成：`ssh user@3090 "wc -l /home/user/rulb/logs/moduleBC/status0.txt"` → 20 行含 `DONE` 即完成
- Module B 已 15/15 DONE（MB_tau/abs/shuf × 5 折，GPU1）
- 数据输出：`/home/user/rulb/experiments/ida/moduleBC/FD002/{ARM}_f{k}/`（train log 在 `/home/user/rulb/logs/moduleBC/`）

## 4. 你的下一步（按顺序）

1. **写 `scripts/summarize_modulebc.py`**（新文件，参考 summarize_gate1.py 的引擎级 bootstrap 5000 次框架）并跑判决：
   - 核心比较（FD002）：MB_tau vs M1p、MC_tau vs M1p、MB_tau vs MB_abs、MB_tau vs MB_shuf、MC_tau vs MB_tau 同款三对
   - 冻结判据（gates.json `modules_BC`）：tau 臂 vs M1p CI 排除 0，**且** tau 臂优于 abs/shuf 对照 → 保留模块；否则按规格**直接删除该模块**（论文主线不受影响）
   - M1p 参照数据在 `experiments/ida/gate0/FD002/M1p_f{k}_s101/`
2. **判决后分叉**：保留 → Gate 2（不确定度校准，判据在 gates.json：u 与 |μ_T−T| 的 Spearman > 0.3 才做安全门）；删除 → M1p 即最终方法，直接进 Gate 2/论文
3. **官方 test 一次解封**：只对最终模型跑一次，协议见门控补充版（禁止调参后重跑）
4. **论文**：骨架已有 `docs/tim_idc/paper/00_structure.md / 01_related_work.md / 02_dataset_protocol.md / 03_experimental_setup.md`；结果章等 modulebc + Gate 2 数字；Gate 0/1 报告可直接改写成实验节
5. **基线复现**（可并行）：CET-MSA（github suosimiaoqianli/Noise-Robust-RUL-CET-MSA）与 BootBoGs-TS（github Gyouness/BootBoGs-TS）已 clone 到远程 `baselines/`，环境未建；AIDGN/LVDACNN/MSMT-FDA 无官方代码（引用数字即可）

## 5. 关键路径速查

| 内容 | 路径 |
|---|---|
| 冻结判据 | `gates.json`（勿改） |
| 先验（μ_T/σ_T 表） | 远程 `experiments/ida/prior/{FD001,FD002}_prior_cutoffs.csv` |
| Gate 0 五臂预测 | 远程 `experiments/ida/gate0/FD00X/{ARM}_f{k}_s{seed}/val_predictions.csv` |
| Gate 1 三臂 | 远程 `experiments/ida/gate1/FD002/H{1,3}_f{k}_s101/`、`h2inner/`、`h2_weights.json` |
| 汇总脚本 | `scripts/summarize_gate0.py --seed N`、`scripts/summarize_gate1.py` |
| 训练脚本 | `scripts/run_ida.py`（五臂）、`run_h1/h2inner/h3/h2finalize.py`、`run_moduleBC.py` |
| 模块实现 | `src/candidates/deg_modules.py`（勿动 `model.py`，队列可能仍引用） |
| 报告 | `docs/tim_idc/gate0_report.md`、`gate0_multiseed.md`、`gate1_report.md`、`GATE0_PREREG.md` |
| 工作日志 | `.workbuddy/memory/2026-11-03.md`、`2026-11-04.md`（运维事故全记录） |
| 队列驱动 | 远程 `driver.py <tasks.txt> <status.txt>`；任务格式 `logname\tcommand` |

## 6. 运维陷阱（血泪教训，务必遵守）

1. **relay/watcher 禁用 pgrep 模式串判进程**——三次自匹配死锁（启动 relay 的 `bash -c` 持有 heredoc 全文不退出，模式串必自匹配）。**改用标记文件**：driver 结束时 touch `/home/user/rulb/logs/X/DONE`，relay 轮询该文件
2. **CUDA_VISIBLE_DEVICES 必须显式设**：run_h3/run_h2inner/run_moduleBC 默认 `cuda:0`，relay 忘设会两队列挤一卡
3. **driver 会无声死亡**：每次巡检核对 `wc -l status` vs 任务总数，缺了就补漏队列
4. 后台进程统一 `setsid nohup ... < /dev/null &`；`cd X && A & B` 有 cwd 坑，一律绝对路径
5. 队列运行期间**不得修改被引用的源文件**（code_sha256 一致性）

## 7. 纪律红线（违反 = 推翻整个审计链）

- 判据冻结在 gates.json；门失败 → 停下写 gateN_report.md 等人工，不换指标/子集/阈值
- **禁止窗口级 bootstrap**，一律 engine 聚类（5000 次）；FD001 n=80、FD002 n=259
- ensemble 与单模型分开报告；官方 test 只解封一次
- 汇总脚本只读 metrics/predictions，不手写数字
- 已确认结论不得重复或推翻：LSC 线 hgb 12.98 优于 plain 13.96；EAC G1/G2 NO-GO；Modern Backbone 失败；保留现有 PE 骨干
- 提交纪律：每个判决/报告/脚本一个 commit，message 写明数字；push 前先 `git pull --rebase` 检查

## 8. 时间线预估（从 18:32 起）

| 时间 | 事件 |
|---|---|
| ~20:00 | MC 队列完成 → 跑 summarize_modulebc |
| ~20:30 | Module B/C 判决（保留/删除） |
| 今晚 | Gate 2 校准实验（若判据达标）+ 官方 test 解封准备 |
| 11-05 | 论文方法+实验章成稿；基线复现数字 |
| 11-06/07 | 内审、润色、投稿 |
