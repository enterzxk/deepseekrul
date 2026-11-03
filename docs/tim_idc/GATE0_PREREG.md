# Gate 0 预注册：因果寿命先验的最小可用性（IDA-PE）

冻结时间：2026-11-03（先于任何 Gate 0 训练运行）。判据同时机读冻结于 `gates.json`。
冲突时以本文件与 gates.json 为准；修改判据必须新开 amendment 文件，禁止原地改。

## 0. 研究主线（本轮唯一主线）

Causal Lifetime Prior → Individualized Degradation Clock (tau = t / mu_T)
→ Degradation-Conditioned PE Predictor（IDA-PE）。

继承结论（不重复实验）：
- PE backbone（CandidateModel, num_layers=1, use_pe=True, EMA 特征, 42,306 参数）
  是不可删除的强基线（单 split FD002 val 12.203）。
- EAC/FiLM（G1 NO-GO：shuffled-E 保留 71% 增益）与现代化骨干（G2：V4 退化
  ~1.5 RMSE）已判负，不再触碰。
- LSC 遗产：外部 HGB 先验（corr 0.81）显著优于端到端联合训练（12.98 vs 15.35，
  plain 13.96，oracle 10.12）→ 本轮把该先验严格化（嵌套 + 阶梯因果）后重测。

## 1. 寿命先验构造（严格因果 + 嵌套交叉拟合）

- 特征：`lsc_feasibility.py` 的 sensor 臂特征族（age + ops 统计 + 传感器统计，
  全部来自 X[1:c]），复用前逐字引用不改动。
- cutoff 网格 c ∈ {50, 60, ..., 200}，c < 机组观测长度 T。
- **外层 5 折**（engine 级 KFold(5, shuffle, seed=20261104)，sorted units）：
  - outer-train 内 **inner 4 折交叉拟合** → outer-train 每机组每 cutoff 的
    OOF 寿命预测（5 个 HGB seed → mu_T=mean，sigma_T=std）；
  - outer-val：只在 outer-train 上训练的 5-seed ensemble → mu_T, sigma_T；
  - OperatingConditionScaler 只在 outer-train fit（FD002/FD004 n_cond=6）。
- **阶梯因果（staircase）**：窗口末端 t 消费最近的 cutoff c ≤ t 的预测；
  禁止向未来 cutoff 插值；t < 50 → outer-train 中位寿命 + flag 标记。
- 泄漏单测（`tests/test_prior_causality.py`，合成数据，全部通过后才能开跑）：
  - T1 篡改机组 cycle>t 的传感器 → cutoff≤t 特征与预测逐位不变 ✓
  - T2 篡改 outer-val 机组真寿命 → 同 outer fold 的 outer-train mu/sigma 不变 ✓
  - T3 篡改 outer-val 机组真寿命 → 该机组自身共同 cutoff 的 mu/sigma 不变 ✓
  - T4 断言所有 consumed cutoff ≤ 窗口末端；阶梯单调性（无插值）✓
  - 附加：engine 聚类统计量手算对照单测 ✓

## 2. Gate 0 臂（全部 zero-init 注入，初始严格等价 M0）

| 臂 | 注入 | 参数 |
|---|---|---|
| M0 | 无（PE 原样） | 42,306 |
| M1 | [mu/250, sigma/250] → zero-init Linear(2→64) 加到 pooled context | +192 |
| M1p | phi(tau)=sin/cos×4 周期 [0.125,0.25,0.5,1.0]（tau=t/mu）+ u=sigma/mu → zero-init Linear(9→64) | +640 |
| M1_shuf | M1 但 mu/sigma 在 fold 内 engines 间 permutation（seed 777+fold） | +192 |
| M1_const | M1 但 mu=fold 训练中位寿命， sigma=0 | +192 |

M1 vs M1p 回答 Q3（是不是加个特征就够了：concat mu vs 个体退化坐标 tau）；
M1_shuf / M1_const 回答 Q4（是不是任何寿命尺度都有效）。

## 3. 预注册判据（FD002 为主，FD001 护栏；engine 聚类 bootstrap，5000 reps）

- **G0.1** M1 vs M0（FD002）：ΔRMSE 95% CI 上界 < 0
- **G0.2** M1_shuf vs M1（FD002）：CI 下界 > 0
- **G0.3** M1_const vs M1（FD002）：CI 下界 > 0
- **G0.4** M1 vs M0（FD001）：CI 上界 < +0.5（护栏）

四条全过 → 进入 Gate 1（消费方式 + H1/H2/H3 对照）。任一失败 → 停止，
输出 gate0_report.md（各 cutoff corr(mu_T,T) 曲线 + 新构造下 hgb 臂结果），
不做任何模块，等待人工。

## 4. 执行计划

- Stage A（方向判断）：seed 101，FD002 全 5 臂 × 5 折 + FD001 M0/M1/M1p × 5 折
  （40 runs，双 3090 约 1.5-2 小时）。
- Stage B（冻结正式数字）：Stage A 通过后扩 3 seeds（101/202/303），
  以 3-seed 汇总出正式 CI。

## 5. 运行纪律

- 队列运行期间不修改 run_ida.py / nested_prior.py（修复必须新版本 + 重跑受影响任务）。
- 每个结果目录落 metrics.json + val_predictions.csv（engine 级窗口明细）。
- 汇总（summarize_gate0.py）只读 CSV，自动判定判据；engine 聚类 bootstrap
  已有手算对照单测。
- 统计单位一律是 engine；窗口级 bootstrap 不得出现在任何结论。

## 6. 2026 基线核实（独立子代理，2026-11-03）

| 论文 | 期刊 | 官方代码 | 复现可行性 |
|---|---|---|---|
| AIDGN-RUL (Abbas et al.) | IEEE TII 2026 | 无 | 中（需自行实现） |
| LVDACNN (Hou et al.) | **IEEE TIM 2026** vol.75 | 无 | 中（结构清晰） |
| MSMT-FDA (Qin et al.) | RESS 2026 vol.272 | 无 | 中 |
| CET-MSA | J. Intelligent Manufacturing 2026 | **有** | 高 |
| BootBoGs-TS | IJCIS 2026 | **有** | 高（协议披露最全：cap 125/130） |

5 篇全部真实存在。Table A（引文成绩）/ Table C（统一协议）分层不变；
优先 BootBoGs-TS + CET-MSA 官方代码复现，LVDACNN 作为同刊自实现基线。
