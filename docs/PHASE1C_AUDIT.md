# Phase 1c 协议审计（RHSE-TSH 3-seed 网格）

> 日期：2026-11-19。审计者：agent-7（只读；未改代码、未跑 GPU）。
> 范围：计划中的 Phase 1c 网格 —— FD001/FD003 × bias/drift/stuck/noise × mag 2/3/5 ×
> seeds 101/202/303 × sensors 0/5/10，gamma=1.30，onset=0.5；FD001 root=`_gate`，
> FD003 root=`d80`；python=`/home/user/miniconda3/envs/ncmapss/bin/python`。
> 代码冻结核对：`results/FREEZE_PHASE1.json` 的 5 个 SHA256 与工作树**逐位一致**
> （本次用 `Get-FileHash` 复核）。因此下面的 bug 都在**已冻结的 Phase 1 代码**里；
> 修 bug 会改变冻结哈希，必须同时决定是否重跑 Phase 1 锚点。

**总判定：test 封存 PASS（静态证明）；`--seed`/`--root` 的 checkpoint 路径 PASS；
Phase 1c 可启动性 FAIL —— §3 有 8 个 P0 pre-experiment 项，§4 有 6 个会改变
Phase 1c 数字的 P0 bug（B1/B3/B4/B5/B6/B16；另有 1 个 P0* B2 只影响 Phase 1 onset=0.2）。**

---

## 0. 计划网格与真实独立格数

| 块 | 计划格数 | 去重后 | 说明 |
|---|---:|---:|---|
| RUL 网格 | 2 subsets × 4 faults × 3 mags × 3 seeds = **72** | **60** | `stuck` 完全忽略 `mag`（`stress_suite.py:40-42`），每个 (subset, seed) 的 mag 2/3/5 是同一格 → 12 个重复 |
| clean cost | 每个 run 内部重算 | **6** | clean 代价与 fault/mag 无关；每个 (subset, seed) 被重算 12 次 |
| EOL FPR | 2 subsets × 3 seeds = **6** | **2** | `tsh_eol_fpr.py` 加载 `net` 但从不使用（`:25,38-40`）；FPR 只依赖 `rhse`/`tr`/`sd`，与 seed 无关 |
| 合计 | **78** | **62** | 建议在冻结清单里写清 60/72 与 2/6，避免把重复格当独立证据 |

**结论**：72 这个数字本身没错（sensors 是 run 内列表），但**独立 RUL 格只有 60**。
若报告“72 runs × 3 seeds”会被误读为 72 个独立实验；必须写“60 unique cells, 72 executions”。

---

## 1. Test 封存证明（queue/eval 路径从不打开官方 test）

### 1.1 调用链（逐跳）

| 跳 | 证据 | 内容 |
|---|---|---|
| 入口 | `paper_b/modules/tsh_rul_eval.py:30` | `prep(a.subset, a.data_dir, a.seed, Path(a.root))` |
| split 参数 | `paper_b/modules/rhse_rul_eval.py:20-21` | **`split='train'`** 与 **`split='val'`** |
| split 守卫 | `paper_b/hc/data_d.py:78-86` | 只接受 `'train'`/`'val'`；其他值在 `:86` `raise ValueError` |
| 数据构建 | `paper_b/hc/data_hc.py:87` | `prepare_train(subset, data_dir, span=span, cuts=tuple(cuts))` |
| 文件打开 | `paper_b/training/data.py:259` → `:35` | `load_train` → `train_{subset}.txt` |
| test reader | `paper_b/training/data.py:38-40` | `load_test` → `test_{subset}.txt` + `RUL_{subset}.txt` |
| test 调用者 | `paper_b/training/data.py:313-321` | `prepare_test` → `load_test` |
| 协议指定的 test 入口 | `paper_b/evaluation/evaluate_test.py:17,23` | `from ..training.data import prepare_test`；`prepare_test(...)`（`experimental_sensor_retrain/official_test/` 为独立冻结批次） |
| 其他模块 | `rhse_tsh.py`, `stress_suite.py` | 纯函数，无文件 I/O |
| checkpoint | `paper_b/modules/temporal_agg.py:40` | 仅 `torch.load(ckpt)` |

### 1.2 精确的 split 参数（原文）

```python
# paper_b/modules/rhse_rul_eval.py:20-21
tr = engine_batch(subset, data_dir, C.SPAN, C.VAL_CUTS, split='train')
va = engine_batch(subset, data_dir, C.SPAN, C.VAL_CUTS, split='val')

# paper_b/hc/data_d.py:86
raise ValueError("split must be 'train' or 'val', got %r" % (split,))

# paper_b/hc/data_hc.py:87
ds = prepare_train(subset, data_dir, span=span, cuts=tuple(cuts))

# paper_b/training/data.py:259, :35
raw = load_train(data_dir, subset)
return _read_table(Path(data_dir) / f'train_{subset}.txt')
```

### 1.3 仓库中的 test 引用（grep 证据）

搜索模式：`test_FD|test\.txt|RUL_FD|split\s*=\s*['\"]test|prepare_test|load_test`，范围 `paper_b/`。

| 位置 | 引用 | 是否在 Phase 1c 路径 |
|---|---|---|
| `paper_b/training/data.py:39-40` | `test_{subset}.txt`, `RUL_{subset}.txt` | **否**（仅 `load_test` 函数体） |
| `paper_b/training/data.py:317` | `test, rul_map = load_test(...)` | **否**（仅 `prepare_test`） |
| `paper_b/evaluation/evaluate_test.py:17,23` | `prepare_test` | **否**（唯一合规 test 入口） |
| `paper_b/experiments/probe_attribution_length.py:9` | `pd.read_csv(... 'test_FD001.txt' ...)` | **否**（独立诊断脚本，不在 queue） |
| `paper_b/experimental_sensor_retrain/official_test/**` | 历史 test 产物/哈希 | **否**（独立目录） |
| `paper_b/reports/PROTOCOL_AUDIT.md:83` | 文字描述 test 协议 | **否**（文档） |

`tsh_rul_eval.py` 的 import 只有 `predict_at, prep, theta_solve`、`calibrate, detect`、
`inject, valid_span`；`rhse_rul_eval.prep` 只调用 `engine_batch(split='train'/'val')`；
`data_hc.build` 只调用 `prepare_train`；`prepare_train` 只调用 `_prepare_split` → `load_train`。
**没有任何一跳调用 `load_test`/`prepare_test`。**

### 1.4 判定与残余风险

- **PASS（静态证明）**：Phase 1c 的 queue/eval 路径不会打开 `test_FD*`/`RUL_FD*`。
- **CONDITIONAL（运行时）**：仓库里存在其他 test reader（上表），且 `data.py` 同时定义
  `load_test`。静态 grep 不能阻止运行时误调用。建议给 queue 加一个 `open`/`pd.read_csv`
  审计钩子：路径匹配 `test_FD|RUL_FD` 即抛异常，并在每个 job 的日志里记录钩子已启用。
- 数据目录参数是目录而非文件（`--data-dir` 默认 `/home/user/rulb/data/CMAPSS`），
  路径拼接只拼 `train_{subset}.txt`；`prepare_train` 的 docstring 也明确
  “Never opens test files”（`data.py:283`），`protocol.json:36` 也写明 test 只在
  FREEZE 之后由 `evaluate_test.py` 打开。

---

## 2. `tsh_rul_eval.py` 是否正确 honor `--seed` 与 `--root`

### 2.1 路径构造（正确）

| 参数 | 证据 | 实际行为 | 判定 |
|---|---|---|---|
| `--root` | `tsh_rul_eval.py:22,30` → `rhse_rul_eval.py:39` | `ckpt = root / run / 'ckpt' / 'best.pt'`，`root=Path(a.root)` | **PASS** |
| `--seed` | `rhse_rul_eval.py:38` | `run = 'HC-D-B0-%s-s%d-D1' % (subset, seed)` | **PASS**（checkpoint 选择） |
| 加载 | `rhse_rul_eval.py:40` → `temporal_agg.py:40-41` | `torch.load(ckpt, map_location='cpu', weights_only=True)`；`strict=True` | **PASS**（键/形状严格） |

即：`--root` 与 `--seed` 都进入 checkpoint 路径，且加载失败会抛异常（不会静默随机初始化）。

### 2.2 语义问题（不是路径 bug，但影响 3-seed 解释）

| 项 | 证据 | 后果 |
|---|---|---|
| 数据划分不随 `--seed` 变 | `protocol.json:2` `split_seed=7`；`data.py:89-96` | 3 个 seed 共享同一 train/val 划分；不是独立数据重复 |
| 故障注入 RNG 不随 `--seed` 变 | `tsh_rul_eval.py:64` `default_rng(1000 + i*100 + j)` | 3 个 seed 的 bias/drift/noise 实现完全相同；只能测 checkpoint 方差，不能测注入方差 |
| EOL FPR 不随 `--seed` 变 | `tsh_eol_fpr.py:25` 加载 `net`，但 `:38-40` 只用 `rhse`/`dy`；`calibrate` 只用 `tr`/`sd`/`rhse` | 3 个 seed 的 EOL FPR 是同一结果；6 个 job 只有 2 个唯一 |
| 无 checkpoint 存在性预检 | `tsh_rul_eval.py:30` 直接 `prep`；对比 `temporal_agg.py:171-172` 有 `if not ckpt.exists(): raise SystemExit` | 缺 checkpoint 时在 `torch.load` 处失败，错误信息不如显式预检清楚 |
| 无 `config.json`/variant/budget 校验 | `temporal_agg.py:31,40-41` 固定 `hc_config('A1', ...)`，只 strict-load 权重 | 同形状但不同 variant/预算的 checkpoint 可能被静默加载；`d80` 同时有 40ep/80ep 格（`SCHEDULER_STATE.md:119`），三个根预算/代码快照不同（`MECHANISM_NUMBERS_STATUS.md:44`），`run_id` 跨 root 不唯一（`SCHEDULER_STATE.md:102-104`） |
| 硬编码 `cuda:0` | `rhse_rul_eval.py:40,66` | 72 个 job 并发会全部挤 GPU0；没有 `--device` |

### 2.3 判定

- **PASS**：`--seed`/`--root` 对 checkpoint 加载的行为正确。
- **FAIL**：把 3 个 seed 当作“3 次独立实验”不成立；它们是**同一划分、同一注入、同一 cut 上的
  3 个 checkpoint 重复**，只能做配对比较。
- **PASS 前提**：6 个 checkpoint 必须逐格预检（见 §3 P0-1），并记录解析后的绝对路径，
  不能只记 `run_id`（跨 `_gate`/`d80`/`d` 重复）。

---

## 3. 72-run queue + EOL FPR 缺失的 pre-experiment 项

| # | 优先级 | 缺失项 | 为什么现在必须补 | 动作 |
|---|---|---|---|---|
| P0-1 | P0 | **6 个 checkpoint 预检** | FD001 `_gate` 与 FD003 `d80` 是不同 root；FD003 `d80` 的 B0 s101/s202/s303 见 `METRIC_AUDIT.md:316-318`；`run_id` 跨 root 重复（`SCHEDULER_STATE.md:102-104`）；`d80` 同时装 40ep/80ep（`SCHEDULER_STATE.md:119`） | 逐格检查 `ckpt/best.pt` 存在、`config.json` variant=A1、`budget_epochs`/`epochs_run`/`best_ep`、`ckpt` SHA256、`preds.csv` parity；记录绝对路径 |
| P0-2 | P0 | **结构化结果输出** | `tsh_rul_eval.py:117-130` 只 print 4 位小数聚合；没有 config、fault、mag、seed、gamma、sensors、ckpt 路径、git/code SHA | 加 `--out` JSONL：每行 = 配置 + clean R0/R3c + R0..R7 RMSE/recovery + 每 sensor/cut 明细 + 时间戳 |
| P0-3 | P0 | **Phase 1c 冻结清单** | 现有 `results/FREEZE_PHASE1.json` 只有 mag=3、gamma=1.3、onset 0.2/0.5（`:11-21`）；没有 mag 2/5、seeds 202/303、EOL FPR 定义、recovery 公式/聚合规则 | 新建 `results/FREEZE_PHASE1C.json`：网格、判定门、指标定义、代码/ckpt SHA256、排除规则、60/72 与 2/6 计数 |
| P0-4 | P0 | **EOL FPR 定义与 job 数** | `tsh_eol_fpr.py:29-30` 对**全部 5 个 cut** 算“末 20%”，cut=0.4 时是中年不是 EOL；`:46` 的 `any(1).mean()` 是“任一传感器-周期率”，不是“任一 window”；`:47` 的 `mean()` 对 14 个传感器平均，不是监控的 3 个 | 固定主口径（建议 per-sensor-cycle，同时报告 any-sensor-cycle 与真正 any-window）；只用最后 cut 或真实寿命末 20%；按 sensor 分解；输出 JSON；**2 个 job 而非 6 个** |
| P0-5 | P0 | **判定门与功效** | 计划书 §4.3 是“R3 或 R5 ≥50% + clean ≤0.2”（`IMPROVEMENT_PLAN_PILOT:120-121`）；Gate 1 是 ≥0.65（`PHASE1_FINAL:53`）；Phase 1c 没有说明用哪个；3 seeds（df=2）的 95% t 半宽 ≈ 2.48·sd_diff，要分辨 0.10 需要 sd_diff ≤0.04 | 预注册：主指标、聚合方式（pooled vs engine-mean）、排除规则、MDE；按 seed 配对报告，不把 3 seed 均值当显著 |
| P0-6 | P0 | **设备/并发调度** | `rhse_rul_eval.py:40,66` 硬编码 `cuda:0`；72 个 job 同时起会 OOM/串行 | 加 `--device` 或每 worker 设 `CUDA_VISIBLE_DEVICES`；限制并发 ≤GPU 数；预计算/缓存 `prep` 的 KMeans/PCA |
| P0-7 | P0 | **检测指标** | `inject` 返回 `pos`，但 `tsh_rul_eval.py:65` 丢弃（`dy_f, _`），`rhse_rul_eval.py:142` 也丢弃；现在只有下游 RUL recovery，没有 detection delay/TPR/FPR | 记录 first-flag delay、per-cycle TPR/FPR、注入传感器命中率；按 mag 分层；用 `pos` 作正类 |
| P0-8 | P0 | **修 bug vs 可比性决策** | R5 clean leak（§4 B1）会改变 drift 结论；修了就要重跑 Phase 1 锚点；不修则 Phase 1c 继承无效 R5 | 先决定：在分支修 B1/B3/B4/B5/B6/B16，bump freeze，重跑 FD001/FD003 mag=3 s101 锚点并记录 delta；或明确把 R5 标为 oracle、不用于 Gate 1 |
| P1-9 | P1 | clean cost 复用 | 每个 (subset, seed) 重算 12 次；6 个唯一值 | 每 (subset, seed) 算一次，其余 run 断言一致 |
| P1-10 | P1 | 数据 manifest | `--data-dir` 未在计划中固定；test 封存依赖“只读 train” | 记录 `train_FD001.txt`/`train_FD003.txt` SHA256；queue 启动时校验 |
| P1-11 | P1 | 输出/登记/幂等 | 无 run 目录约定、无 `research_task` 登记、无防重跑 | 用 `research_task` 幂等 id；结果目录 `results/phase1c/<subset>/<fault>/mag<m>/s<seed>/` |
| P1-12 | P1 | FD004→FD003 变更说明 | 计划书原文是 FD001/FD004（`IMPROVEMENT_PLAN_PILOT:120`）；新网格改成 FD003 | 在 freeze 里写理由：FD003 是单工况 stuck/drift 证据，FD004 是多工况失败边界；若目标是复现失败边界，FD004 缺失是缺口 |
| P1-13 | P1 | per-sensor/per-cut 明细 | 当前 recovery 是所有 rows 的 pooled RMSE（`tsh_rul_eval.py:124-130`），3 个传感器、5 个 cut 混在一起 | 报告每 sensor、每 cut、每 engine 的 RMSE/recovery；至少给 engine-mean 版本 |

---

## 4. 代码 bug 与协议歧义

严重度：P0 = 改变 Phase 1c 数字/结论；P1 = 影响解释或可复现性；P2 = 边界/清理。

| ID | 严重度 | 位置 | 发现 | 影响 | 修复 |
|---|---|---|---|---|---|
| B1 | **P0** | `tsh_rul_eval.py:97`；`rhse_rul_eval.py:179` | R5 用 **clean** `x[max(0, fb-1), jj]` 作为 last-value，而不是 faulted `x_f[fb-1]`；`fb` 是检测周期，≥ fault onset，`x[fb-1]` 是**不可观测的反事实 clean 值** | R5 是 oracle 泄漏，不是 deployable；drift 的 R5 0.984、R6 晚 onset 0.972 都可能被抬高；“last-value 优于 Hθ”这一 Gate 1 结论可能不成立 | 改用 `x_f[max(0, fb-1), jj]`（最后观测值）；若要保留 clean 版本，明确标为 oracle，不能与 R3 比 |
| B2 | **P0\*** | `tsh_rul_eval.py:90`；`rhse_rul_eval.py:171` | R4 的 hold 起点硬编码 `t0 = first + max(1, (last-first)//2)`（50%），完全忽略 `--onset` | onset=0.2 时 R4 从 0.5 才开始 hold，0.2–0.5 仍是 faulted；`PHASE1_TSH_PILOT:74` 的 “0.912 vs 0.833 (+0.079)” 与 `PHASE1_FINAL:56` 的 onset=0.20 行无效。**Phase 1c 是 onset=0.5，故对 1c 不阻塞** | 传入真实 `onset` 或注入的 `t0`；R4 明确标 oracle |
| B3 | **P0** | `tsh_rul_eval.py:130`；`rhse_rul_eval.py:196` | `(base - rmse) / max(base - clean, 1e-9)`；当 R1<R0 时分母被钳到 1e-9，recovery 变成 ±1e9 | FD001 noise 正是 R1=12.526 < R0=12.647（`PHASE1B_HC_NEGATIVE:77`）；recovery 未定义，均值会被污染 | `if base-clean <= δ: recovery = nan / "no_harm"`；同时报告 ΔRMSE = R_method−R0；排除规则写进 freeze |
| B4 | **P0** | `tsh_eol_fpr.py:29-47`；`rhse_detect_eval.py:82-93` | (a) `tsh_eol_fpr` 对**全部 5 cuts** 算 EOL；simple RHSE 只用 cut=0.95（`rhse_detect_eval.py:82`），所以 1.8% vs 0.32% **不是同一 window set**；(b) `:46` 的 “any-window” 实际是“任一传感器-周期率”；(c) `:47` 对 14 传感器平均，不是 3 个监控传感器；(d) 没有真正的 P(≥1 flag per EOL window)、没有 per-sensor/CI | Gate 1 的“EOL FPR ≤2% 通过”只在 per-sensor-cycle 口径成立；按真正 any-window 口径，FD001 约 1−(1−0.0443)^14 ≈ 47%，FD002 ≈86%（独立性假设下的**推算**，不是实测）；这是典型的事后口径选择 | 固定主口径；只用最后 cut 或真实寿命末 20%；同时报告 per-sensor-cycle、any-sensor-cycle、any-window；按 sensor 分解；输出 JSON；与 simple RHSE 同 window set 再比较 |
| B5 | **P0** | `stress_suite.py:40-42`；`:5-6` | `stuck` 完全不用 `mag`；且 `out[t0:last, sensor] = out[t0, sensor]` 是 **z-space 常数**。真实 stuck 是 raw 物理值常数，经 baseline/sd 后 dy 随工况变，S3l 不会→0 | 72 格中 18 个 stuck 格（2 subsets × 3 seeds × 3 mags）只有 6 个唯一；z-space stuck 的 0.895/0.702 可能无法迁移到物理 stuck；文档已承认（`PHASE1_FINAL:10`、`FREEZE_PHASE1.json:32`） | 定义 stuck severity（或每 (subset,seed) 只跑一次）；加 raw-space stuck 臂：冻结 raw 值，重算 dy；在论文里分开写 |
| B6 | **P0** | `rhse_tsh.py:14,39-41,79-85` | S3l 需要 **12 个 valid cycles** 才变成全 faulted（W=12）；`run>=15` 在 t0+25 达到，2-of-3 在 t0+26 确认，**首次 flag = t0+26**，不是文档写的 15（`PHASE1_TSH_PILOT:160`） | onset=0.5、70-cycle window 时，t0=35，首次 flag ≈61，只剩 ~9 cycles；这比“15 步”更能解释 FD003 stuck 0.440 / FD004 0.702 的下降 | 报告 total delay = 26 cycles（t0+26）；按这个预算重估 stuck 可恢复窗口；或减小 W/run_length |
| B7 | P1 | `rhse_tsh.py:103-107` | `if flag[t].sum() > kmax: keep = argsort(-u[t])[:kmax]` 会把已 latch 的传感器丢掉；下一周期 `prev=flag[t-1]` 已不含它，于是 unlatch | “因果锁存”在 >4 个传感器同时 flag 时不成立；EOL FPR 用 14 个传感器时可能触发 | 单独维护 `latched` 状态；kmax 只限制**新确认**，不丢已 latch |
| B8 | P1 | `rhse_tsh.py:66-71,87-89` | 只有 q3l 做了 `maximum(...,1e-7)`；q1/q2/q3h 若为 0，`S/q` 产生 inf/nan | 近常数/死传感器会永远 flag 或 NaN；C-MAPSS 选出的 14 个传感器大多非恒定，但个别低方差传感器仍可能接近 0 | 所有 q 加 floor；`assert np.isfinite(u).all()` |
| B9 | P2 | `rhse_tsh.py:19,39-41,86-91` | S3l 前 11 个 valid cycles 是 NaN；`np.maximum.reduce([...])` 传播 NaN，导致 `u` 整体 NaN，`u > gamma` 为 False | **S1/S2/S3h 在前 11 个 valid cycles 被 S3l 的 NaN 屏蔽**；onset < ~0.17 的极早期故障会晚检测；Phase 1c onset=0.5 不受影响 | `np.nanmax` 或把 NaN 填 0 再 reduce |
| B10 | P2 | `rhse_tsh.py:22,37` | EWMA `v = np.ones(N)` 暖启动；第一个 valid cycle `S3h=1.0` | 窗口起点可能瞬时触发；clean cost/R3c 可能被污染 | 用第一个 residual 初始化，或丢弃前 W 个 cycle |
| B11 | P1 | `stress_suite.py:32-34`；`rhse_detect_eval.py:106-111` | drift 的 `mag` 是 ramp **终点**，`linspace(0,mag,n)`；`n=last-t0` 随 cut 变，所以同一 mag 的斜率不同；`pos = ramp >= 1.0` 也随 mag 变（mag=2 时从 50% 起，mag=5 时从 20% 起） | “mag 2/3/5”不是可比的 drift severity；delay/TPR 的正类定义随 mag 变 | 定义 slope/cycle 或固定 horizon 的终点；`pos` 用固定绝对阈值或 `ramp>0`；报告两者 |
| B12 | P1 | `stress_suite.py:26` | `t0 = first + max(1, int(onset*(last-first)))`，onset 是 **70-cycle window 的比例**，不是发动机寿命比例 | 同一个 onset=0.5，cut=0.4 是早/中年，cut=0.95 是晚年末；文档把它当同一条件比较 | 定义相对寿命或相对 cut 的 onset；按 cut 分层报告 |
| B13 | P1 | `tsh_rul_eval.py:103-104` | R6 阈值 `onset_est < 0.35` 是在同一批 FD001 drift 数据上选的；`onset_est` 由检测周期反推，检测 ≥ 真 onset | R6 0.972 是 in-sample 选择；不是可部署规则 | 在 training engines 上校准阈值；给 0.25–0.45 敏感性；或标为 oracle-onset |
| B14 | P2 | `rhse_rul_eval.py:26-31` | `late` 用 f=0.7/0.85/0.95 的**整个 70-cycle window**，f=0.7 时包含早期 cycle | H/sd 不是严格 late-life；影响所有臂的归一化 | 只用最后 L 个 cycle，或按 age 加权；报告敏感性 |
| B15 | P1 | `temporal_agg.py:40`；`protocol.json:5-11` | `best.pt` 是在 **同一组 5 个 val cuts（0.4–0.95）** 上按 pooled validation RMSE 选的（`docs/DSPT_SENSOR_MODULE_DESIGN_20261114.md:20`）；Phase 1c 又在这 5 个 cuts 上评估 | R0/R1/R2/... 对模型选择是 in-sample；3 seeds 不能修复 | 用 disjoint selection cuts（如 0.4–0.7 选，0.85/0.95 评），或在论文里明确写“in-sample validation”并加限制 |
| B16 | **P0** | `tsh_rul_eval.py:26`；`FREEZE_PHASE1.json:17` | `--gamma` 默认 **1.0**，冻结协议是 **1.30** | 队列若漏传 `--gamma 1.30`，所有数字静默变成 gamma=1.0 的结果 | 把 `--gamma` 设为 required，或启动时 assert 1.30 |

---

## 5. PASS/FAIL checklist

| # | 检查项 | 判定 | 证据 |
|---|---|---|---|
| 1 | Phase 1c eval 路径从不打开官方 test | **PASS**（静态） | §1；`rhse_rul_eval.py:20-21`、`data.py:259` |
| 2 | 运行时 test-seal 强制 | **FAIL** | 无审计钩子；仓库有 `probe_attribution_length.py:9` 等 test reader |
| 3 | `--root` 正确进入 checkpoint 路径 | **PASS** | `tsh_rul_eval.py:30`、`rhse_rul_eval.py:39` |
| 4 | `--seed` 正确进入 checkpoint 路径 | **PASS** | `rhse_rul_eval.py:38` |
| 5 | 3 seeds 是独立数据/注入重复 | **FAIL** | 固定 `split_seed=7`、固定注入 RNG `1000+i*100+j` |
| 6 | 6 个 checkpoint 预检（路径/config/budget/SHA） | **FAIL** | 无 manifest；`run_id` 跨 root 重复 |
| 7 | 72-run 网格唯一 | **FAIL** | 60 unique；stuck 忽略 mag |
| 8 | EOL FPR 定义已冻结 | **FAIL** | B4；全 5 cuts、any-window 误名、14 vs 3 sensors |
| 9 | EOL FPR job 数最小 | **FAIL** | 6 计划 / 2 唯一；net 未使用 |
| 10 | R1<R0 时 recovery 有定义 | **FAIL** | B3；`max(...,1e-9)` |
| 11 | R5 是 deployable（无 clean 泄漏） | **FAIL** | B1；`x[fb-1]` |
| 12 | R4 oracle onset 正确 | **FAIL**（Phase 1c onset=0.5 不受影响） | B2；硬编码 50% |
| 13 | stuck 注入是物理的 | **FAIL** | B5；z-space + 忽略 mag |
| 14 | stuck 延迟报告正确 | **FAIL** | B6；≈26 不是 15 |
| 15 | 结构化结果输出 | **FAIL** | `tsh_rul_eval.py:117-130` 只 print |
| 16 | 设备/并发调度 | **FAIL** | `rhse_rul_eval.py:40,66` 硬编码 cuda:0 |
| 17 | 统计计划/功效已预注册 | **FAIL** | 无 MDE/聚合规则；3 seeds df=2 |
| 18 | Phase 1c 冻结清单 | **FAIL** | 只有 `FREEZE_PHASE1.json`（mag=3） |
| 19 | 代码冻结完整性 | **PASS** | 5 个 SHA256 与 `FREEZE_PHASE1.json:36-40` 一致 |
| 20 | 检测指标（delay/TPR/FPR） | **FAIL** | `pos` 被丢弃（`tsh_rul_eval.py:65`） |
| 21 | clean cost 复用 | **FAIL**（效率） | 72 次 vs 6 唯一 |
| 22 | FD004→FD003 变更已说明 | **FAIL** | 计划书 `IMPROVEMENT_PLAN_PILOT:120` 是 FD001/FD004 |

**总计：PASS 4 / FAIL 18。另有 §4 的 6 个影响 Phase 1c 数字的 P0 bug（B1/B3/B4/B5/B6/B16）与 1 个 P0*（B2，仅影响 Phase 1 onset=0.2）；§3 有 8 个 P0 pre-experiment 项。**

---

## 6. 最小启动门（P0，按顺序）

1. **先决定 bug-fix vs 可比性**：在分支修 B1（R5 clean leak）、B3（recovery 分母）、
   B4（EOL 定义）、B5（stuck）、B6（delay）、B16（gamma 默认）；若 onset≠0.5 还要修 B2。bump freeze，
   重跑 FD001/FD003、mag=3、s101、gamma=1.30、onset=0.5 的 Phase 1 锚点，
   记录修前/修后 delta，并明确是否撤回“drift 上 last-value 更好”的结论。
2. **写 `results/FREEZE_PHASE1C.json`**：网格、判定门、指标定义、代码/ckpt SHA256、
   排除规则、60/72 与 2/6 计数、FD004→FD003 的理由。
3. **6 个 checkpoint 预检**：FD001 `_gate` s101/202/303、FD003 `d80` s101/202/303；
   逐格记录绝对路径、`config.json` variant=A1、`budget_epochs`、`epochs_run`、`best_ep`、
   `ckpt` SHA256、`preds.csv` parity；任一不符即 abort。
4. **加 JSONL 日志 + 检测指标**：每 run 一行配置+指标；每 sensor/cut/engine 明细；
   first-flag delay、per-cycle TPR/FPR、注入传感器命中率；`pos` 不再丢弃。
5. **修 EOL FPR 脚本**：只用最后 cut 或真实寿命末 20%；per-sensor-cycle +
   any-sensor-cycle + 真正 any-window；per-sensor；14 vs 3 监控传感器；JSON；
   与 `net` 解耦；只跑 2 个 job（FD001/FD003，seed 101）。
6. **设备/并发**：加 `--device` 或每 worker `CUDA_VISIBLE_DEVICES`；限制并发；
   先跑 1 个 smoke（FD001 bias mag=3 s101），要求 clean cost ≈+0.024、R3 ≈0.862
   （在合理容差内）再放 60 格队列。
7. **去重**：stuck 只跑 1 个 mag（或定义 severity）；EOL FPR 只跑 2 个唯一 job；
   在 freeze 里写清 60/72、2/6。

---

## 附录 A：代码冻结哈希（本次复核）

| 文件 | SHA256 | 与 `FREEZE_PHASE1.json` |
|---|---|---|
| `paper_b/modules/stress_suite.py` | `ae2673c8…aa67de` | 一致 |
| `paper_b/modules/rhse.py` | `c6817f45…98de38` | 一致 |
| `paper_b/modules/rhse_rul_eval.py` | `5750527f…e0d0d7` | 一致 |
| `paper_b/modules/rhse_tsh.py` | `4e05277b…adeac4d` | 一致 |
| `paper_b/modules/tsh_rul_eval.py` | `c64cb02f…14e909` | 一致 |

## 附录 B：本次读过的文件

必需：`docs/PHASE1_FINAL_20261119.md`、`docs/paper_b_modules/PHASE1_TSH_PILOT_20261119.md`、
`docs/paper_b_modules/PHASE1B_HC_NEGATIVE_20261119.md`、`paper_b/modules/tsh_rul_eval.py`、
`paper_b/modules/rhse_tsh.py`、`paper_b/modules/stress_suite.py`、`paper_b/modules/rhse_rul_eval.py`。

支撑：`paper_b/modules/rhse.py`、`paper_b/modules/tsh_eol_fpr.py`、
`paper_b/modules/rhse_detect_eval.py`、`paper_b/modules/temporal_agg.py`、
`paper_b/hc/data_d.py`、`paper_b/hc/data_hc.py`、`paper_b/hc/config_hc.py`、
`paper_b/training/data.py`、`paper_b/configs/protocol.json`、`results/FREEZE_PHASE1.json`、
`paper_b/evaluation/evaluate_test.py`、`paper_b/experiments/probe_attribution_length.py`、
`docs/paper_b_modules/IMPROVEMENT_PLAN_PILOT_20261119.md`、
`docs/paper_b_hc/MECHANISM_NUMBERS_STATUS.md`、`docs/paper_b_edc/sprint_24h/SCHEDULER_STATE.md`、
`docs/paper_b_edc/sprint_24h/METRIC_AUDIT.md`、`docs/paper_b_edc/sprint_24h/RUN_QUEUE.csv`。

## 附录 C：一句话

**Test 封存和 checkpoint 路径是干净的；但 Phase 1c 现在不能启动：R5 用了 clean 泄漏、
recovery 在 R1<R0 时爆掉、EOL FPR 口径未定、stuck 的 mag 是空转且延迟被低估、
72 格里只有 60 格唯一、结果没有结构化落盘。先修这 6 个 P0，再谈 3-seed 网格。**
