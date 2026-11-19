# Phase 1c 结果（进行中，2026-11-20）

> 队列：`scripts/queue_phase1c.sh`，60 RUL + 2 EOL。**test 封存。**
> 当前进度：23/62（RUL 21/60，EOL 2/2，0 failed）。本文件在队列跑完后更新为终版。

## 1. 协议

- FD001/FD003；bias/drift/stuck/noise；mag 2/3/5（stuck 仅 3）；
  seeds 101/202/303；sensors 0/5/10；gamma=1.30；onset=0.5。
- 3 seeds 是同一划分/注入的 checkpoint 重复，不是独立数据重复。
- 主指标 R3（TSH+Hθ）；R5（TSH+last-value）为弱 deployable 对照；
  R6 已撤回（R5 修复后 hybrid 把晚漂移路由到弱 last-value）。

## 2. FD001 已完成 cell（3-seed）

| fault | mag | n | R3 mean (min–max) | R5 mean | R3−R5 | clean cost |
|---|---:|---:|---|---:|---:|---:|
| bias | 2 | 3 | 0.784 (0.655–0.884) | 0.053 | +0.731 | −0.010 |
| bias | 3 | 3 | 0.826 (0.702–0.913) | 0.078 | +0.748 | −0.010 |
| bias | 5 | 3 | 0.782 (0.625–0.892) | 0.015 | +0.767 | −0.010 |
| drift | 2 | 3 | 0.700 (0.605–0.748) | 0.269 | +0.431 | −0.010 |
| drift | 3 | 3 | 0.797 (0.667–0.862) | 0.362 | +0.435 | −0.010 |
| drift | 5 | 3 | 0.814 (0.667–0.894) | 0.363 | +0.451 | −0.010 |
| stuck | 3 | 3 | **0.872** (0.851–0.895) | −0.037 | +0.908 | −0.010 |

### noise（mixed：2 no_harm + 1 damaging per mag）

| mag | no_harm seeds | damaging seeds | R7 RMSE（no_harm seeds） | R3（damaging） |
|---:|---:|---:|---:|---:|
| 2 | 2 | 1 | **12.248** | −2.040 |
| 3 | 2 | 1 | **12.229** | −2.297 |
| 5 | 2 | 1 | **12.157** | −2.594 |

- 2/3 seeds 的 `R1−R0≤0.05`，故障不伤害 B0；1/3 seed 仍伤害，recovery 为负。
- **noise 从 recovery Gate 单列**：不计入 R3≥0.65；论文报告 R7 的绝对 RMSE 和
  “noise 在 mag≤5 内多数 seed 不造成伤害”。
- R7 因果中值滤波在 no_harm seeds 上稳定在 12.16–12.25，略优于 clean；这是探索性观察，
  不写成主贡献。
- 提交：`0ad2fe21`、`5b6d8597`。

- EOL：FD001 0.24% per-sensor-cycle / 5.0% any-window；FD003 0% / 0%。
- 所有 FD001 非 noise cell 的 R3 均值 ≥0.65；R3 全面领先 R5。
- **seed spread 是 checkpoint 驱动**：s303 的 clean R0 12.887（s101 12.647、
  s202 12.424），单 seed R3 最低 0.605–0.702；不得事后剔除，论文报 mean/min/max。

## 2b. Per-seed clean cost（关键）

clean cost 与 fault/mag 无关，只随 (subset, seed) 变。逐 seed：

| subset | seed | R0 | R3c | cost | Gate ≤0.15 |
|---|---:|---:|---:|---:|---|
| FD001 | 101 | 12.647 | 12.641 | −0.006 | PASS |
| FD001 | 202 | 12.424 | 12.394 | −0.030 | PASS |
| FD001 | 303 | 12.887 | 12.894 | +0.007 | PASS |
| FD003 | 101 | 14.299 | 14.448 | +0.149 | PASS（贴边） |
| FD003 | 202 | 14.507 | 14.466 | −0.041 | PASS |
| FD003 | 303 | 13.110 | 13.364 | **+0.254** | **FAIL** |

**FD003 s303 的 clean cost 0.254 > 0.15，Gate 1 的 clean 门不通过。**  
这不是某个 fault/mag 的问题：clean cost 在注入前计算，对所有 FD003 cell 相同。  
s303 的 clean R0 更低（13.110），但 TSH 误判/重构把它抬到 13.364；同一 gamma=1.30 下
FD001 三 seed 都过，FD003 s303 不过。**不能事后剔除 s303，也不能按子集调 gamma 来救。**

### FD003 bias（partial, 3-seed）

| fault | mag | n | R3 mean (min–max) | R5 mean | clean cost |
|---|---:|---:|---|---:|---|
| bias | 2 | 3 | 0.798 (0.686–0.923) | 0.277 | s303 0.254 **FAIL** |
| bias | 3 | 3 | **0.876** (0.780–0.967) | 0.153 | s303 0.254 **FAIL** |
| bias | 5 | 3 | **0.952** (0.914–0.997) | 0.015 | s303 0.254 **FAIL** |
| drift | 2 | 3 | **0.640** (0.583–0.739) | 0.319 | s303 0.254 **FAIL** |
| drift | 3 | 3 | **0.772** (0.736–0.840) | 0.494 | s303 0.254 **FAIL** |
| drift | 5 | 3 | **0.841** (0.777–0.919) | 0.493 | s303 0.254 **FAIL** |
| noise | 2 | 3 | **0.413** (0.025–0.796) | −2.618 | s303 0.254 **FAIL** |
| noise | 3 | 3 | 0.653 (0.513–0.801) | −2.260 | s303 0.254 **FAIL** |
| noise | 5 | 3 | 0.633 (0.573–0.723) | −1.837 | s303 0.254 **FAIL** |
| stuck | 3 | 3 | **0.570** (0.338–0.932) | −0.244 | s303 0.254 **FAIL** |

- FD003 noise 与 FD001 不同：R1>R0（噪声伤害 B0），R3 0.41–0.65，R5 为负；
  noise 在 FD003 是 damaging，但仍低于 0.65（mag3 0.653 贴边）。
- FD003 stuck 0.570（0.338–0.932），15 步 run-length 的延迟 + 短窗口导致高方差、均值 <0.65。
- **clean 失败适用于所有 FD003 cell**（clean cost 在注入前算，与 fault/mag 无关）。

## 3. 最终 Gate 1（62/62，0 failed）

| 条件 | 结果 |
|---|---|
| clean cost ≤0.15 | **FAIL：FD003 s303 0.254** |
| EOL per-cycle ≤2% | PASS（FD001 0.24%、FD003 0%） |
| FD001 bias/drift/stuck | **全部 PASS** |
| FD003 bias | PASS（0.798 / 0.876 / 0.952） |
| FD003 drift mag3/5 | PASS（0.772 / 0.841） |
| FD003 drift mag2 | **FAIL（0.640）** |
| FD003 stuck | **FAIL（0.570）** |
| FD003 noise | damaging，0.413–0.653，未过 0.65 |
| noise | FD001 no_harm/mixed；FD003 damaging |

**Gate 1：FAIL**，三个独立原因：clean s303、FD003 drift mag2、FD003 stuck。

## 4. 当前判定

- **FD001 单工况条件性成立**：bias/drift/stuck 全部 PASS，clean 三 seed 全过，EOL 0.24%。
- **FD003 是幅度/故障类型/seed 边界**：bias 与大漂移可用；小漂移、stuck、clean s303 失败；
  noise 在 FD003 是 damaging，R3 0.41–0.65。
- **Phase 2 不启动**；路线取舍 PENDING-L2；test 封存。

**一句话：62/62、0 failed；Gate 1 FAIL 于 FD003 s303 clean 0.254、drift mag2 0.640、stuck 0.570；
FD001 单工况条件性成立，FD003 只能写幅度/类型/seed 边界，Phase 2 暂缓。**
