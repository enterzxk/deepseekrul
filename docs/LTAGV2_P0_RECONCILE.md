# LTAG v3 Phase 0：口径对账与共享 R0

> 日期：2026-11-20。依据：`ltag_v3_execution_plan.md` §2。
> **test 封存**；本文件只做验证集口径对账。

## 1. T0.1 12.647 vs 11.696

用同一份 `experiments/hc/cata_gate/preds/FD001_s101.csv`（B0 s101，5 cuts，20 engines）：

| 量 | 值 |
|---|---:|
| `pred_k1` engine_mean | **11.558** |
| `pred_k1` pooled | **12.647** |
| `sd_e(RMSE_e)` | 5.268 |
| 恒等式 `pooled² − engine_mean² − Var_e` | **−0.000000** |

- **12.647 是 s101 的 pooled RMSE**；**11.696 是 6 个 seed 的 engine_mean 均值**
  （当前文件算得 **11.706**，差 0.010，来自 seed 集合/舍入）。
- 两者不是同一 estimand，不矛盾；恒等式在数值上精确成立。
- 结论：**PASS**。后续所有表必须标注 `estimator`，不同 estimator 不得相减。

### 共享 R0（B0，验证集，K=1）

| subset | seeds | n_engines | K1 engine_mean | K1 pooled | LTAG gate engine_mean |
|---|---|---:|---:|---:|---:|
| FD001 | 101/202/303/404/505/606 | 20 | **11.706** | 13.012 | 11.082 |
| FD002 | 101/202/303 | 52 | **11.443** | 12.810 | 11.399 |
| FD003 | 101/202/303 | 20 | **12.634** | 13.972 | 12.787 |
| FD004 | 101/202/303 | 50 | **13.538** | 18.463 | 13.807 |

- 写入 `results/SHARED_R0.json`；字段含 backbone/split/cuts/seeds/n_engines/n_rows/estimator。
- FD001 有 6 seeds，FD002–FD004 只有 3 seeds；后续 `Ens3` 只用 101/202/303。

## 2. T0.2 六项不一致的裁决

| # | 项 | 裁决 |
|---|---|---|
| 1 | FD002 clean cost +0.227 vs +0.132 | **+0.227 是 s101 单 seed；+0.132 是 3-seed 均值**。Phase 1c 表用 3-seed +0.132；单 seed 数字标注 `s101`。 |
| 2 | FINAL §5 “R4” 行 | 数值等于 R3−R5；**改标 R3−R5**，不称 R4。 |
| 3 | EOL FPR estimand | 同时报告 **per-sensor-cycle** 与 **any-window**；Gate 用 per-sensor-cycle，论文 Methods 必须声明。 |
| 4 | Stuck pre/post run-length | FD001 0.927→0.895、FD002 −0.047→0.167、FD003 0.588→0.440、FD004 0.964→0.702；一律标 `pre-run-length` / `post-run-length`。 |
| 5 | FD002/FD004 bias pilot vs FINAL | 引用 **Phase 1c 3-seed**：FD002 0.913/0.844、FD004 0.801/0.502；pilot 单 seed 不进主表。 |
| 6 | LTAG vs RHSE 基线 | 两者 estimator 不同（LTAG engine_mean vs RHSE pooled）；**在 `SHARED_R0.json` 建立共同 R0 之前，不做联合表**。 |

## 3. T0.3 预测缓存（进行中）

- 已确定缓存 schema：`engine, cut, t_end, pred, y_cap, y_raw, cond, cycle, valid_len`。
- 先做 **B0 FD001 s101 的 cuts-only 缓存**与 parity A0.1（K=1，缓存 pred vs `preds.csv` <1e-4）；
  Δ∈{5,10,20,30} 的窗口扫描在 Phase 1 前补齐。
- 缓存目录：`cache/preds/{backbone}/{subset}_s{seed}.parquet`。
- **A0.1 未通过前，Phase 1 不得启动。**

## 4. Phase 0 验收

| 项 | 结果 |
|---|---|
| 恒等式误差 <0.05 | **PASS（−0.000000）** |
| seed 集合对齐 | **PASS**（FD001 6 seeds；其余 3 seeds） |
| 6 项不一致有裁决 | **PASS** |
| `metrics_v2.py` | **已写**：engine_mean/pooled/MAE/bias/NASA/NASA split/paired bootstrap/sign_count |
| 预测缓存 + parity A0.1 | **进行中** |
| test 封存 | **PASS**（未读 test） |

**Phase 0 尚未完全关闭**：缓存 parity 未跑完前，不进入 Phase 1。
