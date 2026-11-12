# Paper B · Run 盘点与三套 Base 定义对账

状态：已完成（第一版）
最后更新：2026-11-12 14:10 (+08)
产出人：实验结果管理员
用途：用户要求 1 的前置（哪些 run 有 `test_pred.csv`、哪些只有 `metrics.json`）+ 要求 6（单种子 A8 / 三种子 Factorial / 历史论文三套 Base 的字段级 diff，找「同名不同义」）。
边界：**只读**。未修改任何实验产物、未重算任何被审文件里的数字，本文件是新增文件。

---

## 0. 结论摘要

1. **权威全量产物在 lab3090**，不在本地、也不在 a40：`/home/user/rulb/paper_b/results/raw/`
   —— **265 个 run**（每个都有 `metrics.json` / `val_pred.csv` / `model.pt`），其中 **113 个有 `test_pred.csv` + `test_metrics.json`**。
   这与主控口径一致（113 / 265）。
2. **本地 Windows 仓库只有 83 个 run（31%）**，且 83 个全部有 test。**本地是部分镜像，不是全量**；
   拿本地状态判断「有没有这个 run」会漏 182 个。
3. **A8 八配置 96 格产物齐全**（8 配置 × 4 子集 × 3 seed），`a8_engine_level.csv` 与产物逐位吻合（抽验 3/3）。
   但 **A8 的 test 覆盖是 36 格，不是报告写的 32 格**：多出 4 格在冻结清单外。
   **已裁定（2026-11-12）**：这 4 格标 `n/a`、不进表；A8 汇总用 32 格，全库双口径汇总用 109 格。
4. **三套 Base 同名不同义**，至少 5 处陷阱（见 §5.2）。最严重的一处：A8 的 `base` 和 Factorial 的 `base`
   是**两个不同的模型**（参数量 77058 vs 81218）。FD001 三 seed 同口径差：
   engine-level **11.0851 vs 11.4316（差 0.35）**，pooled **11.8739 vs 12.0932（差 0.22）**。
   > 注：**不能拿 11.085 和 12.093 相减说「差 1.0」**——前者是 engine-level，后者是 pooled，
   > 这正是本文件 §5 要防的口径混用。
5. **152 个 run 没有 test_pred.csv**，全部无法参与双口径重算（它们只做过验证集评估，符合「测试集冻结后才开」的纪律）。

---

## 1. 三台机器的产物布局（对账表）

| 位置 | 布局 | run 数 | 有 test | 说明 |
|---|---|---:|---:|---|
| **lab3090** `/home/user/rulb/paper_b/results/raw/` | `<tag>/<subset>/s<seed>/` | **265** | **113** | **权威全量**。A8 96 格、VB/VAN、MSM/SAM/HIM、F_* 三套都在这里 |
| **本地** `paper_b/results/raw/`（Windows 仓库） | 同上 | 83 | 83 | 部分镜像（31%）。24 个 seed-202/303 的 F_* run **只有本地有全量** |
| **a40** `<HC_ROOT>/paper_b/results/raw/` | 同上 | 59 | 59 | 只有 s101 层，缺 seed 202/303 |
| **a40** `<HC_ROOT>/experiments/hc/` | `HC-<run_id>/`（HC 协议布局） | 67 个目录 | **0** | 44 个完整（config/manifest/metrics/preds/ckpt/train.log）；**测试集未开，符合纪律** |

> 三套布局并存是历史原因：`paper_b/results/raw/` 是 R2 之前的旧管线，`experiments/hc/` 是 HC 协议的新管线。
> 两者**不能按 tag 混算**，`runs.csv` 里用 `stage` 和 `legacy_metric` 区分。

**镜像缺口（重要）**：

- 本地 83 个 vs lab3090 265 个 → 本地缺 182 个 run。
- 本地有而 a40 没有的：**25 个**（24 个 seed-202/303 的 F_* + `A8_base_L70/FD001/s101`）。
  这 25 个的产物**只存在于本地和 lab3090**，a40 上没有副本。
- `FREEZE_ABLATION_SEED101.json` **只在 lab3090**，本地和 a40 都没有 → 本地无法复核 A8 的冻结顺序。

---

## 2. 逐族产物矩阵（lab3090 权威口径）

| family | run 数 | 有 test | 缺 test 的原因 |
|---|---:|---:|---|
| `A8_base_L70` | 12 | 6 | s101 四子集 + FD001 s202/s303 |
| `A8_full_L70` | 12 | 4 | 只有 s101 |
| `A8_hi_L70` | 12 | 4 | 只有 s101 |
| `A8_ms_L70` | 12 | 4 | 只有 s101 |
| `A8_mshi_L70` | 12 | 4 | 只有 s101 |
| `A8_mssa_L70` | 12 | 6 | s101 四子集 + FD001 s202/s303 |
| `A8_sa_L70` | 12 | 4 | 只有 s101 |
| `A8_sahi_L70` | 12 | 4 | 只有 s101 |
| `F_single_proj_mean_hi0` | 12 | 12 | —（三种子全测） |
| `F_single_proj_basic_hi1` | 12 | 12 | —（三种子全测） |
| `F_full_basic_hi1` | 12 | 12 | —（三种子全测） |
| `VAN_raw_L70` | 12 | 3 | 只有 FD001 三 seed |
| `VAN_dsp_cycle_L70` | 12 | 3 | 只有 FD001 三 seed |
| `NC_base_L70` | 12 | **0** | 未进冻结清单 |
| `SAM_identity_L70` / `SAM_cond_L70` | 12 / 12 | **0** | 未进冻结清单 |
| `MSM_single_proj_L70` / `MSM_multi_noproj_L70` / `MSM_multi3_L70` | 6 / 6 / 6 | **0** | 未进冻结清单 |
| `HIM_hf_wrong_L70` / `HIM_hf_nomono_L70` / `HIM_hB_correct_L70` | 6 / 6 / 6 | **0** | 未进冻结清单 |
| `F_*_hi0` / `F_*_mean_hi1` 等单种子格 | 24（6 族 × 4） | 24 | —（单种子全测） |
| `C_single_static_hi0` / `C_linear_mean_hi0` | 4 / 4 | 4 / 4 | — |
| `ATTR_*`（7 族） | 7 | 7 | — |
| `REPRO_base` / `REPRO_full` | 1 / 1 | **0** | 确定性检查，不测 |
| **合计** | **265** | **113** | **152 个只有 metrics.json** |

**对要求 1 的直接含义**：双口径（capped / raw）重算只对这 **113 个** run 成立。
其余 152 个 run 没有 `test_pred.csv`，无法重算，也不能用 `metrics.json` 里的 `test_*` 反推
（`metrics.json` 没有逐样本预测）。**这 152 个 run 不应出现在任何 capped/raw 对照表里。**

---

## 3. A8 96 格专项核验

### 3.1 产物在，且能复算

- 96 个目录（8 配置 × 4 子集 × 3 seed），每个都有 `config.json` / `metrics.json` / `val_pred.csv` / `model.pt`。
- **抽验 3 格独立复算**（用 `val_pred.csv` 重算，不读 CSV 表）：

| 格 | 复算 engine-mean | 复算 pooled | `metrics.json.rmse` | 结论 |
|---|---:|---:|---:|---|
| `A8_ms_L70 / FD003 / s101` | 11.683283 | 12.356788 | 12.356788 | ✅ 逐位一致 |
| `A8_sahi_L70 / FD004 / s303` | 12.443366 | 16.903265 | 16.903265 | ✅ 逐位一致 |
| `A8_hi_L70 / FD002 / s202` | 10.534477 | 11.846844 | 11.846844 | ✅ 逐位一致 |

  → `a8_engine_level.csv` 的 `val_pool` 列 = `metrics.json.rmse`，`val_eng` 列 = 后算的发动机级均值，
  两列都可从产物复现。**报告 §2 的 96 格表是有据的。**

### 3.2 但 test 覆盖与冻结清单不一致（需裁决）

- `FREEZE_ABLATION_SEED101.json` 声明：`runs` 共 **32 条，全部 seed=101**（8 配置 × 4 子集）。
- 实际磁盘上有 `test_pred.csv` 的 A8 格 = **36 个**：32 个 s101 **+ 4 个超出声明**：
  - `A8_base_L70 / FD001 / s202`、`A8_base_L70 / FD001 / s303`
  - `A8_mssa_L70 / FD001 / s202`、`A8_mssa_L70 / FD001 / s303`
- 这 4 格的 test 值也进了 `a8_engine_level.csv`（`test` 列非空 36 个，报告写「8 × 4 = 32 个测试评估」）。
- `FREEZE_FACTORIAL_S23.json` 覆盖的是 `F_single_proj_mean_hi0` / `F_single_proj_basic_hi1` / `F_full_basic_hi1`
  三个 tag，**不覆盖 `A8_*`**。所以这 4 格目前**没有任何冻结声明**。

**建议→已裁定（2026-11-12，主控）**：这 4 格**标 `n/a`、不进表**，**不补冻结声明**。
理由：冻结清单的意义是「打开测试集之前声明评估哪些格」，事后补声明等于改写评估顺序，比直接剔除更坏。
原始 test 值保留在产物和 `a8_engine_level.csv` 里供追溯，但**任何均值、排名、配对检验都不许用**。
A8 汇总只用 **32 格**；全库双口径汇总用 **109 格**（113 − 4）。
台账已同步：`runs.csv` 新增这 4 行，`audit_verdict=n/a`、`in_gate_a_table=false`、
`registry_note` 标 `OUTSIDE-FREEZE`。

### 3.3 `A8_full_L70` 与 `F_full_basic_hi1` 是同一批 run

12 格逐位一致（`val_pool` 全等，误差 0）。**但 `A8_full_L70` 的目录独立存在，config 也在。**
读表时若按 tag 去重，会把同一批 run 算两次；按「数值去重」又可能误删真重复。**台账用 `(variant, subset, seed)` 做键时，这两个 tag 必须映射到同一格。**

**已采纳（2026-11-12，主控）**：映射 `A8_full_L70 → F_full_basic_hi1`。台账已落：
8 个 `F_full_basic_hi1`（seed 202/303）行的 `registry_note` 写入别名说明，
登记 `A8_full_L70` 时必须复用这 8 行，不得新增同键行。

---

## 4. HC 协议布局（a40）

- `experiments/hc/` 下 **67 个目录**，命名 `HC-<stage>-<variant>-<subset>-s<seed>[-W1[rN]|-<UTC>]`。
- 完整（config + manifest + metrics + preds + ckpt + train.log 齐全）：**44 个**。
- **`test_pred.csv` 数量 = 0** —— HC 管线只做验证集，测试集要等独立冻结，**符合「不通过官方 TEST 选模型」的纪律**。
- 未完整的 23 个：`config.json` 有 66 个，但 manifest/metrics 只有 44 个 → 有 22 个是启动后失败/中断的 run
  （与 `runs.csv` 里 `status=failed/smoke/unclosed` 的行对应）。
- **这 44 个 HC run 不参与本轮的 capped/raw 重算**，因为它们没有测试预测。

---

## 5. 三套 Base 的字段级 diff（要求 6 核心）

### 5.1 逐字段对照

取自各族 `FD001/s101/config.json`（HC 管线无此三套，故不列）：

| 字段 | `A8_base_L70`（A8 全关） | `F_single_proj_mean_hi0`（Factorial base） | `VAN_raw_L70`（历史 vanilla） | `VAN_dsp_cycle_L70`（历史 vanilla+cycle） |
|---|---|---|---|---|
| `model` | dspt | dspt | **vanilla** | **vanilla** |
| `ms_mode` | **off** | **single_proj** | full（**惰性**） | full（**惰性**） |
| `sa_mode` | **off** | **mean** | off | off |
| `hi_source` | **none** | **hf** | none | none |
| `hi_mono` | none | correct | none | none |
| `hi_lambda` | 0.5 | **0.0** | 0.5 | 0.5 |
| `use_cycle` | True | True | **False** | **True** |
| `input_mode` | raw | raw | raw | **dsp** |
| 实测参数量 | **77058** | **81218** | **72257** | **74625** |

**完全相同**（27 个字段，可直接比）：`subset, seed, span=70, epochs=40, patience=12, batch=512, lr=1e-3,
weight_decay=1e-4, d_model=64, layers=2, dropout=0.2, mode=dual, hi_mono_coef=0.1, use_pad_mask, aux_dual,
instnorm, rul_cap=125, split_seed=7, fold=0, val_frac=0.2, n_cond=6, cuts=[0.4,0.55,0.7,0.85,0.95],
legacy_rul, global_hi, legacy_ms_pad, legacy_ema, data_dir`

→ **训练预算与数据划分完全同协议**，三套 Base 的差异只在模型结构与输入表示。

### 5.2 「同名不同义」清单（5 处）

| # | 同名 | 实际两种含义 | 后果 |
|---|---|---|---|
| 1 | **`base`** | A8：`ms=off, sa=off, hi=none`（三开关全关，77058 参数）<br>Factorial：`ms=single_proj, sa=mean, hi_lambda=0`（保留单尺度卷积 + 均值池化，只关 HI 监督，81218 参数） | **两个不同模型**。FD001 三 seed 同口径：engine-level 11.0851 vs 11.4316（差 **0.35**）；pooled 11.8739 vs 12.0932（差 **0.22**）。跨表比较必然对不上 |
| 2 | **「HI 关」** | A8：`hi_source=none`（分支不存在，77058）<br>Factorial：`hi_source=hf` + `hi_lambda=0.0`（分支存在、损失权重为 0，81218） | 「关 HI 监督」≠「无 HI 分支」：多出的 4160 个参数就是 HI 分支本身。按名字当成同一件事会错 |
| 3 | **`ms_mode=full`** | 在 `VAN_*` 里是**惰性字段**：`model=vanilla` 根本不读 `ms_mode` | 按字段读会误判成「多尺度」，实际是 vanilla |
| 4 | **`VAN_*` 两个 base** | `VAN_raw`：`use_cycle=False, input_mode=raw`<br>`VAN_dsp_cycle`：`use_cycle=True, input_mode=dsp` | 同 model=vanilla，输入表示完全不同，性能差 0.9–2.1 RMSE |
| 5 | **`full`** | `A8_full_L70` 与 `F_full_basic_hi1` 数值逐位相同（同一批 run，两个 tag）；另有 `F_full_basic_hi0`（hi 关）与 `F_full_mean_hi1` | 按 tag 计数会重复；按名字猜配置会错 |

### 5.3 一句话判据

> 引用任何「base」数字前，必须同时说明 **三开关取值** 和 **tag 前缀**（`A8_` / `F_` / `VAN_`）。
> 只写「base」的句子在本项目里无法定位到 run。

---

## 6. 对要求 1（双口径重算）的可用性判定

| 类别 | run 数 | 能否 capped/raw 重算 | 说明 |
|---|---:|---|---|
| 有 `test_pred.csv`（lab3090） | 113 | ✅ 可以 | 含 A8 36 格、F_* 三种子 36 格、单种子 F_* 24 格、C_* 8 格、ATTR 7 格、VAN 6 格 |
| 只有 `metrics.json` | 152 | ❌ 不可以 | 无逐样本预测；`test_*` 也不存在（这些 run 根本没测） |
| HC 协议 run（a40） | 44 完整 | ❌ 不可以 | 只做验证集，`test_pred.csv` = 0 |
| 本地镜像独有（a40 无） | 25 | ✅ 可以（本地可算） | 但与 lab3090 副本需核对是否同一份 |

**特别注意**：113 这个数字里含 **4 个越界的 A8 格**（§3.2）。若主控裁定这 4 格无效，
可用于双口径重算的 run 数应记为 **109**。

---

## 7. 待裁决 / 待补

1. ~~**A8 4 格 test 越界**~~ → **已裁定**：标 `n/a`、不进表、不补声明；A8 汇总用 32 格，双口径汇总用 109 格。
   台账 4 行已登记（`audit_verdict=n/a`）。
2. **本地镜像只有 31%**：是否把 lab3090 的 265 个 run 全量 rsync 一份到 a40 或本地，避免单点？
3. **25 个 run 只有本地有全量副本**（a40 无）：如果本地机器故障，这 25 个的逐样本预测就没了。
4. **`FREEZE_ABLATION_SEED101.json` 未进本地/仓库**：冻结顺序无法在本地复核，建议纳入版本控制。
5. **`A8_full_L70` ≡ `F_full_basic_hi1`**：台账去重键必须显式映射这两个 tag，否则重复计数。

---

## 8. 复算方法（任何人可重跑）

```bash
# 权威全量（lab3090）
ssh lab3090
cd /home/user/rulb/paper_b/results/raw
find . -maxdepth 4 -name metrics.json | wc -l      # 265
find . -maxdepth 4 -name test_pred.csv | wc -l     # 113

# 发动机级复算（逐 run）
python3 - <<'PY'
import csv, math, os
d = "A8_ms_L70/FD003/s101"
rows = list(csv.DictReader(open(os.path.join(d, "val_pred.csv"))))
per = {}
for r in rows:
    per.setdefault(r["unit"], []).append(float(r["y_pred"]) - float(r["y_true"]))
eng = [math.sqrt(sum(x*x for x in v) / len(v)) for v in per.values()]
print("engine-mean", sum(eng) / len(eng))
PY
```

- 验证集每台引擎恰好 5 个窗口，故 `pooled² = mean_e MSE_e`；`engine-mean = mean_e RMSE_e`，
  两者只差聚合顺序（Jensen），**不是同一个数**。
- 测试集每台引擎 1 个窗口，pooled 与 engine-mean 相同。
