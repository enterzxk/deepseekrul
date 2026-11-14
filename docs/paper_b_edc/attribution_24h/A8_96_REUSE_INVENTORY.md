# A8 96 格可复用清单 & `runs.csv` 128 行差异说明（2026-11-14）

用途：任务 B-3/B-4 的复用底账 + 台账口径澄清。只读核对，未重算任何已发布数字（主效应为独立复算，见 §3）。

---

## 0. 一句话

**96 格 = `ablation8.json` 的 8 臂 × 4 子集 × 3 seed，全部 `use_cycle=True`，val 96 格齐、test 盘上 36 格（32 冻结 + 4 额外）。**
`runs.csv` 的 128 行**不是它的台账**——那 128 行里只有 4 格来自 A8，且是冻结清单外的 4 格。

---

## 1. 覆盖面（实测）

| 项 | 实测 | 出处 |
|---|---|---|
| 配置臂 | 8（`base / ms / sa / hi / ms+sa / ms+hi / sa+hi / full`） | `paper_b/experiments/grids/ablation8.json` |
| 训练格 | **96** = 8 × 4 子集 × 3 seed（101/202/303） | `paper_b/results/a8_engine_level.csv` 96 行 |
| `use_cycle` | **96/96 为 True**（本网格没有 cycle-off 臂） | `ablation8.json` |
| val | **96/96**（engine-level 可从 `val_pred.csv` 重算；`val_pool` 为 pooled） | 同上 |
| test | **36**（冻结清单 32 = seed 101 × 8 臂 × 4 子集；另有 4 格在清单外） | `RUN_INVENTORY.md` §3 |
| 冻结外 4 格 | `A8_base_L70/FD001/s202,s303` + `A8_mssa_L70/FD001/s202,s303` → 标 `n/a`，不进表 | `A8_ABLATION_96CELL.md` §0/§3 |

**测试覆盖是 36 不是 32**，但**能进表的是 32**。两个数都要写，否则会有人拿 36 去对 32 的账。

---

## 2. 32 个格子的数（val engine-level，mean ± sd over 3 seeds）

| 臂 | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| base（全关） | 11.0851 ± 0.4704 | 10.9744 ± 0.0685 | 11.3813 ± 0.5976 | 11.8064 ± 0.4478 |
| +ms | 10.5454 ± 0.3141 | 11.0964 ± 0.2487 | 11.4933 ± 0.1742 | 12.2264 ± 0.4733 |
| +sa | 10.4308 ± 0.5793 | 10.7821 ± 0.3347 | 11.1164 ± 0.5823 | 12.0845 ± 0.3999 |
| +hi | 11.1640 ± 0.4415 | 10.6424 ± 0.3520 | 11.2793 ± 0.1526 | 11.5801 ± 0.1670 |
| ms+sa | 10.2402 ± 0.5537 | 10.6746 ± 0.4168 | 10.9828 ± 0.3878 | 12.1350 ± 0.8505 |
| ms+hi | 10.4951 ± 0.1362 | 11.0340 ± 0.1816 | 11.5764 ± 0.3294 | 11.5792 ± 0.2846 |
| sa+hi | 10.4306 ± 0.7384 | 10.8935 ± 0.2734 | 11.1875 ± 0.7181 | 12.4751 ± 0.1740 |
| full（= V0C） | 10.3588 ± 0.3166 | 10.7614 ± 0.2665 | 10.9873 ± 0.8902 | 11.8182 ± 0.3440 |

（与 `A8_ABLATION_96CELL.md` §2 逐格一致；`full` 行 ≡ `F_full_basic_hi1`，12 格逐位相同，已由 `tmp/audit_hc/identity_check.py` 验证。）

**test（seed 101，仅描述，不作选择依据）**

| 臂 | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| base | 11.5467 | 10.9754 | 12.2663 | 12.4975 |
| +ms | 10.6641 | 11.5376 | 11.6011 | 11.7438 |
| +sa | 10.5503 | 10.9092 | 12.8285 | 12.2271 |
| +hi | 11.4514 | 11.3808 | 12.1512 | 12.4621 |
| ms+sa | 10.8223 | 10.8482 | 10.9459 | 12.6534 |
| ms+hi | 10.9295 | 11.6957 | 11.0877 | 12.5517 |
| sa+hi | 10.7885 | 10.9584 | 11.2979 | 12.7747 |
| full | 10.2881 | 10.5227 | 11.8032 | 12.6285 |

---

## 3. 能从 96 格直接算出的量（已独立复算，与报告 §4 逐位吻合）

主效应 = 「开关为 on 的 4 臂均值」−「开关为 off 的 4 臂均值」，每个子集 24 格对 24 格：

| 开关 | FD001 | FD002 | FD003 | FD004 | 全部 |
|---|---:|---:|---:|---:|---:|
| ms | −0.3678 | +0.0685 | +0.0188 | −0.0468 | **−0.0818** |
| sa | −0.4573 | −0.1589 | −0.3641 | +0.3302 | **−0.1625** |
| hi | +0.0367 | −0.0490 | +0.0141 | −0.1999 | **−0.0495** |

**能支撑的结论（本网格内）**：`sa` 在 FD001–FD003 一致为负、在 FD004 反号；`hi` 在 FD004 为负；
`ms` 主效应接近 0 但**在 FD004 与 `hi` 有交互**（`+ms` 12.2264 vs `base` 11.8064 更差，
`ms+hi` 11.5792 却是 FD004 最好的一格）——**"ms 有帮助/有害"这两种说法都不成立，成立的是"ms 需要 hi"**。

**不能支撑的**：

1. **cycle 的贡献**——96 格里 `use_cycle` 全 True，没有 off 臂。base 骨干的 cycle-off 在 `NC_base_L70`（12 格），强骨干要等 V0N 12 格。
2. **跨骨干的普适性**——这 8 臂的差异全在 `full` 骨干（`ms` 的开关是 `off↔full`，不是 `full↔single_proj`）。`Selected`（`single_proj`）不在网格里。
3. **预算效应**——96 格全是 `ep40/pat12/span70`，没有 80ep 臂。
4. **test 侧的种子离散度**——test 只有 seed 101。

---

## 4. `runs.csv` 128 行是什么（差异说明）

**构成**（`docs/paper_b_hc/registry/runs.csv`，2026-11-14 实测）：

| 组成 | 行数 | host | 说明 |
|---|---:|---|---|
| HC stage-C（A0/A1/A2/A3a/A3p/A4/A5/A2cal/A3bcal/SMOKE） | 100 | a40 `YNDXTEA0034` | 含 **24 行 failed**、**6 行 smoke**、1 行 superseded |
| `F_*` legacy 回填（3 族） | 24 | a40 | **只登 s202/s303**，每族 8 行 |
| A8 冻结清单外 4 格 | 4 | lab3090 | `A8_base_L70/FD001/s202,s303`、`A8_mssa_L70/FD001/s202,s303` |
| **合计** | **128** | | 其中 `status=ok` 仅 **97** 行 |

**所以 128 行是"发射/回填台账"，不是"结果清单"**：它含失败与 smoke，且绝大多数在 a40。
按 128 行去数"跑过多少格"会同时**高估**（失败/ smoke 也算行）和**低估**（96 格只登了 4 格）。

**具体缺口**：

| 缺口 | 规模 | 后果 |
|---|---|---|
| A8 96 格只登 4 格 | 92 格不在 runs.csv | 从台账看会以为 A8 只跑了 4 格 |
| `F_full_basic_hi1` 缺 s101 | 4 格 | 台账 8 行 → 会得出"8 run"而不是 12 |
| `F_single_proj_basic_hi1` 缺 s101 | 4 格 | 同上 |
| `F_single_proj_mean_hi0` 缺 s101 | 4 格 | 同上 |
| `NC_base_L70` 完全未登 | 12 格 | cycle-off 在台账里"不存在" |
| `VAN_*` / `MSM_*` / `SAM_*` / `HIM_*` / `ATTR_*` / `C_*` / `REPRO_*` 未登 | 见 `RUN_INVENTORY.md` §2 | 台账不含这些族 |

**权威口径**：`RUN_INVENTORY.md`（lab3090 全量 **265 run / 113 有 test**）> `a8_engine_level.csv`（96 格）> `seed_robustness_raw.csv`（36 格：3 族 × 12）> `runs.csv`（128 行发射台账）。
**引用任何"跑了多少格"必须写明来源文件**，四个数互不相等，且都不该被当成另一个。

---

## 5. 口径规则（复用前必读）

1. **val 一律从 `val_pred.csv` 重算 engine-level**。legacy run 的 `metrics.json["val_engine_rmse"]` 存的是 **pooled**（实测 `A8_ms_L70/FD001/s101`：字段 11.0889，engine-level 10.3321）。
2. **val 两个口径不能互换**：`F_full_basic_hi1` val engine-mean 10.9814 / val pooled 12.9821（差 2.0）。
3. **test 只有 pooled**（`test_rmse` = `test_pooled_rmse`），且只有 seed 101；不要和 3-seed val 并排比较精度。
4. **`full` 臂 = `F_full_basic_hi1` = V0C**，同一批 12 个 run、三个名字；引用时写清用哪个名字，避免被当成三次独立训练。
5. **A8 的选轮用 pooled、评估用 engine-level**（报告 §6 已记录）：这是已记录的缺陷，不是可修正项——本轮不重跑、不修。

---

## 6. 复用对照表（任务 B 逐项）

| 任务 B 项 | 现成数据 | 够不够 | 要补什么 |
|---|---|---|---|
| B-1 cycle 独立贡献 | base 骨干 `NC_base_L70` 12 格（val） | **半够** | 强骨干 V0N 12 格（已在队列，见 `RUN_QUEUE_V0N.md`） |
| B-2 原始传感器/工况偏差/EMA | `ATTR_*` 7 格（全 s101）+ `VAN_raw_L70` 12 格 | **不够**（n=1） | 若要把结论写成"增量"，需补 seed 202/303 |
| B-3 MS/SA/HI 主效应+交互 | A8 96 格 | **够** | 无 |
| B-4 精简 vs 完整 | `F_full_basic_hi1` vs `F_single_proj_basic_hi1`（各 12，test 全） | **够** | 无 |
| B-5 Age-only / +Cond / +Sensor | `A0` 4 格（cycle-only）+ `VAN_dsp_cycle_L70` 12 格 | **半够** | Age+Cond 无现成臂 |

---

## 7. 复算命令

```bash
# A8 主效应与逐格均值（本机）
python tmp/audit_hc/a8_reuse_check.py     # 输出 §2 表与 §3 主效应，可对 A8_ABLATION_96CELL.md 逐位核
# runs.csv 构成
python tmp/audit_hc/reg_diff.py           # 输出 §4 的组成表
```

（脚本在 `tmp/audit_hc/`，按仓库惯例 `tmp/` 不入库；输出见 §8。）

---

## 8. 实测输出

```
A8 rows=96  arms=8  subsets=4  seeds=3
test filled=36  (frozen 32 + extra 4)
main effect ms: FD001 -0.3678 FD002 +0.0685 FD003 +0.0188 FD004 -0.0468  ALL -0.0818
main effect sa: FD001 -0.4573 FD002 -0.1589 FD003 -0.3641 FD004 +0.3302  ALL -0.1625
main effect hi: FD001 +0.0367 FD002 -0.0490 FD003 +0.0141 FD004 -0.1999  ALL -0.0495

runs.csv rows=128
  status: {'failed': 24, 'smoke': 6, 'ok': 97, 'superseded': 1}
  host:   {'YNDXTEA0034': 100, 'a40': 24, 'lab3090': 4}      ← a40 有两种写法，见 §4 注
  stage:  {'C': 100, 'B': 28}
  F_full_basic_hi1: rows=8 seeds=['202','303'] val_metric_kind=['pooled']
  A8 rows in runs.csv=4      distinct variants=15

RUN_QUEUE.csv rows=12 unique_run_ids=12 dupes=0
  arms=['V0N'] tags(parent)=['F_full_basic_hi1']
  subsets=['FD001','FD002','FD003','FD004'] seeds=['101','202','303']
  use_cycle=['False'] expected_params=['115075'] status=['queued']
```

**§4 注（host 列的两种 a40 写法）**：`host` 列里 `YNDXTEA0034` 100 行 + `a40` 24 行，**是同一台 a40 的两个标签**
（前者是 HC stage-C 的机器名，后者是 legacy 回填时写的主机别名）。按 host 统计时必须先合并，
否则会得出"a40 只有 24 行"的错数。**这台机器本轮只作历史文件来源，不再起训练。**
