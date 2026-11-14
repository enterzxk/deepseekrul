# Paper B 身份核对与可复用结果清单（2026-11-14）

用途：本轮任务 A（重建真实强基线）与任务 B（贡献来源归因）的前置事实核查。
作者：实验结果管理员。只读核对，未改动任何实验产物、未重算任何已发布数字。

结论先行：**三个身份都能钉死，但其中两个的"已有数据"与任务书的假设不符**（`A0` 不是
TrainDefaults；`V0N` 确实零运行，但 cycle-off 在**另一条骨干**上存在 12 格）。

---

## 0. 五条结论

1. `DSPT-V0-Cycle` ≡ `F_full_basic_hi1` ≡ `A8_full_L70`：**同一批 12 个 run，三个名字**，
   逐格逐位一致（12/12）。val engine-mean **10.9814**，test pooled **11.5116**（12 格全有 test）。
2. `DSPT-TrainDefaults` **不是 `A0`**，而是 `A8_ms_L70`：12 格 val 全有，**val engine-mean 11.3404**；
   test 只有 seed 101 四格（pooled **11.3867**）。参数 82114 与 `DSPT(max_span=70)` 默认构造**逐一相等**。
3. `A0` 是 **cycle-only**（`cycle_only=True`，无传感器）：4 格、仅 s101、val **26.29–39.98**、无 test。
   把它当 TrainDefaults 会把基线写成比实际差 3 倍。
4. `V0N`（强骨干 + cycle off）**零运行**：全仓库无 `V0*` 目录、无 `use_cycle=false` 且
   `ms=full/sa=basic/hi=hf` 的 config。**但 cycle-off 不是完全没跑过**——`NC_base_L70` 有 12 格
   （base 骨干、无 test），cycle 在 base 骨干上的独立贡献 = **+1.0405**（engine-level）。
5. `docs/paper_b_hc/registry/runs.csv`（128 行）**不是全量台账**：`F_*` 只登记 s202/s303（每族 8 行），
   s101 不在其中；A8 96 格只登记 4 个冻结清单外的格。**引用"12 run"必须引 `paper_b/results/`，
   不能引 runs.csv。**

---

## 1. 三个身份的真实运行记录

| 名字 | 真实 tag | 配置（config.json 实测） | run | val 覆盖 | test 覆盖 | val engine-mean | test pooled mean | params@span70 |
|---|---|---|---:|---|---|---:|---:|---:|
| `DSPT-V0-Cycle` | `F_full_basic_hi1` = `A8_full_L70` | dspt, ms=full, sa=basic, hi=hf, hi_mono=correct, hi_lambda=0.5, cycle=on, input=raw, ep40/pat12/span70/cap125 | 12 | 12（4 子集 × 3 seed） | **12** | **10.9814** | **11.5116** | 115651 |
| `DSPT-TrainDefaults` | `A8_ms_L70` | dspt, ms=full, sa=**off**, hi=**none**, hi_mono=none, cycle=on, input=raw, 同协议 | 12 | 12 | **4**（仅 s101） | **11.3404** | 11.3867（n=4） | 82114 |
| 历史 `Selected` | `F_single_proj_basic_hi1` | dspt, ms=**single_proj**, sa=basic, hi=hf, hi_mono=correct, cycle=on, 同协议 | 12 | 12 | **12** | **10.8960** | **11.1338** | 114755 |
| （对照）`V0H` | `A3p` | full/basic/none + cycle 冻结注入 | 4 ok（s101） | 4 | 0 | FD001 10.8951 / FD002 10.2371 / FD003 11.0023 / FD004 11.366（pooled 字段，见 §5） | — | — |
| （对照）`A0` | `A0` | cycle-only，`cycle_only=True`，无传感器 | 4（s101） | 4 | 0 | 30.0986 / 26.2874 / 39.9834 / 37.1218 | — | — |

补充：`V0H ≡ A3p` 在代码里是**显式声明**的（`config_hc.py:49-58`：`V0H` 注释 `== A3p`），
但 A3p 只有 4 个 ok 格、全在 a40、全 s101、无 test——**n=1/子集，不能给种子离散度**。

---

## 2. 身份是怎么钉死的（三条独立证据，不靠名字）

1. **配置逐字段**：`paper_b/results/raw/F_full_basic_hi1/FD001/s101/config.json` 的
   `ms_mode/sa_mode/hi_source/hi_mono/hi_lambda/use_cycle/input_mode/epochs/patience/span/split_seed/val_frac/rul_cap`
   与 `config_hc.py` 的 `V0C` 三元组（`BACKBONE` full/basic/hf + cycle 冻结）逐项相同。
2. **参数逐位复算**（在 lab3090 用 `~/miniconda3/envs/ncmapss/bin/python` 实测）：

   ```
   DSPT(max_span=70)                                  -> 82114   = A8_ms_L70  ✓
   DSPT(max_span=70, ms=off, sa=off)                  -> 77058   = A8_base_L70 ✓
   DSPT(max_span=70, full, basic, hf, correct)        -> 115651  = A8_full_L70 / F_full_basic_hi1 ✓
   DSPT(max_span=70, single_proj, basic, hf, correct) -> 114755  = F_single_proj_basic_hi1 ✓
   ```

   （`DSPT()` 默认 `max_span=100` 时是 84034；差 `30×64=1920` 正是 `temporal_pe` 少 30 行。
   **参数差不是代码漂移**，是 span 不同。这条差点被我写成"代码漂移"，写下来防止下次误判。）
3. **逐格逐位**：`a8_engine_level.csv` 的 `A8_full_L70` 12 格与 `seed_robustness_raw.csv` 的
   `F_full_basic_hi1` 12 格，`val_eng`/`test` 全部相等（`compared=12 mismatched=0`）。
   即：**A8 的 full 臂就是 V0C，不是两次训练。**

---

## 3. 与任务书假设不符的三处

### 3.1 `A0` 不是 TrainDefaults

`config_hc.py:75-76`：`'A0': dict(label='cycle-only baseline', cycle_mode='none',
cycle_source='raw', cycle_only=True)`。即**只喂年龄、不喂传感器**。
它的 val 26–40 正是"仅用年龄预测 RUL"的量级，与 `GATE_A.cycle_only_min_gap_pct=10.0`
的设计一致（用来证明"基准不能靠年龄解决"）。

真正的 TrainDefaults（`DSPT()` 默认 `full/off/none/cycle=on`）**早就跑过**：`A8_ms_L70`，
12 格 val 齐、test 4 格。**任务 A 不需要为 TrainDefaults 新训练。**

### 3.2 `V0N` 零运行——但 cycle-off 在 base 骨干上有 12 格

实测：`paper_b/results/raw/` 下无 `V0*` 目录；全仓库 `config.json` 里 `"use_cycle": false`
只有 3 个（`VAN_raw_L70/FD001/s101,202,303`，model=vanilla），加上 `nocycle.json` 网格的
`NC_base_L70` 12 格（`ms=off, sa=off, hi=none, use_cycle=false`）。

所以：

- **强骨干 cycle-off（V0N）确实不存在**，任务 B 第 1 项要在强骨干上做，必须新训练 12 格。
- **但"cycle 有没有用"不是零证据**：`A8_base_L70`（cycle on）vs `NC_base_L70`（cycle off），
  同为 base 骨干、同 split、同 seed 集，engine-level 差：

  | 子集 | cycle on | cycle off | Δ（off − on） |
  |---|---:|---:|---:|
  | FD001 | 11.0851 | 11.6948 | +0.6097 |
  | FD002 | 10.9744 | 11.4344 | +0.4600 |
  | FD003 | 11.3813 | 12.6599 | +1.2786 |
  | FD004 | 11.8064 | 13.6199 | +1.8135 |
  | **全部** | **11.3118** | **12.3523** | **+1.0405** |

  结论写法必须是：**"在 base 骨干、本协议下，cycle 值约 1.04 RMSE"**，不能写成"cycle 值 1.04"。
  强骨干上的边际值未知，且方向可能不同（传感器已经很强时年龄的边际信息更少——这是假设，不是结论）。
  `NC_base_L70` **无 test**，所以这一项只能在 val 上说。

### 3.3 `runs.csv` 不是全量

128 行里：`F_*` 每族 8 行（s202/s303），**s101 全缺**；A8 只登记 4 行
（`A8_base_L70/FD001/s202,s303` + `A8_mssa_L70/FD001/s202,s303`，即冻结清单外的 4 格）。
`A0`/`A3p` 全在 a40（`host=YNDXTEA0034`）。
**引用 run 数时以 `paper_b/results/` + `RUN_INVENTORY.md` 为准**（lab3090 全量 265 run / 113 有 test）。

---

## 4. 可复用结果清单（任务 A/B 直接可用 vs 必须新训练）

| 家族 | 配置 | val n | test n | 口径 | 可直接支撑 | 不能用于 |
|---|---|---:|---:|---|---|---|
| `F_full_basic_hi1` / `A8_full_L70` | V0C（强基线） | 12 | 12 | val engine-mean / test pooled | 任务 A 强基线；任务 B-3（MS/SA/HI 交互，full 臂）；任务 B-4（精简 vs 完整） | 无（本轮最完整的一族） |
| `F_single_proj_basic_hi1` | Selected | 12 | 12 | 同上 | 任务 B-4（精简 vs 完整） | 当"已发表基线"引用前须确认 factorial 选择偏置 |
| `A8_ms_L70` | TrainDefaults | 12 | 4（s101） | val engine-mean / test pooled | 任务 A 的 TrainDefaults 身份 | 跨 seed 的 test 比较（只有 s101） |
| A8 八臂（base/ms/sa/hi/ms+sa/ms+hi/sa+hi/full） | 2×2×2 | 96 | 36（32 冻结 + 4 额外） | val engine-mean / test pooled | 任务 B-3 主效应+交互（**同骨干、同协议、cycle 全开**） | 任务 B-1（无 cycle-off 臂）；把 test 用于选模型 |
| `NC_base_L70` | base 骨干 + cycle off | 12 | 0 | val engine-mean | 任务 B-1 的 base 骨干一半 | test 层任何说法 |
| `A3p`（=V0H） | 强骨干 + cycle on + 无 HI | 4（s101） | 0 | pooled 字段（见 §5） | 任务 B-3 的 HI 主效应（强骨干、n=1） | 种子离散度、test |
| `A0` | cycle-only | 4（s101） | 0 | pooled | 证明"年龄不足以解题"（Gate A ii） | 当基线 |
| `VAN_raw_L70` / `VAN_dsp_cycle_L70` | vanilla 骨干 ± cycle | 12 | 3（FD001） | val engine-mean | 任务 B-5（Age-only 能解释多少） | 四子集 test |
| `F_*_hi0` / `F_*_mean_hi1` 等单种子格 | 24 格 | 24 | 24 | pooled/engine 混 | 只作方向性参考 | 任何 n≥3 的说法 |

**任务 B 逐项的可复用判定**（给经理）：

| 项 | 现状 | 判定 |
|---|---|---|
| B-1 Cycle 独立贡献 | base 骨干有（+1.0405，val）；强骨干无 | **部分可复用**；强骨干需 12 格 V0N |
| B-2 原始传感器 / 工况偏差 / EMA 增量 | `ATTR_*` 7 格（s101，ATTR_legacy_ema 等）+ `VAN_raw` | **部分可复用**（n=1，需补 seed 才能下结论） |
| B-3 MS/SA/HI 主效应+交互 | A8 96 格，同骨干同协议 | **完全可复用**，不需新训练 |
| B-4 精简 vs 完整 | `F_full_basic_hi1` vs `F_single_proj_basic_hi1`（各 12，全 test） | **完全可复用** |
| B-5 Age-only / Age+Cond / Age+Sensor | `A0`（cycle-only，4 格）+ `VAN_dsp_cycle`（12 格，FD001 test 3） | **部分可复用**；Age+Cond 无现成臂 |

---

## 5. 一个必须写下来的口径陷阱

**legacy run 的 `metrics.json["val_engine_rmse"]` 不是 engine-level，而是 pooled。**
实测（FD001 s101）：

```
F_full_basic_hi1   metrics.rmse = 11.4584 = metrics.val_engine_rmse   ← 都是 pooled
                   从 val_pred.csv 重算：engine-mean 10.7175 / pooled 11.4584
A8_ms_L70          metrics.val_engine_rmse = 11.0889 (pooled) ；engine-mean 10.3321
NC_base_L70        metrics.val_engine_rmse = 13.6797 (pooled) ；engine-mean 12.7834
```

**我自己在本次核对里先踩了一次**：用 `metrics.val_engine_rmse` 读 `NC_base_L70`，得出
"cycle 值 2.90"，换成从 `val_pred.csv` 重算后是 **1.0405**。差 1.86，全部来自口径。
`a8_engine_level.csv` 的 `val_eng` 列是对的（我按 6 格抽验，全部逐位吻合），
**结论：engine-level 一律从 `val_pred.csv` 重算，不读 metrics.json 字段名。**

同一批 run 的两个口径（V0C）：val engine-mean **10.9814** / val pooled **12.9821**；
test 两个口径相等（**11.5116**，test 侧 `test_rmse` 就是 pooled）。
**这两个 val 数相差 2.0，绝不能互相替代**；`runs.csv` 里 `F_*` 的 `val_rmse` 正是 pooled 那一支
（`val_metric_kind=pooled, legacy_metric=true`），拿它和 A8 的 engine-level 比会得到"几乎相等"的假象。

---

## 6. 复算命令与输出（可重跑）

```bash
# 1) 身份逐格比对（本地 Windows）
python tmp/audit_hc/identity_check.py    # compared=12 mismatched=0
#    脚本在 tmp/audit_hc/（按仓库惯例 tmp/ 不入库），输出见 §6.1
# 2) 参数复算（lab3090）
ssh lab3090 'cd ~/deepseek-rul && ~/miniconda3/envs/ncmapss/bin/python /tmp/dspt_params2.py'
#    span 70 defaults 82114 / span 100 84034 / base 77058 / full 115651 / single_proj 114755
# 3) engine-level 重算（lab3090，对每个 run 的 val_pred.csv）
ssh lab3090 'python3 /tmp/nc_eng.py'      # NC_base: FD001 11.6948 / FD002 11.4344 / FD003 12.6599 / FD004 13.6199 / ALL 12.3523
# 4) 零运行证据
ssh lab3090 'ls -d ~/rulb/paper_b/results/raw/V0* 2>/dev/null || echo NONE'
grep -rl '"use_cycle": false' paper_b/results/raw --include=config.json
```

### 6.1 实测输出（2026-11-14 本机与 lab3090）

```
== 1. V0C identity: A8_full_L70 vs F_full_basic_hi1, cell by cell ==
  compared=12 mismatched=0
== 2. means (val engine-level / test pooled) ==
  F_full_basic_hi1:        n=12 val=10.9814 test=11.5116 (test_n=12)
  F_single_proj_basic_hi1: n=12 val=10.8960 test=11.1338 (test_n=12)
  A8_ms_L70:               n=12 val=11.3404 test_n=4 test=11.3867
  A8_base_L70:             n=12 val=11.3118 test_n=6 test=11.4836
== 3. runs.csv coverage ==
  F_full_basic_hi1:        rows=8 seeds=['202','303'] val_metric_kind=pooled
  F_single_proj_basic_hi1: rows=8 seeds=['202','303'] val_metric_kind=pooled
  A8 rows in runs.csv: 4      total rows: 128
== 4. A0 vs TrainDefaults (config fields) ==
  A0 rows=4 val=['30.0986','26.2874','39.9834','37.1218'] test=['','','','']
```

```
DSPT(max_span=70)                                  -> 82114    (A8_ms_L70)      ✓
DSPT(max_span=70, ms=off, sa=off)                  -> 77058    (A8_base_L70)    ✓
DSPT(max_span=70, full, basic, hf, correct)        -> 115651   (F_full_basic_hi1) ✓
DSPT(max_span=70, single_proj, basic, hf, correct) -> 114755   (F_single_proj_basic_hi1) ✓
```

```
NC_base_L70  engine-level（从 val_pred.csv 重算）
  FD001 11.6948 / FD002 11.4344 / FD003 12.6599 / FD004 13.6199 / ALL 12.3523
A8_base_L70  engine-level
  FD001 11.0851 / FD002 10.9744 / FD003 11.3813 / FD004 11.8064 / ALL 11.3118
cycle 贡献（off − on，base 骨干）：+0.6097 / +0.4600 / +1.2786 / +1.8135 / ALL +1.0405
```

---

## 7. 对任务 A/B 的直接建议（只给事实约束，不替经理决策）

1. **任务 A 不需要新训练**：V0C / TrainDefaults / Selected 三个身份都有 12 格 val；
   V0C 与 Selected 还有 12 格 test。可直接冻结为强基线三元组。
2. **任务 B-3、B-4 不需要新训练**：96 格 + 两族 12 格已齐，且协议一致（ep40/pat12/span70/cap125/split7/val0.2）。
3. **唯一必须补的是 V0N 12 格**（强骨干 cycle-off）：否则"cycle 在强骨干上的贡献"只能从
   base 骨干的 +1.0405 外推，而外推正是本报告 §3.2 明确禁止的写法。
4. **`A3p`（V0H）若要用，先补 seed 202/303**（现 4 格全 s101），否则 HI 主效应在强骨干上仍是 n=1。
