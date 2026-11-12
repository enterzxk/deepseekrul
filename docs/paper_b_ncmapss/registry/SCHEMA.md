# N-CMAPSS DS02 实验台账 schema

状态：**schema 已定，0 run**（等数据接入后按此登记）
最后更新：2026-11-12 20:40 (+08)
维护人：实验结果管理员
适用：`docs/paper_b_ncmapss/registry/`（**与 C-MAPSS 的 `docs/paper_b_hc/registry/` 完全独立，互不引用、互不写入**）
基准提交：`ac94c29`，分支 `ncmapss-ds02`

> 本文件只定义**记什么、怎么记**。协议内容（字段白名单、可观测性边界、LOEO 划分）以
> [DATASET_PROTOCOL.md](../DATASET_PROTOCOL.md) 为准，阶梯定义以
> [BASELINE_RESULTS.md](../BASELINE_RESULTS.md) 为准，冻结程序以
> [NCMAPSS_FINAL_RESULTS.md](../NCMAPSS_FINAL_RESULTS.md) 为准。冲突时以那三份为准，本文件跟着改。

---

## 0. 一句话

**一个 run 一行，一行只能有一个来源。** 数字从 `experiments/ncmapss/<run_id>/` 的
`metrics.json` / `preds.csv` 复算，文档里的任何数字必须能定位到 `run_id`。

---

## 1. 格数：先算清楚再开跑

### 1.1 两种错误算法（都会算错）

| 算法 | 结果 | 为什么错 |
|---|---:|---|
| 6 台 × 6 折 × 3 seed × 4 rung | 432 | **重复计数**。LOEO 的「6 折」就是「6 台各留出一次」，乘起来等于把每台算了 6 遍 |
| 6 折 × 3 seed × 4 rung | 72 | **N0 没有 seed 轴**。age-only 是 `y = f(cycle)`，无参数、无初始化、无训练随机性，3 个 seed 得到同一组数 |

### 1.2 正确算法

**开发集（LOEO，6 折 = 留出 2 / 5 / 10 / 16 / 18 / 20 各一次）**

| rung | 折数 | seed 数 | 行数 | 说明 |
|---|---:|---:|---:|---|
| N0 age-only | 6 | **—** | **6** | 无参数，无 seed 轴；每折 1 行（预测被留出的那台） |
| N1 vanilla | 6 | 3 | **18** | 每折每 seed 训练 1 次 |
| N2 DSPT Base | 6 | 3 | **18** | 同上 |
| N3 frozen DSPT | 6 | 3 | **18** | MS/SA/HI 用 C-MAPSS 冻结值，**不许因 DS02 结果改** |
| **开发合计** | | | **60** | 其中**训练出的 checkpoint = 54 个** |

**测试集（官方单位 11 / 14 / 15，**冻结后**才评估）**

| rung | seed 数 | 单位数 | 行数 | 说明 |
|---|---:|---:|---:|---|
| N0 age-only | — | 3 | **3** | 无 checkpoint |
| N1 / N2 / N3 | 3 | 3 | **9 each = 27** | 每个 rung 在**全部 6 台开发机**上重训 3 个 seed，再用这 3 个 checkpoint 各评 3 台测试机 |
| **测试合计** | | | **30** | 冻结进入测试的 checkpoint = 9 个 |

**总计：60 + 30 = 90 行台账；训练 checkpoint 54 个；冻结进入测试 9 个；
`NCMAPSS_FINAL_RESULTS.md` 的表是 4 rung × 3 单位 = 12 个格（每格由 3 个 seed 聚合）。**

### 1.3 测试集口径：**已定，不许改**

**裁定（2026-11-12，经理）：全部 6 台开发机重训 3 seed → 9 个 checkpoint → 评 3 台测试机 → 27 行。**
依据：用户 §九 阶段 4 写死「使用全部 6 台开发发动机训练，冻结模型设计后，评估 3 台官方测试发动机」。

被否决的另一种做法（拿 54 个 LOEO checkpoint 直接评测试机，`54 × 3 = 162` 行）不是可执行口径：
「哪一折代表最终模型」没有定义，等于给「挑最好那折」留了后门。

**硬规则（写死，审稿人会问）**：

> **最终测试模型 ≠ 任何一折模型。**
> LOEO 的 54 个 checkpoint **只用于开发集报告**，**不参与任何测试集评价**，不进出测试的 checkpoint 清单。
> 冻结清单里的 `checkpoints[]` 只能列 9 个「全开发机重训」的 checkpoint；出现 LOEO checkpoint 即冻结无效。

### 1.4 测试集上能说和不能说的

测试机只有 **3 台**（11 / 14 / 15），每台 3 个 seed。因此：

| 能说 | 不能说 |
|---|---|
| 每台机各自的误差是多少（`rmse_raw_unit` 逐行） | 「跨机 95% CI」——n=3，没有这个精度 |
| 每台机的 seed 均值 ± seed sd（sd 只反映优化随机性） | 把 seed sd 当成总体不确定性 |
| 三台机之间的误差差异有多大 | 「DSPT 在测试集上显著优于基线」——3 台机双侧符号检验最小 p = 0.25 |

`NCMAPSS_FINAL_RESULTS.md` 的 12 格表（4 rung × 3 单位）**每格只能报 seed 均值 ± seed sd**，
表头必须写「seed mean ± seed sd (n=3 seeds, 1 unit)」，**不得写成跨机置信区间**。

### 1.4 有效样本数不是 18

**6 折 × 3 seed = 18 个数字，但不是 18 个独立样本。**
- 独立单位是**发动机**（开发集 6 台），seed 只衡量优化随机性。
- 6 折的训练集高度重叠（每折只差 1 台），不能当独立重复。
- 台账用 `independent_unit=engine` 记这一点；任何跨 seed 的区间估计必须用 `df = n_engines − 1`，
  不是 `n_seeds − 1`（C-MAPSS 那边踩过这个坑：`SEED_ROBUSTNESS.md` 用 df=3 使区间偏窄 26%）。

---

## 2. run_id 命名（强制）

```
NCM-<phase>-<rung>-<fold>-s<seed>-<YYYYMMDDTHHMMZ>
```

- `phase`：`dev` | `test`
- `rung`：`N0` | `N1` | `N2` | `N3`（N4/N5 等 C-MAPSS 机制冻结后再加，编号先占位）
- `fold`：`LOEO-2` | `LOEO-5` | `LOEO-10` | `LOEO-16` | `LOEO-18` | `LOEO-20`（留出的那台）；
  `phase=test` 时写 `ALLDEV`（在全部 6 台上重训）或 `LOEO-<n>`（若采用 §1.3 第二种做法）
- `seed`：`101` | `202` | `303`；**N0 省略整个 `-s<seed>` 段**
- 时间戳 UTC 到分钟，取**训练结束**（`metrics.json` 落盘）时刻

示例：
```
NCM-dev-N3-LOEO-10-s101-20261113T0215Z
NCM-dev-N0-LOEO-10-20261113T0220Z
NCM-test-N3-ALLDEV-s101-20261120T0900Z
```

**同一 run_id 不得重跑。** 重跑换时间戳，并在 `registry_note` 写清「重跑原因 + 前一次 run_id」。

---

## 3. `runs.csv` 列定义（50 列，固定）

```
run_id, phase, rung, model, fold, held_out_engine, n_train_engines, seed,
git_sha, code_md5, data_sha256, split_hash,
feature_set, feature_list, config_path, config_sha256,
selection_rule, epochs_run, best_epoch,
ckpt_path, ckpt_sha256, preds_path, pred_instant, n_units, n_flights,
rmse_raw_unit, mae_raw_unit, bias_raw_unit, nasa_raw_unit,
rmse_cap125_unit, mae_cap125_unit, bias_cap125_unit, nasa_cap125_unit,
rmse_raw_flightpooled, rmse_cap125_flightpooled,
seg_early_raw, seg_mid_raw, seg_late_raw,
seg_early_cap125, seg_mid_cap125, seg_late_cap125, seg_boundaries,
freeze_id, freeze_ts_utc, eval_start_utc,
status, audit_verdict, independent_unit, registry_note, audited_by
```

### 3.1 逐列含义

| # | 列 | 取值 / 说明 |
|---:|---|---|
| 1 | `run_id` | §2 格式，唯一 |
| 2 | `phase` | `dev` / `test` |
| 3 | `rung` | `N0` / `N1` / `N2` / `N3` |
| 4 | `model` | 实际模型名，与 config 一致（`age_only` / `vanilla` / `dspt_base` / `dspt_frozen`） |
| 5 | `fold` | `LOEO-<留出机号>` / `ALLDEV` |
| 6 | `held_out_engine` | 该行预测的是哪台机；`phase=test` 时填该测试机号 |
| 7 | `n_train_engines` | 该 run 训练用了多少台（dev 折 = 5，test 重训 = 6） |
| 8 | `seed` | `101/202/303`；**N0 留空** |
| 9 | `git_sha` | 训练时工作区 SHA；`WORKTREE` 必须同时有 `code_md5` |
| 10 | `code_md5` | 关键文件 md5 的 JSON：`model` / `train` / `data` / `metrics` / `protocol` |
| 11 | `data_sha256` | 所用 HDF5 的 sha256（`N-CMAPSS/DS02/*.h5`） |
| 12 | `split_hash` | LOEO 折定义（留出机列表 + 训练机列表）的 sha256 |
| 13 | `feature_set` | 白名单版本号，如 `WS14+W4+CYC`（见 §4） |
| 14 | `feature_list` | **实际进入模型的列名**，`;` 分隔，必须能在 §4 白名单里逐个对上 |
| 15 | `config_path` | `experiments/ncmapss/<run_id>/config.json` |
| 16 | `config_sha256` | 上述 config 的 sha256（冻结时用来证明「跑的就是冻结的那份」） |
| 17 | `selection_rule` | `inner_val`（训练折内选轮）/ `fixed_epochs:<n>` / `none`（N0） |
| 18 | `epochs_run` | 实际跑完的 epoch 数 |
| 19 | `best_epoch` | 选轮得到的 epoch；`selection_rule=none` 时留空 |
| 20 | `ckpt_path` | `experiments/ncmapss/<run_id>/ckpt/best.pt`；N0 留空 |
| 21 | `ckpt_sha256` | checkpoint 的 sha256；N0 留空。**冻结清单里逐条列出** |
| 22 | `preds_path` | 逐发动机预测文件 `preds.csv`（`engine_id, flight_cycle, y_true_raw, y_true_cap125, y_pred`） |
| 23 | `pred_instant` | 预测时刻，本协议固定 `flight_end` |
| 24 | `n_units` | 该行涉及的发动机数（dev 折 = 1，test = 1） |
| 25 | `n_flights` | 该行涉及的飞行段数（预测点个数） |
| 26–29 | `rmse/mae/bias/nasa` **`_raw_unit`** | **主口径**：raw 标签、飞行级预测、逐机算后等权平均 |
| 30–33 | 同四项 **`_cap125_unit`** | **补充口径**：标签截断到 125 后同法计算 |
| 34 | `rmse_raw_flightpooled` | 把所有飞行段混在一起算（不按机等权） |
| 35 | `rmse_cap125_flightpooled` | 同上，capped 口径 |
| 36–41 | `seg_*_raw` / `seg_*_cap125` | early / mid / late 三段的 RMSE，两种口径各一套 |
| 42 | `seg_boundaries` | 分段边界，**必须预先固定**，如 `early:RUL>75;mid:25<=RUL<=75;late:RUL<25` |
| 43 | `freeze_id` | 授权本次测试评估的冻结清单 id；`phase=dev` 留空 |
| 44 | `freeze_ts_utc` | 该冻结清单的 `freeze_ts_utc`（从清单抄，不自己写） |
| 45 | `eval_start_utc` | 本次测试评估**实际开始**时刻；必须 > `freeze_ts_utc` |
| 46 | `status` | `running` / `ok` / `failed` / `smoke` / `superseded` |
| 47 | `audit_verdict` | `pass` / `fail` / `n/a`（**只放判断，不放人名**） |
| 48 | `independent_unit` | 本协议固定 `engine`（见 §1.4） |
| 49 | `registry_note` | 异常、偏差、重跑原因、溯源信息都写这里 |
| 50 | `audited_by` | **只放人名**（如「实验结果管理员」）。不确定性和判断不许写进这一列 |

### 3.2 两条硬规则（从 C-MAPSS 台账的教训来）

1. **`audited_by` 只放人名。** 「来源不明」「抽检 5/24」这类溯源信息进 `registry_note`。
   把不确定性编码进身份字段，会让任何按审计人分组的统计失效。
2. **口径列不许混。** `_unit` 与 `_flightpooled` 是两种聚合，`raw` 与 `cap125` 是两种标签变换。
   四者组合里的任何一个数，引用时必须同时说明是哪一种。C-MAPSS 那边把 pooled 当 engine-level
   用了整整一晚，就是因为列名没带口径。

---

## 4. 特征白名单（记进 `feature_set` / `feature_list`）

依据 [DATASET_PROTOCOL.md](../DATASET_PROTOCOL.md) §3 与主控裁决：

| 组 | 列 | 允许 | 记法 |
|---|---|---|---|
| `W` | `alt, Mach, TRA, T2`（4 列，工况） | ✅ 可观测 | `W4` |
| `X_s` | 14 列实测传感器（`Wf, Nf, Nc, T24, T30, T48, T50, P15, P2, P21, P24, Ps30, P40, P50`） | ✅ 可观测 | `WS14` |
| `A.cycle` | 累计**飞行**周期（不是 1Hz 序号） | ✅ | `CYC` |
| `A.unit` | 机号 | ⚠️ **只作分组**，不得作为输入 | — |
| `X_v` | 14 列虚拟传感器 | ❌ 禁入 | — |
| `T` | 10 维健康参数 θ（**不是温度**） | ❌ 禁入（答案键） | — |
| `A.hs` / `A.Fc` | 健康状态 / 故障类别 | ❌ 禁入（可能泄漏） | — |
| `Y` | RUL（飞行周期） | 只进标签与评价分支 | — |

**台账校验**：`feature_list` 里出现任何禁入列 → 该行 `audit_verdict=fail`，退出主证据表。

---

## 5. 目录结构

```
docs/paper_b_ncmapss/registry/
  SCHEMA.md              # 本文件
  runs.csv               # 唯一台账（本管理员写）
  REGISTRY_NOTES.md      # 规则与逐次登记记录（append-only）
  FREEZE_TEMPLATE.json   # 冻结清单模板（见 §6）
  freeze/<freeze_id>.json# 实际冻结清单
  launch/<run_id>.json   # 训练进程启动时写（O_CREAT|O_EXCL，独占创建）
  close/<run_id>.json    # 结束后写

<workdir>/experiments/ncmapss/<run_id>/
  config.json  manifest.json  metrics.json  preds.csv  train.log  ckpt/
```

**写入边界**：本管理员只写 `docs/paper_b_ncmapss/registry/` 下的 `runs.csv`、
`REGISTRY_NOTES.md`、`freeze/*.json`、`SCHEMA.md`。**不写** `paper_b/ncmapss/`、
`experiments/ncmapss/`，**不改** C-MAPSS 的 `docs/paper_b_hc/`（一个字都不动）。

**同步方向**：台账以**本地仓库**为准，单向镜像到 3090 的同一相对路径；
`launch/`、`close/` 由 3090 上的训练进程写，单向拉回本地。两边都不双向写。

---

## 6. 冻结清单

模板见 [FREEZE_TEMPLATE.json](FREEZE_TEMPLATE.json)。三栏是本 schema 的硬要求：

| 栏 | 含义 | 为什么必须有 |
|---|---|---|
| `freeze_ts_utc` | 清单写定并提交的时刻 | 证明「先声明，后看结果」 |
| `freeze_content_sha256` | 清单内容（**去掉三栏后**）的 sha256 | 证明清单写完没被改过；去掉三栏是为了避免自引用 |
| `eval_start_utc` | 第一次读取测试机预测/标签的时刻 | 与上一栏比较，`eval_start_utc > freeze_ts_utc` 才算合规 |

**违规判定**：`eval_start_utc <= freeze_ts_utc`，或 `freeze_content_sha256` 与
提交版本的复算值不符 → 该批测试结果**不具备独立确认性**，只能作为探索记录。

---

## 7. 登记流程

1. 扫 `launch/` 新文件 → `runs.csv` 追加行，`status=running`，`audit_verdict=n/a`。
2. 扫 `close/` → 补 `closed_utc`、`epochs_run`、`best_epoch`、`ckpt_*`、`preds_path`。
3. 从 `experiments/ncmapss/<run_id>/metrics.json` 取指标，**不要手抄、不要二次计算**；
   `raw` 与 `cap125` 两套都要落。
4. **交叉核对**：用 `preds.csv` 复算 RMSE/MAE，与 `metrics.json` 差 < 1e-6，否则 `fail`。
5. 审计通过后写入 `NCMAPSS_FINAL_RESULTS.md` / `BASELINE_RESULTS.md` 的表
   （**只写数字，不改表结构**；那两份文档归郑总）。
6. 每次登记结束更新本文件顶部「最后更新」和 `REGISTRY_NOTES.md`。
7. 写回前做哈希乐观并发校验（见 `REGISTRY_NOTES.md` §4）。

---

## 8. 审计清单（每次至少抽查 3 条）

1. 产物齐全：`config.json` / `manifest.json` / `metrics.json` / `preds.csv` / `train.log` / `ckpt/best.pt`。
2. 条数对得上：`preds.csv` 行数 = 该 run 覆盖的飞行段数（`n_flights`）。
3. 能复算：`preds.csv` 重算 RMSE/MAE，与 `metrics.json` 差 < 1e-6；raw 与 cap125 两套都核。
4. 无泄漏：`feature_list` ⊆ 白名单；`X_v` / `T` / `A.hs` / `A.Fc` 零出现；`unit` 不在特征里。
5. 无未来信息：预测时刻 = 飞行结束；特征只用当前及过去航班；截掉未来航班后历史特征不变。
6. 无测试集调参：`phase=test` 的行必须有 `freeze_id`，且 `eval_start_utc > freeze_ts_utc`。
7. 预算一致：同 rung 各折的 epochs/patience/batch/lr 一致，差异必须有 `registry_note`。
8. 代码没漂移：`git_sha` + `code_md5` 与 `config_sha256` 对应的那份一致。
9. 种子真实：`seed` 与 `metrics.json`、日志里的 seed 一致；N0 必须为空。
10. 重复检查：同 `(phase, rung, fold, seed)` 是否已有 `ok` 行；重复标 `superseded` 并说明。
