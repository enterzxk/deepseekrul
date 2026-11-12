# N-CMAPSS DS02 实验台账 schema

版本：**v15** · 2026-11-13
状态：**schema 已定，0 run**（等数据接入后按此登记）
最后更新：2026-11-13 00:50 (+08)
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
| 6 折 × 3 seed × 4 rung | 72 | **N0 没有 seed 轴**。N0 是 age-only 小 MLP（4,353 参数），内部**固定 seed 101**、每折 1 格，所以 3 个 seed 只会得到同一组数 |

### 1.2 正确算法

**开发集（LOEO，6 折 = 留出 2 / 5 / 10 / 16 / 18 / 20 各一次）**

| rung | 折数 | seed 数 | 行数 | 说明 |
|---|---:|---:|---:|---|
| N0 age-only | 6 | **—** | **6** | **无 seed 轴（固定 seed 101）；模型为单输入小 MLP，4,353 参数**；每折 1 行（预测被留出的那台）。保留 MLP 是刻意的——时钟基线越强，DSPT 赢它才有意义 |
| N1 vanilla | 6 | 3 | **18** | 每折每 seed 训练 1 次 |
| N2 DSPT Base | 6 | 3 | **18** | 同上 |
| N3 frozen DSPT | 6 | 3 | **18** | MS/SA/HI 用 C-MAPSS 冻结值，**不许因 DS02 结果改** |
| **开发合计** | | | **60** | 其中**训练出的 checkpoint = 54 个** |

**测试集（官方单位 11 / 14 / 15，**冻结后**才评估）**

| rung | seed 数 | 单位数 | 行数 | 说明 |
|---|---:|---:|---:|---|
| N0 age-only | — | 3 | **3** | 小 MLP，固定 seed 101；是否存 checkpoint 见 §3.1 第 26–27 列 |
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

### 1.5 有效样本数不是 18

**6 折 × 3 seed = 18 个数字，但不是 18 个独立样本。**
- 独立单位是**发动机**（开发集 6 台），seed 只衡量优化随机性。
- 6 折的训练集高度重叠（每折只差 1 台），不能当独立重复。
- 台账用 `independent_unit=engine` 记这一点；任何跨 seed 的区间估计必须用 `df = n_engines − 1`，
  不是 `n_seeds − 1`（C-MAPSS 那边踩过这个坑：`SEED_ROBUSTNESS.md` 用 df=3 使区间偏窄 26%）。

---

### 1.6 一次训练、两个评价点 → 两行台账（v12）

不再有「一个 run 一行指标」的假设：**一次训练产出两行**，`selection_rule` 分别为 `inner_val` 与
`fixed_passes:200`，`run_id` 用 `-iv` / `-f200` 区分，配对键是 `(phase, rung, fold, seed)`。

| 阶段 | 训练/评估次数 | 评价点 | 台账行数 | 现状 |
|---|---:|---|---:|---|
| dev | 60（N0 6 + N1/N2/N3 各 18） | 2 | **120** | 已实现 |
| test | 30（N0 3 + N1/N2/N3 各 9） | 2 | **60** | **路径尚未实现，本批不产生 test 行** |
| **合计** | 90 | | **180** | 120 已可跑，60 待实现 |

**三条机械检查**（对应报表规则，缺一条就无法证明「没挑列」）：

1. 同一 `(phase,rung,fold,seed)` 必须**恰好两行**，缺一行 → 该格 `fail`。
2. 两行 `selection_rule` 必须不同（`inner_val` ≠ `fixed_passes:200`），相同 → `fail`（等于报了两遍同一列）。
3. 两行数值若**完全相同**，`coincident` 必须为 `true`；不同则为 `false`。
   **巧合与漏报在表里长得一样**，机械检查分不出来，所以只能要求显式声明。

> **`coincident` 没有 `n/a`。** 代码里 `select_epoch_star` 对 N0 同样调用，两个评价点无条件产出
> （`run_loeo.py` 每格两行），所以 **N0 也是两行**，只能是 `true`/`false`。此前写的「N0 单点 → n/a」
> 是错的：留着这个枚举值，等于给写入端一个错的合法值，审计第 20 条会放它过去。

**N1c 诊断臂不进本表。** `N1c = vanilla + cond` 用来查 N1 是否缺通路，`DIAG_RUNGS` 明写它
「never part of the 60-cell grid」。按裁决它落 **`experiments/ncmapss/diag/runs_diag.csv`**，
用同一套 63 列 schema、`phase=diag`，预测与指标也落 `experiments/ncmapss/diag/`。
**`runs_<subset>.csv` 是冻结网格的证据表，不包含非预注册行**；论文里 N1c 只出现在诊断附录。

**test 路径未实现，模型来源先写死：** `phase=test` 的 60 行本批不产生。将来实现时，
测试模型只能是**在全部 6 台 dev 机上重训、每 seed 一个 checkpoint**，**不拿任何一折的模型评 test**
（否则「哪一折代表最终模型」无定义）。


---

## 2. run_id 命名（强制）

```
^(N0|N1|N1c|N2|N3)_LOEO-<engine>(_s<seed>)?_ev<stop>$
```

示例：`N0_LOEO-2_ev20`、`N1_LOEO-2_s101_ev200`。

- `<engine>`：该折**留出**的发动机号（不是折序号）
- `_s<seed>`：**N0 省略**（无 seed 轴）
- `<stop>`：评价点，`epoch*` 的实际值（如 `ev20`、`ev63`）或固定 `ev200`
- **不用时间戳**：同一 (rung, fold, seed, stop) 在协议冻结后只允许存在一次，重跑按 §3.7 作废旧行
- 这条格式是**从实际产物反推**的（`preds_path` 里嵌着 run_id，120/120），改 schema 而不改 120 行数据

- `phase`：`dev` | `test` | `diag`（`diag` 只用于 `runs_diag.csv`）
- `rung`：`N0` | `N1` | `N2` | `N3`（N4/N5 等 C-MAPSS 机制冻结后再加，编号先占位）
- `fold`：`LOEO-2` | `LOEO-5` | `LOEO-10` | `LOEO-16` | `LOEO-18` | `LOEO-20`（留出的那台）；
  `phase=test` 时写 `ALLDEV`（在全部 6 台上重训）或 `LOEO-<n>`（若采用 §1.3 第二种做法）
- `seed`：`101` | `202` | `303`；**N0 省略整个 `-s<seed>` 段**
- `stop`：**评价点，`iv`（`inner_val`）/ `f200`（`fixed_passes:200`）**。一次训练产出两行，靠这一段区分；**同一 (phase,rung,fold,seed) 的两行 `train_id` 相同**（配对键就是这四个字段，不另设列）
- 时间戳 UTC 到分钟，取**训练结束**（`metrics.json` 落盘）时刻

示例：
```
NCM-dev-N3-LOEO-10-s101-20261113T0215Z
NCM-dev-N0-LOEO-10-20261113T0220Z
NCM-test-N3-ALLDEV-s101-20261120T0900Z
```

**同一 run_id 不得重跑。** 重跑换时间戳，并在 `registry_note` 写清「重跑原因 + 前一次 run_id」。

---

## 3. `runs_<subset>.csv` 列定义（64 列，固定）

```
run_id, subset, phase, rung, model, fold, held_out_engine, n_train_engines,
seed, git_sha, code_md5, data_sha256, split_hash, protocol_blob, train_plan_sha256, feature_set,
feature_list, kmeans_fit_units, kmeans_seed, config_path, config_sha256, freeze_source, freeze_source_sha256, selection_rule,
epochs_run, best_epoch, ckpt_path, ckpt_sha256, preds_path, pred_instant, n_units, n_flights,
rmse_raw_unit, mae_raw_unit, bias_raw_unit, nasa_raw_unit, rmse_cap125_unit, mae_cap125_unit, bias_cap125_unit, nasa_cap125_unit,
rmse_raw_flightpooled, rmse_cap125_flightpooled, seg_early_relative_capped, seg_early_relative_raw, seg_early_absolute_capped, seg_early_absolute_raw, seg_mid_relative_capped, seg_mid_relative_raw,
seg_mid_absolute_capped, seg_mid_absolute_raw, seg_late_relative_capped, seg_late_relative_raw, seg_late_absolute_capped, seg_late_absolute_raw, seg_boundaries, freeze_id,
freeze_ts_utc, eval_start_utc, status, audit_verdict, coincident, independent_unit, registry_note, audited_by
```

### 3.1 逐列含义

| # | 列 | 取值 / 说明 |
|---:|---|---|
| 1 | `run_id` | §2 格式，唯一 |
| 2 | `subset` | 子集 id：`DS02` / `DS05` / `DS06` / `DS07`。与文件名互为校验——行写错文件会被检查器抓出来；跨子集拼接时也不用靠文件名猜 |
| 3 | `phase` | `dev` / `test` / `diag`（`diag` 只用于 `runs_diag.csv`） |
| 4 | `rung` | `N0` / `N1` / `N2` / `N3` |
| 5 | `model` | 实际模型名，与 config 一致（`age_only` / `vanilla` / `dspt_base` / `dspt_frozen`） |
| 6 | `fold` | `LOEO-<留出机号>` / `ALLDEV` |
| 7 | `held_out_engine` | 该行预测的是哪台机；`phase=test` 时填测试机号 |
| 8 | `n_train_engines` | dev 折 = 5，test 重训 = 6 |
| 9 | `seed` | `101/202/303`；**N0 留空**（N0 内部固定 seed 101，不构成 seed 轴，所以台账不记 seed） |
| 10 | `git_sha` | 训练时工作区 SHA；`WORKTREE` 必须同时有 `code_md5` |
| 11 | `code_md5` | 关键文件 md5 的 JSON：`model` / `train` / `data` / `metrics` / `protocol` |
| 12 | `data_sha256` | 所用 HDF5 的 sha256 |
| 13 | `split_hash` | LOEO 折定义（留出机 + 训练机）的 sha256 |
| 14 | `protocol_blob` | 冻结协议文档的 git blob 哈希：`git rev-parse <freeze_commit>:docs/paper_b_ncmapss/DATASET_PROTOCOL.md`。**blob 抗 amend/rebase，commit 不抗**；对不上的 run 视为在未知协议下产出。**必须在仓库内用 `git rev-parse` 核，不要拿工作区文件算 sha256**——`.gitattributes` 规定 `*.md/*.json/*.csv` 为 `eol=lf`，Windows 上 CRLF 重存会让 blob 变而内容看起来没变 |
| 15 | `train_plan_sha256` | `paper_b/configs/protocol.json` 的 sha256。四档共用同一训练计划，**记哈希不记手抄数值** |
| 16 | `feature_set` | 白名单版本号，如 `WS14+W4+CYC` |
| 17 | `feature_list` | **实际进入模型的列名**，`;` 分隔，必须逐个落在 §4 白名单里 |
| 18 | `kmeans_fit_units` | 工况聚类（`n_cond=6`）拟合用的机号，`;` 分隔。**只许当前折的训练机**；N0/N1 留空。出现留出机 → 作废 |
| 19 | `kmeans_seed` | k-means 随机种子；N0/N1 留空 |
| 20 | `config_path` | `experiments/ncmapss/<run_id>/config.json` |
| 21 | `config_sha256` | 上述 config 的 sha256 |
| 22 | `freeze_source` | N3 冻结参数来源：`paper_b/results/FREEZE_ABLATION_SEED101.json` + `paper_b/configs/protocol.json`；非 N3 留空。**N3 = MS+SA，`hi_source=none`，本批不含 HI 臂**（见 §3.3） |
| 23 | `freeze_source_sha256` | 上述两份文件的内容哈希，JSON `{path: sha256}`。**不许手抄参数值** |
| 24 | `selection_rule` | `inner_val`（训练折内选轮）/ `fixed_passes:<n>` / `fixed_epochs:<n>` / `none`。**DS02 一律记 `inner_val`**（内层选轮，不是 40 步）；**N0 也是训练出来的小 MLP**，同样记 `inner_val`，不许写 `none` |
| 25 | `epochs_run` | 实际跑完的 epoch 数 |
| 26 | `best_epoch` | 选轮得到的 epoch；`selection_rule=none` 时留空 |
| 27 | `ckpt_path` | `experiments/ncmapss/<run_id>/ckpt/best.pt`。**N0 也有 4,353 参数，应当存 checkpoint**；若实现不存，必须在 `registry_note` 写明原因，不能留空不说 |
| 28 | `ckpt_sha256` | checkpoint 的 sha256；冻结清单逐条列出。N0 同上，留空须在 `registry_note` 说明 |
| 29 | `preds_path` | 逐发动机预测 `preds.csv`，**固定 5 列**：`unit, cycle, y_true_raw, y_true_cap125, y_pred`。**不得出现裸 `y_true`**——它等于 `min(raw,125)`，是一个默认走 capped 的隐式口径；口径由 `evaluate(criterion=...)` 显式选，CSV 只提供两个标好名的标签。列名用数据集自己的 `unit` / `cycle`，不用 `engine_id` / `flight_cycle` |
| 30 | `pred_instant` | 预测时刻，固定 `flight_end` |
| 31 | `n_units` | 该行涉及的发动机数（dev 折 = 1，test = 1） |
| 32 | `n_flights` | 该行涉及的飞行段数（预测点个数） |
| 33 | `rmse_raw_unit` | **主口径**：raw 标签、飞行级预测、逐机算后等权平均 |
| 34 | `mae_raw_unit` | 同上口径的 MAE |
| 35 | `bias_raw_unit` | 同上口径的 Bias |
| 36 | `nasa_raw_unit` | 同上口径的 NASA Score（必须同时声明求和规则与样本数） |
| 37 | `rmse_cap125_unit` | **补充口径**：标签截断 125 后同法计算 |
| 38 | `mae_cap125_unit` | capped 口径 MAE |
| 39 | `bias_cap125_unit` | capped 口径 Bias |
| 40 | `nasa_cap125_unit` | capped 口径 NASA Score |
| 41 | `rmse_raw_flightpooled` | 所有飞行段混算（不按机等权），raw 口径。**补充**，不是主口径 |
| 42 | `rmse_cap125_flightpooled` | 同上，capped 口径 |
| 43 | `seg_early_relative_capped` | early 段 RMSE，相对分段（按该机**完整寿命**的三分位），capped 口径 |
| 44 | `seg_early_relative_raw` | early 段 RMSE，相对分段（按该机**完整寿命**的三分位），raw 口径 |
| 45 | `seg_early_absolute_capped` | early 段 RMSE，绝对分段（RUL 阈值，见 `seg_boundaries`），capped 口径 |
| 46 | `seg_early_absolute_raw` | early 段 RMSE，绝对分段（RUL 阈值，见 `seg_boundaries`），raw 口径 |
| 47 | `seg_mid_relative_capped` | mid 段 RMSE，相对分段（按该机**完整寿命**的三分位），capped 口径 |
| 48 | `seg_mid_relative_raw` | mid 段 RMSE，相对分段（按该机**完整寿命**的三分位），raw 口径 |
| 49 | `seg_mid_absolute_capped` | mid 段 RMSE，绝对分段（RUL 阈值，见 `seg_boundaries`），capped 口径 |
| 50 | `seg_mid_absolute_raw` | mid 段 RMSE，绝对分段（RUL 阈值，见 `seg_boundaries`），raw 口径 |
| 51 | `seg_late_relative_capped` | late 段 RMSE，相对分段（按该机**完整寿命**的三分位），capped 口径 |
| 52 | `seg_late_relative_raw` | late 段 RMSE，相对分段（按该机**完整寿命**的三分位），raw 口径 |
| 53 | `seg_late_absolute_capped` | late 段 RMSE，绝对分段（RUL 阈值，见 `seg_boundaries`），capped 口径 |
| 54 | `seg_late_absolute_raw` | late 段 RMSE，绝对分段（RUL 阈值，见 `seg_boundaries`），raw 口径 |
| 55 | `seg_boundaries` | 两套边界同时装：`{"relative": [0.3333, 0.6667], "absolute": [75, 25]}`。**先冻结再跑**；`absolute` 的 (75,25) 是待 DS02 真实 RUL 尺度确认后可改**一次**的默认值，不许看完结果调 |
| 56 | `freeze_id` | 授权本次测试评估的冻结清单 id；`phase=dev` 留空 |
| 57 | `freeze_ts_utc` | 该冻结清单的 `freeze_ts_utc`（从清单抄，不自己写） |
| 58 | `eval_start_utc` | 本次测试评估**实际开始**时刻；必须 > `freeze_ts_utc` |
| 59 | `status` | `running` / `ok` / `failed` / `smoke` / `superseded` |
| 60 | `audit_verdict` | `pass` / `partial` / `fail` / `n/a`（**只放判断，不放人名**）。`partial` = 指标可从产物复算、但权重不可验（无 checkpoint），**不得读成完全通过** |
| 61 | `coincident` | **只有 `true` / `false`**。双评价点两行数值**完全相同**时必须 `true`（巧合与漏报在表里长得一样，机械检查分不出来）；不同则 `false`。**没有 `n/a`**：代码对 N0 也产出两个评价点，单点档在本网格不可达 |
| 62 | `independent_unit` | 固定 `engine`（见 §1.5） |
| 63 | `registry_note` | 异常、偏差、重跑原因、溯源信息都写这里 |
| 64 | `audited_by` | **只放人名**。不确定性和判断不许写进这一列 |

### 3.2 两条硬规则（从 C-MAPSS 台账的教训来）

1. **`audited_by` 只放人名。** 「来源不明」「抽检 5/24」这类溯源信息进 `registry_note`。
   把不确定性编码进身份字段，会让任何按审计人分组的统计失效。
2. **口径列不许混。** `_unit` 与 `_flightpooled` 是两种聚合，`raw` 与 `cap125` 是两种标签变换。
   四者组合里的任何一个数，引用时必须同时说明是哪一种。C-MAPSS 那边把 pooled 当 engine-level
   用了整整一晚，就是因为列名没带口径。

---

### 3.3 训练计划与工况编码（v2 新增）

**训练计划四档共用，来自 `paper_b/configs/protocol.json`，不许改：**

| 项 | 值 |
|---|---|
| epochs | 40 |
| patience | 12 |
| batch | `min(512, N_train)` |
| lr | 1e-3 |
| weight_decay | 1e-4 |
| d_model | 64 |
| layers | 2 |
| dropout | 0.2 |
| seeds | 101 / 202 / 303 |

只允许改输入宽度（51 列）。**哪一档在这个计划下不收敛，那是要报的发现，不是改超参的许可证**——
和 N3 冻结是同一条规则。台账每行记 `train_plan_sha256`（对 `protocol.json` 算），不记手抄数值。

**工况编码 `n_cond=6`（DS02 重新定义，信息匹配的关键点）：**

- 对 `W`（4 列工况）做 k-means(k=6)，**只在当前 LOEO 折的训练机上拟合**。
- 台账记 `kmeans_fit_units` 与 `kmeans_seed`。
- **硬校验**：`kmeans_fit_units` 里出现该折的留出机 → 该 run 直接作废（`audit_verdict=fail`）。
- 论证必须写进代码注释：**这个 embedding 不带来 N1 没有的信息**——`W` 的 mean/std 本来就在共享 51 列里，
  聚类只是同一种信息的另一种编码。任何让 DSPT 档「输入严格更丰富」的改动是协议违规，不是架构选择。

**N3 冻结来源与范围（写死，防止从台账反推出不存在的证据）：**

`freeze_source` = `paper_b/results/FREEZE_ABLATION_SEED101.json` + `paper_b/configs/protocol.json`，
两者内容哈希记入 `freeze_source_sha256`。**不许在 DS02 侧手抄一份参数值**——抄一份参数不等于「冻结」。

> **N3 = MS+SA，`hi_source=none`。本批 N0–N3 不含 HI 臂。**
> HI 的跨数据集结论由 **N4**（DSPT + Structural HI）/ **N5**（DSPT-HC）产生，两者要等 C-MAPSS
> 核心 HI 结构冻结后才跑正式对照，**本轮不产生任何 HI 结论**。
> 台账不得出现任何暗示「N3 测过 HI」的表述；出现即视为伪造证据。

**协议冻结锚（当前）：**

| 项 | 值 |
|---|---|
| `freeze_commit` | **`9ac1744`** |
| `protocol_blob` | **`a3c40c634da7a3bb44ace788f74896b17e51c3fb`** |
| 核验命令 | `git rev-parse 9ac1744:docs/paper_b_ncmapss/DATASET_PROTOCOL.md` |

**完整锚历史（从 git 重建，共 13 代）**——注意：审计早期写的是「五次前移」，**那个数字是错的**，协议正文实际有 13 个不同 blob——协议正文在本分支上一共产生过 **13 个不同 blob**，
不是五次。只记最后几次会让「这个 run 用的是哪版协议」出现缺口：

| commit | blob | 说明 |
|---|---|---|
| `c481289` | `2e4df8dd4fc4895f4395d0a50758d9bac33d5826` | ncmapss: DS02 branch skeleton - protocol discipline, baseline ladder,  |
| `4bfa5d8` | `386b13a574c8e325fc3848d1c3513602d6afa964` | ncmapss: DS02 data dictionary from the official tables + observability |
| `7cc5824` | `1fa8ec4cae30223b00ab5137ef9bbee3eddec1b0` | ncmapss: evaluation protocol section (6.1-6.9) - unit of analysis, LOE |
| `e1de832` | `2e3f36b08847d56e74c3c5e5d1b4703c1a6b86d9` | ncmapss: freeze the 51-column flight feature contract and the span=70  |
| `c69db46` | `668c98a9e64fe4ae2b3314a54da45571ac6e38d0` | ncmapss: freeze the training plan (protocol.json verbatim) and the k=6 |
| `d9e49af` | `dc347037629e589000c154c01e3dd005f71607fd` | ncmapss: split schedule from label construction - rul_cap/val_cuts not |
| `5b228ad` | `6c184bbfe395325c367b1e460689b178ae48f1f3` | ncmapss: bound the 'same core structure' claim - stopping rule differs |
| `7f01909` | `4458d9ea05c965251aabb782bba9765304ebf686` | ncmapss: eval protocol corrections before any run - primary=engine_mea |
| `922a2d1` | `ebfbfcb8d41c7f4dd0bfc3d0c35a50225b57d7d1` | ncmapss: amend the epoch budget to 1000 optimizer steps and mark the 1 |
| `e7ad15f` | `35b40d8bce8e2b6ee39408c2e36e39482462f05e` | ncmapss: fix the budget on passes not steps (40 passes, batch 64 -> 24 |
| `1c76b37` | `443462ca03087d5aff2058778de2d47b3331728c` | ncmapss: fold-local target standardisation and a 200-pass budget (1200 |
| `7e9415e` | `21af549e448415113ee2ac61f11a5679400eddac` | ncmapss: replace the fixed budget with inner-validation early stopping |
| **`9ac1744`** | **`a3c40c634da7a3bb44ace788f74896b17e51c3fb`** | **当前冻结锚** |

> **`9ac1744` 之后协议冻结，直到第一批 LOEO 开发集结果出来为止不再改 §4/§6。**
> 分界线（写进审计）：**改动必须由「测量失效」证明，不能由「数字不符合预期」触发**。
> 全量开跑之后再改，就不再是修仪器，而是挑选。


---

### 3.4 分段误差：两套规则 × 两种口径（v3 修正）

12 列 = **3 段 × 2 规则 × 2 口径**，指标只放 RMSE：

- **规则维不能省。** `relative`（按该机完整寿命三分位）是本轮新增的主规则，`absolute`（RUL 阈值）
  用来对齐 C-MAPSS 表。v2 把这一维漏了，两套边界只能留一套，等于把新增的东西丢了。
- **MAE 不进这 12 列。** 分段表里 RMSE 是主；逐段 MAE 由 `evaluate()` 返回、留在每次 run 的
  逐段 CSV 里，台账不冻结它，避免表撑到 24 列。
- **相对分段的归一化尺度必须来自该机完整标签轨迹**，不能来自被评估子集。
  测试集每台机只评最后一个飞行，若用 `block.max()`，`frac` 恒为 1.0，所有行都被判成 `early`，
  指标退化成常数且**不报错**。实现上由数据层传入 `unit_lifetime: dict[unit, max_rul]`，缺了就报错。
- **`absolute` 的 (75, 25) 是默认值**，可在 DS02 真实 RUL 尺度确认后改**一次**，
  且必须改在冻结之前；看完结果再调 = 协议违规。

---

### 3.5 capped 口径在 DS02 上失效（实测，v6）

实测：dev 最长 **89 个飞行**（unit 5）、test 最长 **76 个飞行**；对应 `max(Y)` = **88 / 75**。全部 < 125，因此
`y_true_cap125 == y_true_raw` **处处成立**，capped 不是第二个独立口径。

| 事项 | 规则 |
|---|---|
| `pred.csv` 两列 | **照写**（审计要求保留原始与封顶两个标签，用户 §六） |
| 结果表 | capped 标 `inactive (no unit exceeds the cap)`，**不与 raw 并排印表** |
| 12 个 `seg_*_capped` 列 | 字段结构不变，照常登记；**报表时标 inactive** |
| 措辞 | **不得称 capped 为「更严格的口径」**，也不得把 capped 与 raw 的差异算作模型属性——那是标签定义的属性 |

**口径区分（务必分清）**：`89` / `76` 是**飞行数**；相对三分位的分母是 `max(Y)`，官方标签在最后一个飞行上是 0，所以 **`max(Y)` = 88 / 75**。`unit_lifetime()` 返回的就是 `max(Y)`（`data.py:184`），`evaluation.py:359` 拿它当分母。结果表里这一列叫 **`max_rul`**，要报飞行数另起一列 `n_flights`——两者差 1，但定义不同，混用会让寿命列对不上分段分母。

> 这条正好是 C-MAPSS 那晚的反面：那边 cap 生效、capped/raw 差一倍；这边 cap 不生效、两者恒等。
> 所以「哪个口径更严」不是先验结论，必须按数据集实测，写进台账而不是靠记忆。

---

### 3.6 训练预算与停止规则（现行口径，v11 校正）

> **注意**：本节此前写的是「40 遍、batch 64、240 次更新」，那已被协议第三次修订取代。
> 现行口径以 `9ac1744` 的 `DATASET_PROTOCOL.md` §4.2 为准，下面是照抄。

| 项 | 值 |
|---|---|
| 过数据遍数 | **200**（`epochs = 200`） |
| batch | **`min(64, N_train)` = 64** → 每遍 6 次更新 |
| 总更新数 | **≈1200**（与 C-MAPSS 的 ~1560 同量级） |
| 停止规则 | **内层验证早停，绝不在留出机上停** |
| `epoch*` 取法 | 外层折 `u_o` 的 5 台训练机轮换出 1 台做内层验证 → 5 条内层曲线 → 取**均值曲线的 argmin**，地板 20 遍 |
| 重训 | 在全部 5 台训练机上按 `epoch*` 精确重训，无早停，再评 `u_o` |
| 内层训练上限 | 400 遍 |
| `patience 12` | 记录但**不作为停止规则** |

**两个上报点（同一训练、各存一次权重、各评一次）：**

| 评价点 | `selection_rule` | `run_id` 段 | 说明 |
|---|---|---|---|
| `inner_val` | `inner_val` | `-iv` | 内层均值曲线 argmin 处的 `epoch*` |
| `fixed_passes:200` | `fixed_passes:200` | `-f200` | 固定跑满 200 遍的末点 |

两列并排、同等字号、表头写明停止规则；**正文不许挑有利的一列下结论**。
若两列排序不同，那个分歧本身是发现。

**为什么报两个点**：实测 `epoch*` 落在 20–94，而外层曲线最优点在 157–198，**系统性错位约 2 倍**；
同一档 5 个内层最小值的散布达 13–19 倍。也就是说，在六台机上「什么时候停」既不可估计、也不可迁移——
**报一个单点就是在藏这件事**。

**诊断字段**（`metrics.json`，不进台账列）：`holdout_min`、`holdout_min_epoch`，
标 `diagnostic only - not used for model selection`。不做早停、报分取固定点，固定预算只有在曲线
也藏起来的时候才藏得住过拟合。


---

### 3.7 作废（VOID）的登记方式（v11）

被查出缺陷、数字不得再引用的 run **不删、不静默覆盖**，按下面这套记：

| 字段 | 值 |
|---|---|
| `status` | `superseded`（被补齐后的新 run 取代） |
| `audit_verdict` | `fail` |
| `registry_note` | **以 `VOID:` 开头**，写清缺了哪条通路/哪个缺陷、当时比的是什么、替代 run_id |
| 新 run | 新行，`registry_note` 以 `SUPERSEDES:` 开头并指向旧 run_id |

**不新增状态值**：`status` 仍只有 `running/ok/failed/smoke/superseded`，`audit_verdict` 仍只有
`pass/fail/n/a`。需要表达「作废」时用上表组合，不要再造 `void` 之类的枚举——多一个状态就多一处判断分叉。

> 适用场景（当前）：N1 若确认缺 `cond` / `dual_span` 通路，旧数字按此作废；
> 读者必须能看到「这个对照存在过、为什么撤回、被什么替代」。

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
  runs_<subset>.csv      # 每子集一张台账（本管理员写）
  REGISTRY_NOTES.md      # 规则与逐次登记记录（append-only）
  FREEZE_TEMPLATE.json   # 冻结清单模板（见 §6）
  freeze/<freeze_id>.json# 实际冻结清单
  launch/<run_id>.json   # 训练进程启动时写（O_CREAT|O_EXCL，独占创建）
  close/<run_id>.json    # 结束后写

<workdir>/experiments/ncmapss/<run_id>/
  config.json  manifest.json  metrics.json  preds.csv  train.log  ckpt/
```

**写入边界**：本管理员只写 `docs/paper_b_ncmapss/registry/` 下的 `runs_<subset>.csv`、
`REGISTRY_NOTES.md`、`freeze/*.json`、`SCHEMA.md`。**不写** `paper_b/ncmapss/`、
`experiments/ncmapss/`，**不改** C-MAPSS 的 `docs/paper_b_hc/`（一个字都不动）。

**同步方向**：台账以**本地仓库**为准，单向镜像到 3090 的同一相对路径；
`launch/`、`close/` 由 3090 上的训练进程写，单向拉回本地。两边都不双向写。

---

## 6. 冻结清单

模板见 [FREEZE_TEMPLATE.json](FREEZE_TEMPLATE.json)；现有清单与作废状态见
[freeze/INDEX.md](freeze/INDEX.md)。

| 栏 | 含义 | 为什么必须有 |
|---|---|---|
| `freeze_ts_utc` | 清单写定并提交的时刻 | 证明「先声明，后看结果」；**必须参与哈希**（v17 起），否则可以被改早而不被发现 |
| `freeze_content_sha256` | 清单内容（**去掉两栏后**）的 sha256 | 证明清单写完没被改过。排除哈希自身是为了避免自引用；排除 `eval_start_utc` 是因为它在开封之后才写 |
| `eval_start_utc` | 第一次读取测试机预测/标签的时刻 | 与上一栏比较，`eval_start_utc > freeze_ts_utc` 才算合规 |

**违规判定**：`eval_start_utc <= freeze_ts_utc`，或 `freeze_content_sha256` 与
提交版本的复算值不符 → 该批测试结果**不具备独立确认性**，只能作为探索记录。

**v17 改了什么**：v16 把 `freeze_ts_utc` 排除在哈希外，于是唯一能证明「先冻后开」的字段
自己不受保护——把它改成任意更早的时刻，摘要照样对得上。v17 只排除两栏。
代价是 v16 签的清单在 v17 规则下摘要必然不符，所以**旧清单的摘要必须按签名时的规则复算**，
这一点写在 `freeze/INDEX.md`。

**重签不能修复时序**：清单晚于第一次读测试集时，重签只是把同一件事挪到更晚的时刻。
正确做法是如实写进 `sequencing.honest_limitation`，并用 `completeness` 证明没有选择自由度。

**三条结构性字段**（v17 起，缺一即视为清单不完整）：

| 栏 | 断言 |
|---|---|
| `completeness` | `runs[]` == 完整叉积 40/40，`excluded_cells == 0`。完整叉积在结构上无法表达「只报有利的格」，闸门要防的自由度因此不存在 |
| `epoch_star_values` | 40 格的 `epoch*` 与六折原值，签名时从封存台账机械复算。运行器**读取**这些数，不许自己再算——否则 §6.10 的「开封前冻结」无法执行 |
| `sequencing` | 冻结之前是否读过测试集、时序是否违约、与上一版清单逐项差了什么 |

一条命令自证（哈希 / 完整性 / `epoch*` / 时序四项）：

```bash
python scripts/nc_freeze.py verify docs/paper_b_ncmapss/registry/freeze/<active>.json
```

---

## 7. 登记流程

**出汇总前的硬门禁（v14，来自一次真实事故）**：

```python
# 每个 (rung, held_out_engine) 的 seed 数：N0 必须恰好 1，其余必须恰好 3
for (rung, eng), seeds in by_cell.items():
    want = 1 if rung == "N0" else 3
    assert len(seeds) == want, f"{rung} engine {eng}: {len(seeds)} seeds, expected {want}"
```

**不齐就拒绝出汇总**，报错信息里点名是哪一格少 seed。理由：2026-11-12 有人用 116/120 行的
中间数据算了最终表，其中 N3 的 engine 20 只有 1 个 seed（16.61），被当成 3 seed 均值报了出去
（实际 12.77），六折均值也跟着错（14.17 → 13.53）。**1 个 seed 的均值和 3 个 seed 的均值在表里
长得一模一样，只有数字不同**——这种错不会自己暴露，只有断言能拦。

1. 扫 `launch/` 新文件 → `runs_<subset>.csv` 追加行，`status=running`，`audit_verdict=n/a`。
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
9. 种子真实：`seed` 与 `metrics.json`、日志里的 seed 一致；**N0 的 `seed` 列为空**，但实现内部必须固定 seed 101（日志里要能看到），且不得出现 seed 202/303 的 N0 行。
10. 重复检查：同 `(phase, rung, fold, seed)` 是否已有 `ok` 行；重复标 `superseded` 并说明。
11. **测试行不含 LOEO checkpoint**：`phase=test` 的 `ckpt_sha256` 必须能在冻结清单的 9 个
    「全开发机重训」checkpoint 里找到；命中任何 LOEO checkpoint → `fail`，冻结作废。
12. **测试集离散度口径**：`phase=test` 的汇总只能出现 seed 均值 ± seed sd；
    出现「跨机 CI」「跨机 sd」字样 → `fail`（测试机只有 3 台，没有这个精度）。
13. **工况聚类只用训练机**：`kmeans_fit_units` ⊆ 该折训练机集合；
    出现留出机或测试机 → `fail`，该 run 退出主证据表。
14. **训练计划未被改**：`train_plan_sha256` 与 `protocol.json` 实算值一致；
    四档必须是同一个哈希。不一致 → `fail`（改超参不是调参自由，是协议违规）。
15. **N3 冻结来源可追**：`freeze_source_sha256` 的两份文件哈希与仓库实算一致；
    台账里出现手抄的参数值而非读取来源 → `fail`。
16. **分段规则齐全**：12 个 `seg_*` 列必须两套规则（`relative` + `absolute`）都齐；
    `relative` 的归一化尺度来自该机完整寿命而非被评估子集（否则测试集会退化成常数）；
    缺任一套或尺度用错 → `fail`。
17. **协议 blob 对得上**：`protocol_blob` 与 `git rev-parse <freeze_commit>:docs/paper_b_ncmapss/DATASET_PROTOCOL.md`
    一致。不一致 → `fail`，该 run 视为在未知协议下产出（commit 可被 amend，blob 不会）。
    **核验必须在仓库内跑 `git rev-parse`，不得拿工作区文件算 sha256 去比**：`.gitattributes`
    把 `*.md/*.json/*.csv` 定为 `eol=lf`，工作区里一次 CRLF 重存就会让字节变而内容看起来没变。
18. **`preds.csv` 无裸口径列**：列必须恰好是 `unit, cycle, y_true_raw, y_true_cap125, y_pred`；
    出现裸 `y_true`、`engine_id`、`flight_cycle` 任一 → `fail`。裸 `y_true` 等于 `min(raw,125)`，
    是隐式的 capped 默认口径，和「池化当发动机级」是同一类错误。

19. **训练预算按 内层选轮**：从该 run 的 `config.json` 取 `N_train`（训练窗口数），核对
    `batch == min(64, N_train)` 且 `epoch_star is set`，且 `N_train/batch` 向上取整 = 6 次/遍。
    不等于 → `fail`（跑错预算）。**故意不加 `n_train_windows` 列**：开跑在即，加列会让写入端与台账错位。

20. **双评价点成对**：同 `(phase,rung,fold,seed)` 恰好两行（`-iv` + `-f200`），
    两行 `selection_rule` 不同；数值相同则 `coincident=true`，不同则 `false`。
    缺行、重复、出现 `n/a`、`coincident` 与数值不一致 → `fail`。
    **为什么 `n/a` 算错**：代码对 **N0 也调用 `select_epoch_star`、无条件产出两个评价点**，
    所以本网格里不存在「只有单评价点的档」。`n/a` 是不可达值——留着它等于给写入端一个
    合法但错误的取值，而审计会把 `n/a` 当成「没填」放过去。半年后看到 `n/a` 的人
    不会知道它是不可达，只会以为是漏写，所以这里把「为什么不可达」写死。

## 9. 开跑前自检（每批 run 之前跑一遍，任何一条不过就不许开跑）

```bash
cd <repo>
# 1. 协议没漂移：冻结提交、当前 HEAD、台账三处 blob 必须相同
git rev-parse 9ac1744:docs/paper_b_ncmapss/DATASET_PROTOCOL.md   # 期望 a3c40c634da7a3bb44ace788f74896b17e51c3fb
git rev-parse HEAD:docs/paper_b_ncmapss/DATASET_PROTOCOL.md      # 必须与上一行相同
grep -c a3c40c634da7a3bb44ace788f74896b17e51c3fb docs/paper_b_ncmapss/registry/runs_DS02.csv  # 每行都要命中

# 2. 工作区干净（3090 是只读消费者）
git fetch && git reset --hard origin/ncmapss-ds02
git status --porcelain            # 必须为空

# 3. 冻结文件在且是冻结态
test -f paper_b/configs/protocol.json
test -f paper_b/results/FREEZE_ABLATION_SEED101.json
python -c "import json;d=json.load(open('paper_b/results/FREEZE_ABLATION_SEED101.json'));assert d['frozen'] is True"

# 3b. 冻结清单自证：哈希 / 完整性 / epoch* / 时序，四项全 OK 才许开跑
python scripts/nc_freeze.py verify docs/paper_b_ncmapss/registry/freeze/<active>.json

# 4. 台账锁没有残留
test ! -e docs/paper_b_ncmapss/registry/runs_<subset>.csv.lock
```

**第 1 条是最便宜、也最容易漏的一条**：一条 `git rev-parse` 就能证明协议正文在冻结后没被动过
（2026-11-12 就用它证明过 `ed8f0cd` 只动了复现审计、`DATASET_PROTOCOL.md` 的 blob 未变）。
开跑后再发现协议漂移，那一批 run 全部作废。

第 2 条对应工作区拓扑：**`origin/ncmapss-ds02` 是唯一真源，3090 是只读消费者**，
每批跑之前 `reset --hard`，不在 3090 上提交、不留未跟踪的代码文件。

