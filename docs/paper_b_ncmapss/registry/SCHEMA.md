# N-CMAPSS DS02 实验台账 schema

版本：**v8** · 2026-11-12
状态：**schema 已定，0 run**（等数据接入后按此登记）
最后更新：2026-11-12 22:05 (+08)
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

## 3. `runs.csv` 列定义（62 列，固定）

```
run_id, phase, rung, model, fold, held_out_engine, n_train_engines, seed,
git_sha, code_md5, data_sha256, split_hash, protocol_blob, train_plan_sha256, feature_set, feature_list,
kmeans_fit_units, kmeans_seed, config_path, config_sha256, freeze_source, freeze_source_sha256, selection_rule, epochs_run,
best_epoch, ckpt_path, ckpt_sha256, preds_path, pred_instant, n_units, n_flights, rmse_raw_unit,
mae_raw_unit, bias_raw_unit, nasa_raw_unit, rmse_cap125_unit, mae_cap125_unit, bias_cap125_unit, nasa_cap125_unit, rmse_raw_flightpooled,
rmse_cap125_flightpooled, seg_early_relative_capped, seg_early_relative_raw, seg_early_absolute_capped, seg_early_absolute_raw, seg_mid_relative_capped, seg_mid_relative_raw, seg_mid_absolute_capped,
seg_mid_absolute_raw, seg_late_relative_capped, seg_late_relative_raw, seg_late_absolute_capped, seg_late_absolute_raw, seg_boundaries, freeze_id, freeze_ts_utc,
eval_start_utc, status, audit_verdict, independent_unit, registry_note, audited_by
```

### 3.1 逐列含义

| # | 列 | 取值 / 说明 |
|---:|---|---|
| 1 | `run_id` | §2 格式，唯一 |
| 2 | `phase` | `dev` / `test` |
| 3 | `rung` | `N0` / `N1` / `N2` / `N3` |
| 4 | `model` | 实际模型名，与 config 一致（`age_only` / `vanilla` / `dspt_base` / `dspt_frozen`） |
| 5 | `fold` | `LOEO-<留出机号>` / `ALLDEV` |
| 6 | `held_out_engine` | 该行预测的是哪台机；`phase=test` 时填测试机号 |
| 7 | `n_train_engines` | dev 折 = 5，test 重训 = 6 |
| 8 | `seed` | `101/202/303`；**N0 留空**（N0 内部固定 seed 101，不构成 seed 轴，所以台账不记 seed） |
| 9 | `git_sha` | 训练时工作区 SHA；`WORKTREE` 必须同时有 `code_md5` |
| 10 | `code_md5` | 关键文件 md5 的 JSON：`model` / `train` / `data` / `metrics` / `protocol` |
| 11 | `data_sha256` | 所用 HDF5 的 sha256 |
| 12 | `split_hash` | LOEO 折定义（留出机 + 训练机）的 sha256 |
| 13 | `protocol_blob` | 冻结协议文档的 git blob 哈希：`git rev-parse <freeze_commit>:docs/paper_b_ncmapss/DATASET_PROTOCOL.md`。**blob 抗 amend/rebase，commit 不抗**；对不上的 run 视为在未知协议下产出。**必须在仓库内用 `git rev-parse` 核，不要拿工作区文件算 sha256**——`.gitattributes` 规定 `*.md/*.json/*.csv` 为 `eol=lf`，Windows 上 CRLF 重存会让 blob 变而内容看起来没变 |
| 14 | `train_plan_sha256` | `paper_b/configs/protocol.json` 的 sha256。四档共用同一训练计划，**记哈希不记手抄数值** |
| 15 | `feature_set` | 白名单版本号，如 `WS14+W4+CYC` |
| 16 | `feature_list` | **实际进入模型的列名**，`;` 分隔，必须逐个落在 §4 白名单里 |
| 17 | `kmeans_fit_units` | 工况聚类（`n_cond=6`）拟合用的机号，`;` 分隔。**只许当前折的训练机**；N0/N1 留空。出现留出机 → 作废 |
| 18 | `kmeans_seed` | k-means 随机种子；N0/N1 留空 |
| 19 | `config_path` | `experiments/ncmapss/<run_id>/config.json` |
| 20 | `config_sha256` | 上述 config 的 sha256 |
| 21 | `freeze_source` | N3 冻结参数来源：`paper_b/results/FREEZE_ABLATION_SEED101.json` + `paper_b/configs/protocol.json`；非 N3 留空。**N3 = MS+SA，`hi_source=none`，本批不含 HI 臂**（见 §3.3） |
| 22 | `freeze_source_sha256` | 上述两份文件的内容哈希，JSON `{path: sha256}`。**不许手抄参数值** |
| 23 | `selection_rule` | `inner_val`（训练折内选轮）/ `fixed_passes:<n>` / `fixed_epochs:<n>` / `none`。**DS02 一律记 `fixed_passes:200`**（200 遍，不是 40 步）；**N0 也是训练出来的小 MLP**，同样记 `fixed_passes:200`，不许写 `none` |
| 24 | `epochs_run` | 实际跑完的 epoch 数 |
| 25 | `best_epoch` | 选轮得到的 epoch；`selection_rule=none` 时留空 |
| 26 | `ckpt_path` | `experiments/ncmapss/<run_id>/ckpt/best.pt`。**N0 也有 4,353 参数，应当存 checkpoint**；若实现不存，必须在 `registry_note` 写明原因，不能留空不说 |
| 27 | `ckpt_sha256` | checkpoint 的 sha256；冻结清单逐条列出。N0 同上，留空须在 `registry_note` 说明 |
| 28 | `preds_path` | 逐发动机预测 `preds.csv`，**固定 5 列**：`unit, cycle, y_true_raw, y_true_cap125, y_pred`。**不得出现裸 `y_true`**——它等于 `min(raw,125)`，是一个默认走 capped 的隐式口径；口径由 `evaluate(criterion=...)` 显式选，CSV 只提供两个标好名的标签。列名用数据集自己的 `unit` / `cycle`，不用 `engine_id` / `flight_cycle` |
| 29 | `pred_instant` | 预测时刻，固定 `flight_end` |
| 30 | `n_units` | 该行涉及的发动机数（dev 折 = 1，test = 1） |
| 31 | `n_flights` | 该行涉及的飞行段数（预测点个数） |
| 32 | `rmse_raw_unit` | **主口径**：raw 标签、飞行级预测、逐机算后等权平均 |
| 33 | `mae_raw_unit` | 同上口径的 MAE |
| 34 | `bias_raw_unit` | 同上口径的 Bias |
| 35 | `nasa_raw_unit` | 同上口径的 NASA Score（必须同时声明求和规则与样本数） |
| 36 | `rmse_cap125_unit` | **补充口径**：标签截断 125 后同法计算 |
| 37 | `mae_cap125_unit` | capped 口径 MAE |
| 38 | `bias_cap125_unit` | capped 口径 Bias |
| 39 | `nasa_cap125_unit` | capped 口径 NASA Score |
| 40 | `rmse_raw_flightpooled` | 所有飞行段混算（不按机等权），raw 口径。**补充**，不是主口径 |
| 41 | `rmse_cap125_flightpooled` | 同上，capped 口径 |
| 42 | `seg_early_relative_capped` | early 段 RMSE，相对分段（按该机**完整寿命**的三分位），capped 口径 |
| 43 | `seg_early_relative_raw` | early 段 RMSE，相对分段（按该机**完整寿命**的三分位），raw 口径 |
| 44 | `seg_early_absolute_capped` | early 段 RMSE，绝对分段（RUL 阈值，见 `seg_boundaries`），capped 口径 |
| 45 | `seg_early_absolute_raw` | early 段 RMSE，绝对分段（RUL 阈值，见 `seg_boundaries`），raw 口径 |
| 46 | `seg_mid_relative_capped` | mid 段 RMSE，相对分段（按该机**完整寿命**的三分位），capped 口径 |
| 47 | `seg_mid_relative_raw` | mid 段 RMSE，相对分段（按该机**完整寿命**的三分位），raw 口径 |
| 48 | `seg_mid_absolute_capped` | mid 段 RMSE，绝对分段（RUL 阈值，见 `seg_boundaries`），capped 口径 |
| 49 | `seg_mid_absolute_raw` | mid 段 RMSE，绝对分段（RUL 阈值，见 `seg_boundaries`），raw 口径 |
| 50 | `seg_late_relative_capped` | late 段 RMSE，相对分段（按该机**完整寿命**的三分位），capped 口径 |
| 51 | `seg_late_relative_raw` | late 段 RMSE，相对分段（按该机**完整寿命**的三分位），raw 口径 |
| 52 | `seg_late_absolute_capped` | late 段 RMSE，绝对分段（RUL 阈值，见 `seg_boundaries`），capped 口径 |
| 53 | `seg_late_absolute_raw` | late 段 RMSE，绝对分段（RUL 阈值，见 `seg_boundaries`），raw 口径 |
| 54 | `seg_boundaries` | 两套边界同时装：`{"relative": [0.3333, 0.6667], "absolute": [75, 25]}`。**先冻结再跑**；`absolute` 的 (75,25) 是待 DS02 真实 RUL 尺度确认后可改**一次**的默认值，不许看完结果调 |
| 55 | `freeze_id` | 授权本次测试评估的冻结清单 id；`phase=dev` 留空 |
| 56 | `freeze_ts_utc` | 该冻结清单的 `freeze_ts_utc`（从清单抄，不自己写） |
| 57 | `eval_start_utc` | 本次测试评估**实际开始**时刻；必须 > `freeze_ts_utc` |
| 58 | `status` | `running` / `ok` / `failed` / `smoke` / `superseded` |
| 59 | `audit_verdict` | `pass` / `fail` / `n/a`（**只放判断，不放人名**） |
| 60 | `independent_unit` | 固定 `engine`（见 §1.5） |
| 61 | `registry_note` | 异常、偏差、重跑原因、溯源信息都写这里 |
| 62 | `audited_by` | **只放人名**。不确定性和判断不许写进这一列 |

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
| `freeze_commit` | **`e7ad15f`** |
| `protocol_blob` | **`443462ca03087d5aff2058778de2d47b3331728c`** |
| 核验命令 | `git rev-parse e7ad15f:docs/paper_b_ncmapss/DATASET_PROTOCOL.md` |

> **已作废的旧锚（按时间顺序）**：
> `5b228ad` / `6c184bbf…` → `7f01909` / `4458d9ea…` → `922a2d1` / `ebfbfcb8…` → **`e7ad15f` / `35b40d8b…`（当前）**。
> 前两次前移：主指标标注、相对分段分母（真 bug）、HI 范围收窄。
> 第三次：训练预算按实测数据量改（40 epochs 在 DS02 上只有 40 个梯度步）、capped 实测失效。
> 第四次：**对齐量从「优化步数」改为「过数据遍数」**——1000 步在单折 380 窗口、batch=380 下
> 等于 1000 遍，是 C-MAPSS 200 遍的 25 倍，会让四档全部评在过拟合最深点。
> **`e7ad15f` 之后协议冻结，直到第一批 LOEO 开发集结果出来为止不再改 §4/§6**；再改就是真调参。


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

实测：dev 最长 **89** 个飞行周期（unit 5），test 最长 **76**。全部 < 125，因此
`y_true_cap125 == y_true_raw` **处处成立**，capped 不是第二个独立口径。

| 事项 | 规则 |
|---|---|
| `pred.csv` 两列 | **照写**（审计要求保留原始与封顶两个标签，用户 §六） |
| 结果表 | capped 标 `inactive (no unit exceeds the cap)`，**不与 raw 并排印表** |
| 12 个 `seg_*_capped` 列 | 字段结构不变，照常登记；**报表时标 inactive** |
| 措辞 | **不得称 capped 为「更严格的口径」**，也不得把 capped 与 raw 的差异算作模型属性——那是标签定义的属性 |

> 这条正好是 C-MAPSS 那晚的反面：那边 cap 生效、capped/raw 差一倍；这边 cap 不生效、两者恒等。
> 所以「哪个口径更严」不是先验结论，必须按数据集实测，写进台账而不是靠记忆。

---

### 3.6 训练预算：对齐「遍数」不是「步数」（v7）

决定泛化的是**训练集被过多少遍**，不是优化器更新了多少次。DS02 单折只有 380 个窗口，
batch 取 380 时一个 step 就是完整一遍，所以「1000 步」实际是 1000 遍。

| | C-MAPSS | DS02（v7） |
|---|---|---|
| 过数据遍数（epochs） | 40 | **40** |
| batch | 512 | **`min(64, N_train)` = 64** |
| 每遍更新数 | ~39 | **6** |
| 总更新数 | ~1560 | **240** |

- 台账记 `epochs_run = 200`、`selection_rule = fixed_passes:200`、`batch_effective = 64`（写在 `registry_note`）。
- **`protocol.json` 的 `epochs: 40` 在这里恰好仍是 40**，但语义从「40 轮」变成「200 遍」，
  且 `batch` 被覆盖为 64——所以仍要记 `epochs_overridden` / `batch_overridden`，不能因为数字碰巧相同就不记。

**诊断字段（只诊断，不参与选择）：**

`metrics.json` 里必须同时给 `holdout_min` 与 `holdout_min_epoch`，报表每行除 final-epoch 的数之外，
一并报这两个值，并标注 `diagnostic only - not used for model selection`。

理由：不做早停、报分取最后一轮，**固定预算只有在曲线也藏起来的时候才藏得住过拟合**。
报了最小值就能看出固定终点是停在斜坡上还是平台上，四档的曲线形状对比比单点更有信息量。
台账**不加列**（开跑在即，加列会让写入端错位），只在 `registry_note` 记一句「holdout_min 见 metrics.json」。

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

19. **训练预算按 200 遍**：从该 run 的 `config.json` 取 `N_train`（训练窗口数），核对
    `batch == min(64, N_train)` 且 `epochs_run == 200`，且 `N_train/batch` 向上取整 = 6 次/遍。
    不等于 → `fail`（跑错预算）。**故意不加 `n_train_windows` 列**：开跑在即，加列会让写入端与台账错位。

## 9. 开跑前自检（每批 run 之前跑一遍，任何一条不过就不许开跑）

```bash
cd <repo>
# 1. 协议没漂移：冻结提交、当前 HEAD、台账三处 blob 必须相同
git rev-parse e7ad15f:docs/paper_b_ncmapss/DATASET_PROTOCOL.md   # 期望 443462ca03087d5aff2058778de2d47b3331728c
git rev-parse HEAD:docs/paper_b_ncmapss/DATASET_PROTOCOL.md      # 必须与上一行相同
grep -c 443462ca03087d5aff2058778de2d47b3331728c docs/paper_b_ncmapss/registry/runs.csv  # 每行都要命中

# 2. 工作区干净（3090 是只读消费者）
git fetch && git reset --hard origin/ncmapss-ds02
git status --porcelain            # 必须为空

# 3. 冻结文件在且是冻结态
test -f paper_b/configs/protocol.json
test -f paper_b/results/FREEZE_ABLATION_SEED101.json
python -c "import json;d=json.load(open('paper_b/results/FREEZE_ABLATION_SEED101.json'));assert d['frozen'] is True"

# 4. 台账锁没有残留
test ! -e docs/paper_b_ncmapss/registry/runs.csv.lock
```

**第 1 条是最便宜、也最容易漏的一条**：一条 `git rev-parse` 就能证明协议正文在冻结后没被动过
（2026-11-12 就用它证明过 `ed8f0cd` 只动了复现审计、`DATASET_PROTOCOL.md` 的 blob 未变）。
开跑后再发现协议漂移，那一批 run 全部作废。

第 2 条对应工作区拓扑：**`origin/ncmapss-ds02` 是唯一真源，3090 是只读消费者**，
每批跑之前 `reset --hard`，不在 3090 上提交、不留未跟踪的代码文件。

