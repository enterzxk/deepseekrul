# Paper C run 台账 schema（`docs/paper_c/registry/`）

状态：**v2**，2026-11-22。维护人：实验结果管理员（单一写入者）。
v1 → v2 的变更：60 格（两 arm）、A/B 两段、主臂 `k0p`、`gate/alpha` 轨迹为强制产物、
去重键补 `base_model`（v1 是错的）、退出码三档。变更记录见 `REGISTRY_NOTES.md` §0。

```
runs.csv           台账本体（append-only，52 列）
run_queue.csv      允许启动的格（由 grid.json 生成，60 行）
grid.json          网格定义（改这里，不要手改 queue）
schema_cols.py     列顺序的唯一来源（gen_queue 与 checker 都 import 它）
gen_queue.py       生成 queue（并在 runs.csv 无数据行时重写表头）
check_registry.py  检查器（12 项断言，退出码三档）
selftest_registry.py  检查器的 21 个正负用例（可重跑，不是口头声明）
FIXED40_ANCHORS.csv / .md   30 格冻结锚点（epochs_run/best_epoch/patience/window/lr/wd）
FREEZE_20261122.json        预注册 / wrapper / manifest 的 sha256（冻结第 ③ 步）
REGISTRY_NOTES.md  规则与变更记录
```

---

## 1. 列（52）

| # | 列 | 含义 |
|---:|---|---|
| 1 | `run_id` | 见 §2 |
| 2 | `wave` | 波次，本批 `W1` |
| 3 | `stage` | `P2`；只有 `Terminal` 允许解锁 TEST |
| 4 | `segment` | `A`（V0 复现）/ `B`（+CADF）；与 `arm` 强制一致 |
| 5 | `arm` | `v0` / `cadf` |
| 6 | `base_model` | `aidgn` / `two_phase` / `dual_mixer` / `ffve` / `fcstgnn` |
| 7 | `cadf_variant` | `k0p`（本批）；`v0` 行固定 `none` |
| 8 | `cadf_k0_mode` | `prop` / `abs` / `none` |
| 9 | `cadf_k0_value` | **这一列就是 `k0_resolved`**：`k0p` 时为 `ceil(0.2*span)` |
| 10 | `subset` | `FD001` / `FD004`（本批只这两个） |
| 11 | `seed` | `101` / `202` / `303` |
| 12 | `span` | 窗口长度，**按 (模型, 子集)**：aidgn 50、two_phase 45、dual_mixer 30、ffve 30、fcstgnn FD001 30 / FD004 50 |
| 13 | `split_protocol` | `80/20 engine-level, split_seed=7` |
| 14 | `split_seed` | `7` |
| 15 | `gate` | **只记实验类型**（`P2_cross_arch`） |
| 16 | `in_gate_table` | **能否进 Gate 表**（独立布尔列） |
| 17 | `hypothesis` | 该格要检验的那一句话 |
| 18 | `host` | `lab3090` / `lab-a40` |
| 19 | `git_sha` | 实际跑过的代码提交（**不是清单里的**） |
| 20 | `code_md5` | 五个训练相关文件的 md5，JSON 字符串 |
| 21 | `config_sha256` | 该 run 的 `config.json` 哈希 |
| 22–23 | `launched_utc` / `closed_utc` | 起止时刻（UTC） |
| 24 | `status` | `ok` / `failed` / `smoke` / `running` / `superseded` |
| 25–26 | `epochs_run` / `best_epoch` | **A 段对账的第一道信号**（见 §6） |
| 27 | `budget_truncated` | `true` = 打满 40 轮上限（不是收敛） |
| 28–33 | `val_engine_mean_rmse` / `val_pooled_rmse` / `val_mae` / `val_bias` / `eol95_mae` / `val_nasa` | **验证集**指标，engine-level 为口径 |
| 34–35 | `test_rmse` / `test_score` | **封存**；只有 `stage=Terminal` 才允许填 |
| 36–38 | `ckpt_path` / `log_path` / `preds_path` | 产物路径 |
| 39 | `gate_alpha_trace` | 逐 epoch `‖gate.weight‖`/`‖gate.bias‖`/逐通道 `|alpha|` 轨迹（`arm=cadf` 必填） |
| 40 | `gate_open_epoch` | 首次 `mean(gate) > 0.01` 的 epoch；从未打开写 `none` |
| 41–42 | `gate_norm_last` / `alpha_last` | 末轮 `‖gate‖` / `|alpha|`（判断「没醒」的标量） |
| 43 | `baseline_source` | `frozen_ledger`（A）/ `local_v0`（B） |
| 44 | `baseline_ref` | A：冻结台账 `physical_id`；B：**本批 A 段那条 `run_id`** |
| 45 | `baseline_val_engine_mean_rmse` | 基线同口径值（**读**，不手抄） |
| 46 | `delta_val_rmse` | **= 基线 − 本格**，见 §5 |
| 47 | `paired_win_rate` | 逐机配对胜率；`arm=v0` 必须为空 |
| 48 | `registry_note` | 备注：配置差异、异常、重跑原因 |
| 49–50 | `audited_by` / `audit_verdict` | 审计人（只放人名）/ `pass`·`fail`·`n/a` |
| 51 | `determinism_only` | 布尔；确定性重复行标 `true`，**不得写进 verdict** |
| 52 | `baseline_version` | `open_source_matched_20261112@d7bd9c17` |

**枚举之外一律拒绝**：`status` 五个、`audit_verdict` 三个、三个布尔列只接受 `true`/`false`、
`arm` 两个、`segment` 两个。**不存在 `unclosed`**——进程在就是 `running`，进程没了又没有 close
记录就是 `failed`。

---

## 2. run_id（两条格式，按 arm 分，不混用）

```
A 段  ^pc-v0-(aidgn|two_phase|dual_mixer|ffve|fcstgnn)-(FD001|FD004)-s(101|202|303)-W1(r\d+)?$
B 段  ^pc-cadf-(k0p)-(aidgn|two_phase|dual_mixer|ffve|fcstgnn)-(FD001|FD004)-s(101|202|303)-W1(r\d+)?$
```

`r\d+` 是**同波次重试**（`W1r1`），不是新格。两条格式是**有意的**：`v0` 行没有 CADF 变体，
硬塞一个 `none` 段只会让 id 变长而不带信息。

---

## 3. 去重键与波次

- **去重键 = (`baseline_version`, `arm`, `base_model`, `cadf_variant`, `subset`, `seed`)**。
  同键只允许一条 `ok`。
  > v1 的键漏了 `base_model`——30 格一落盘，同一 (子集, seed) 的 5 个方法会互相判成重复。
  > 它没在 v1 炸，只因为当时 `runs.csv` 是空的。自测里有一条用例专门守这个。
- `failed` 的重试**不占位**；`superseded` 是更正行，允许与 `ok` 同键共存。
- 进 Gate 表要过波次过滤：`run_id` 匹配 `-W1(r\d+)?$` 且 `status=ok`，同格取**最早的 ok**。

---

## 4. 基线：两段用两个来源，**不得互换**

| 段 | `baseline_source` | 基线 | 为什么 |
|---|---|---|---|
| **A** | `frozen_ledger` | `paper_b/experiments/open_source_matched_20261112/physical_runs.csv` @ `d7bd9c17`，`source_condition=fixed40` | A 段就是**复现**那 30 格，所以对它 |
| **B** | `local_v0` | **本批 A 段同一 (模型, 子集, seed) 的 `run_id`** | 跨机器漂移与效应同量级；A40 的数不能当 3090 的基线 |

`author_limit`（另有 10 行）**不得配对**——拿它配 +CADF，delta 是两个变量的差。

**B 段指回冻结台账 = 检查器直接拒**（§6 第 6 项）。这不是风格问题：那样配出来的 Δ 里混着
「A40 vs 3090」的漂移，而 Gate 1 判的效应就是零点几个 RMSE。

---

## 5. 符号与口径（写死，避免符号反）

```
delta_val_rmse = baseline_val_engine_mean_rmse - val_engine_mean_rmse
```

**正 = 本格更好**（RMSE 更低）。检查器逐行验证恒等式，差 > 1e-9 即拒。

**同一个列在两段里含义不同，必须分开读**：

- **A 段**：`delta` 是**对账量**，不是模块效应。它回答「3090 复现 A40 偏了多少」。
- **B 段**：`delta` 是**模块效应**（+CADF 对同机同环境同种子的 V0）。

指标口径：

- **一律 engine-level**：先在每台机上算，再跨机等权平均。
- **`metrics.json` 里的 `val_engine_rmse` 是 pooled，禁用**（Paper B 已定性；Paper C 沿用）。
- **EOL(0.95)**：每台机在 95% 寿命截断点的绝对误差，跨机平均 → `eol95_mae`。
  不退化 = `Δ_EOL_MAE ≥ −0.05`（行里存原始 Δ，判定在 Gate 脚本里做）。
- 方向性结论必须**逐格测噪声**：单格 Δ 小于该格尾段波动区间时不得作为方向证据。

---

## 6. 检查器（12 项，退出码三档）

```bash
python docs/paper_c/registry/check_registry.py     # 0 PASS / 1 FAIL / 2 ABORT
python docs/paper_c/registry/selftest_registry.py  # 21 个用例，必须全 OK
```

| # | 断言 | 退出码 |
|---:|---|---|
| 1 | `runs.csv` 表头 == `schema_cols.COLS` | 2（读不了就不读） |
| 2 | 枚举域 + `segment`/`arm` 一致（A=v0, B=cadf） | 1 |
| 3 | `run_id` 匹配**自己 arm 的**格式 | 1 |
| 4 | 去重键（含 `base_model`）：同键一条 `ok` | 1 |
| 5 | `ok` 行必填；两 arm 附加项不同（B 缺轨迹即拒；gate 从未打开却进 Gate 表即拒） | 1 |
| 6 | 基线解析：A→冻结台账（`fixed40`）；B→**本批 A 段那一格** | 1 |
| 7 | `delta == baseline − 本格` | 1 |
| 8 | TEST 封存（`stage != Terminal` 时 `test_*` 必空） | 1 |
| 9 | `ok` 行必须在队列里，且队列 `launch_allowed=true` | 1 |
| 10 | `span`/`epochs_run`/`best_epoch`/`k0` 与 grid 一致 | 1 |
| 11 | **B 段闸门**：有 B 结果时，A 段 30 格齐 + 非截断格 `|Δ| ≤ 0.2` | **2** |
| 12 | 冻结件哈希未变（预注册 / wrapper / manifest） | **2** |

**第 11 项的例外按预注册 R2**：`epochs_run == 40` 的 5 格（预算切断）偏离**不单独构成 ABORT**，
须结合未截断的 25 格判断。检查器从 `FIXED40_ANCHORS.csv` **读出**这 5 格，不硬编码名字——
锚点重生成时例外集自动跟着变。

**第 11 项只做聚合那一半**。`epochs_run`/`best_epoch` 逐格比与逐机差（mean/sd/同号比例/P95）
是军师 `gate1.py` 的对账脚本，**两边不是同一道闸门，不要以为跑了检查器就等于对完账**。

**第 12 项的意义**：预注册一旦冻结，任何改动都会让检查器 exit 2，直到重记哈希并在
`REGISTRY_NOTES.md` 说明改了什么。这条防的是「跑完再改判据」。

---

## 7. 写入边界

本管理员写 `docs/paper_c/registry/` 下的全部文件。
**不写** `paper_c/`（牢莫的代码）、**不写** `results/`（runner 的产物）、
**不改** `paper_b/` 的基线台账。`launch/`、`close/` 由 runner 写，单向拉回。

唯一一次他人写入：`68551059` 由牢莫提交了 `frozen_v0_fixed40.json`（冻结 V0 证据导出）。
保留为只读参照物；本管理员不再改它，对账锚点的权威件是 `FIXED40_ANCHORS.csv`。
