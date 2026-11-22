# Paper C run 台账 schema（`docs/paper_c/registry/`）

状态：v1，2026-11-22。维护人：实验结果管理员（单一写入者）。
范围：**只登记 +CADF 的新跑**（30 格）。V0 基线不重跑、不重登，用 `baseline_ref` 指向已发布的基线台账。

```
runs.csv        台账本体（append-only，45 列）
run_queue.csv   允许启动的格（由 grid.json 生成）
grid.json       网格定义（改这里，不要手改 queue）
gen_queue.py    生成 queue
check_registry.py  检查器（退出码非 0 拒绝提交）
REGISTRY_NOTES.md  规则与变更记录
```

---

## 1. 列（45）

| # | 列 | 含义 |
|---:|---|---|
| 1 | `run_id` | 见 §2 |
| 2 | `wave` | 波次，本批 `W1` |
| 3 | `stage` | `P2`；只有 `Terminal` 允许解锁 TEST |
| 4 | `base_model` | `aidgn` / `two_phase` / `dual_mixer` / `ffve` / `fcstgnn` |
| 5 | `cadf_variant` | variant tag：`k0a` / `k0p` / `eng`（见 grid.json） |
| 6 | `cadf_k0_mode` | `abs` / `prop` / `none` |
| 7 | `cadf_k0_value` | K0 实际取值（绝对周期数或 ceil(0.2L) 的结果） |
| 8 | `subset` | `FD001` / `FD004`（本批只这两个） |
| 9 | `seed` | `101` / `202` / `303` |
| 10 | `span` | 窗口长度，本批 50 |
| 11 | `split_protocol` | `80/20 engine-level, split_seed=7` |
| 12 | `split_seed` | `7` |
| 13 | `gate` | **只记实验类型**（`P2_cross_arch`） |
| 14 | `in_gate_table` | **能否进 Gate 表**（独立布尔列） |
| 15 | `hypothesis` | 该格要检验的那一句话 |
| 16 | `host` | `lab3090` / `lab-a40` |
| 17 | `git_sha` | 实际跑过的代码提交（**不是清单里的**） |
| 18 | `code_md5` | 五个训练相关文件的 md5，JSON 字符串 |
| 19 | `config_sha256` | 该 run 的 `config.json` 哈希 |
| 20–21 | `launched_utc` / `closed_utc` | 起止时刻（UTC） |
| 22 | `status` | `ok` / `failed` / `smoke` / `running` / `superseded` |
| 23–24 | `epochs_run` / `best_epoch` | 打满预算还是早停 |
| 25 | `budget_truncated` | `true` 表示打满预算（不是收敛） |
| 26–31 | `val_engine_mean_rmse` / `val_pooled_rmse` / `val_mae` / `val_bias` / `eol95_mae` / `val_nasa` | **验证集**指标，engine-level 为口径 |
| 32–33 | `test_rmse` / `test_score` | **封存**；只有 `stage=Terminal` 才允许填 |
| 34–36 | `ckpt_path` / `log_path` / `preds_path` | 产物路径 |
| 37 | `baseline_ref` | 基线行的 `physical_id`，见 §4 |
| 38 | `baseline_val_engine_mean_rmse` | 基线同口径值（从基线台账取，不手抄） |
| 39 | `delta_val_rmse` | **= 基线 − 本格**（正 = CADF 更好），见 §5 |
| 40 | `paired_win_rate` | 逐机配对胜率（CADF 胜出的机数 / 总机数） |
| 41 | `registry_note` | 备注：配置差异、异常、重跑原因 |
| 42–43 | `audited_by` / `audit_verdict` | 审计人（只放人名）/ `pass`·`fail`·`n/a` |
| 44 | `determinism_only` | 布尔；确定性重复行标 `true`，**不得写进 verdict** |
| 45 | `baseline_version` | `open_source_matched_20261112@d7bd9c17` |

**枚举之外的值一律拒绝**：`status` 只有五个、`audit_verdict` 只有三个、
三个布尔列只接受 `true`/`false`。**不存在 `unclosed`**——进程在就是 `running`，
进程没了又没有 close 记录就是 `failed`。

---

## 2. run_id

```
^pc-cadf-(k0a|k0p|eng)-(aidgn|two_phase|dual_mixer|ffve|fcstgnn)-(FD001|FD004)-s(101|202|303)-W1(r\d+)?$
```

`r\d+` 是**同波次重试**（`W1r1`），不是新格。换 variant 用新的 variant tag，不靠 `r` 后缀。

---

## 3. 去重键与波次

- **去重键 = (`baseline_version`, `cadf_variant`, `subset`, `seed`)**。同键只允许一条 `ok`。
- `failed` 的重试**不占位**；`superseded` 是更正行，允许与 `ok` 同键共存。
- **进 Gate 表要过波次过滤**：`run_id` 匹配 `-W1(r\d+)?$` 且 `status=ok`，
  同格取**最早的 ok**。只用格键会让历史 preliminary run 顶掉本波证据。

---

## 4. 基线必须钉在 `fixed40`

基线台账 `paper_b/experiments/open_source_matched_20261112/physical_runs.csv`
（分支 `codex/paper-b-open-source-reproduction`，`d7bd9c17`）里有**两种** `source_condition`：

| condition | 覆盖 | 能否配对 |
|---|---|---|
| `fixed40` | 5 方法 × FD001/FD004 × 3 seed = **30 格，正好一格对一格** | **只能用它** |
| `author_limit` | 另有 10 行（其中 5 行落在 FD001/FD004） | **不得配对** |

`author_limit` 是作者训练预算下的另一条口径。拿它去配 +CADF 的 run，
得到的 delta 是两个变量的差，不是模块的贡献。检查器会拒（`source_condition != fixed40`）。

---

## 5. 符号与口径（写死，避免符号反）

```
delta_val_rmse = baseline_val_engine_mean_rmse - cadf_val_engine_mean_rmse
```

**正 = CADF 更好**（RMSE 更低）。检查器逐行验证这个恒等式，差 > 1e-9 即拒。

指标口径：

- **一律 engine-level**：先在每台机上算，再跨机等权平均。
- **`metrics.json` 里的 `val_engine_rmse` 是 pooled，禁用**（Paper B 已定性；Paper C 沿用同一禁令）。
- **EOL(0.95)**：每台机在 95% 寿命截断点上的绝对误差，跨机平均 → 即 `eol95_mae`。
  经理裁定「不退化」= **Δ ≤ +0.05**（同符号约定：Δ 正 = CADF 更好，所以不退化是 Δ ≥ −0.05；
  行里存的是原始 Δ，判定在 Gate 脚本里做）。
- 方向性结论必须**逐格测噪声**：单格 Δ 小于该格尾段波动区间时不得作为方向证据。

---

## 6. 检查器

```bash
python docs/paper_c/registry/check_registry.py
```

九项断言（表头、枚举、run_id、去重、ok 行必填、基线解析、delta 恒等式、TEST 封存、queue 覆盖）。
**退出码非 0 拒绝提交。** 检查器本身用 9 个正负用例验证过会真的拦（见 REGISTRY_NOTES §3）。

---

## 7. 写入边界

本管理员写 `docs/paper_c/registry/` 下的全部文件。
**不写** `paper_c/`（牢莫的代码）、**不写** `results/`（runner 的产物）、
**不改** `paper_b/` 的基线台账。`launch/`、`close/` 由 runner 写，单向拉回。
