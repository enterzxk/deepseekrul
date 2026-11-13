# RUN_QUEUE.csv —— 列级 provenance（每列这一轮验过没有）

> 「这张表可信」是一句没法验证的话；「这一列本轮验过」是可以验的。
> 同一张表里不同列的可靠性本来就不一样，所以列要分开标。

生成方式：一次扫描（`tmp/audit_hc/hash_scan.py`，在 3090 上跑）→ 一次组装（`merge_queue_final.py`）。
**没有任何一列是继承上一版的。** 第一版曾有 6 列继承，本轮已全部改为同一次扫描或直接派生。

| 列 | provenance | 说明 |
|---|---|---|
| `run_id` | scan | 目录名 |
| `root` | scan | 目录相对路径（`d` / `d80` / `d/probe80` / `d40diag` / `d80b160` / `d80diag` / `_gate` / `_gate160` / `_dup_d80m5` / `_verify`） |
| `arm` `subset` `seed` | metrics.json | 直读 |
| `budget_epochs` | metrics.json | `budget.epochs` |
| `budget_override` | metrics.json | 直读，原样保留 |
| `override_kind` | derived | `smoke`（epochs_run ≤ 1）/ `audit`（codecheck*/audit*）/ `diag_budget` / `retrain` |
| `archived` | derived | root 前缀 `_dup` / `_verify` |
| `is_gate_cell` | derived | root 前缀 `_gate` |
| `status` | literal | 每行都有 `metrics.json`，故为 `done`；「没跑」的行不在本表 |
| `usable` | derived | smoke / audit / archived / byte-duplicate → `no` |
| `epochs_run` `best_ep` `K` | metrics.json | 直读 |
| `early_stopped` | derived | `epochs_run − best_ep == patience(12)` 且 `epochs_run < budget` |
| `val_engine_mean_rmse` | **metrics.json 直读** | 论文唯一引用口径 |
| `val_pooled_rmse` | **metrics.json 直读** | 参考，不与 engine-mean 混用 |
| `rmse_uncensored` | **metrics.json 直读** | 参考，不受封顶边界影响 |
| `preds_md5` | **recomputed** | 名字为兼容 `aggregate_metrics.py` 保留，实际是 **sha256** |
| `preds_sha256` `metrics_sha256` `history_sha256` `ckpt_sha256` | **recomputed** | 本轮在 3090 上重算；`best.pt` 也有哈希，所以「数值同、权重不同」查得出来 |
| `preds_rows` | scan | 数 `preds.csv` 行数 |
| `has_ckpt` | derived | `ckpt_sha256` 非空 |
| `duplicate_of` | derived | 字节相同的另一格（`preds_sha256` 相等） |
| `gate_status` | derived | `verified`（四项与门格全同）/ `gate_reproduction`（门格本身）/ `duplicate_bytes` / `unverifiable`（该格无门格）/ `mismatch` |
| `files_sha256` | metrics.json | `code_fingerprint.files_sha256`，覆盖 `arrival/hi_d/data_d/train_d/config_hc` |
| `frozen_sha256` | metrics.json | 冻结权重文件哈希 |
| `code_lock_version` `code_lock_git_sha` `code_lock_md5` | lock 文件 | 每行的代码版本归属；20 个 blob 的逐文件 md5 在 `registry/code_lock.json`，本表指向它、不复制 |
| `created_utc` `elapsed_s` | metrics.json | 直读 |
| `run_dir` `preds_path` | derived | 由 `root` + `run_id` 拼 |
| `note` | derived | 判定理由，人写 |

## 两个必须随表引用的口径

1. **`preds_md5` 是 sha256**。列名沿用旧名以免打断管线，值不是 md5。引用哈希时写 `preds_sha256`。
2. **`usable=no` 的 17 个门格不是样本**。它们与原始格 `preds`/`metrics`/`ckpt` 逐字节相同，只作复现证据。

## 样本说明（论文可直接用）

> 台账 93 个目录里有 17 个是门格（`_gate`），与原始格逐字节相同，只作复现证据，不计入样本量。
> 因此 FD001 上 B0 是 **6 个 seed，不是 12 个**。

## 证据面（谁验证过什么）

| 结论 | 可用证据等级 |
|---|---|
| FD001 的 17 格 | **A**：同 HEAD 门格四项全同（`preds`/`val`/`epochs_run`/`best_ep`） |
| FD004 的 16 格 | **B**：无一格有门格 |
| FD002 / FD003 | **B**：无一格有门格 |

**A 级证据只存在于 FD001 的 17 格**，其余全部为 B 级，引用时必须带限定。
