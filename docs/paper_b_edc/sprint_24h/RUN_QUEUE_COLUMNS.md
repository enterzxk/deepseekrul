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
| `provenance` | derived | **两个问题，两个答案**：`commit` = 该代码状态能追到某个 commit；`tree_only` = 追不到（状态在仓库里不存在）；`no_fingerprint` = 该 `metrics.json` 早于指纹字段。**与 `gate_status` 正交，不能互相顶替** |
| `scan_utc` `scan_host` `scan_id` | scan | **文件自己回答「我是哪一轮、在哪台机器上生成的」**。`scan_id` = 整批扫描内容的 sha256 前 12 位：内容相同则 id 相同，任何一格变了 id 就变 |

## 为什么表里要带 `scan_utc` / `scan_id`

`mtime` 在文件系统里，**不在文件里**。一旦这份 CSV 被复制、被 `scp`、被 `git checkout`，`mtime` 就变成复制时间，
而「这批数是哪一轮扫的」这条线索**永久消失**。

今晚已经出现过一次「混合产物」：`mech_*.json` 三新三旧交错，每个文件单独看都正常，只有整批看才露馅——
当时靠的是 `mtime`，而 `mtime` 正是复制就会丢的那种证据。

所以把生成信息写进文件本身：**同一批扫描的 93 行共享同一个 `scan_id`；如果哪天有人手工改了一格，
`scan_id` 对不上重算值，整表立刻可知不是同一次扫描的产物。**

## `gate_status` 与 `provenance` 是两个问题

今晚最容易混的一句话是「这格是 A 级，所以它可追溯」。**两者正交：**

```
gate_status   A  = 同一代码状态再跑一遍，四项逐位相同   → 这一跑是可复现的
provenance    commit = 那个代码状态在 git 里有对应提交  → 这一跑可追到源码
```

**台账里同时存在「A 级 + tree_only」的格**（`M6-FD001-s101`、`B3-FD004-s101`）：
门格复现了它们，证明代码状态是确定性的；但那个代码状态**从未提交**，所以追不到 commit。

**这两个字段不能互相顶替**，因为：

- 只有 `gate_status`：会得出「可复现 ⇒ 可追溯」的假结论；
- 只有 `provenance`：会得出「追不到 commit ⇒ 不可信」的假结论——**而实测它是可复现的**。

### 那 10 行 `tree_only` 的来源

10 行全部落在 `12:14–13:53Z`，分成两个指纹：

```
1883b283…   2 行   12:14–12:25Z
2d185f6a…   8 行   13:24–13:53Z
```

这个窗口正好对上两个**就地打补丁**的脚本（本地时间 20:07 / 20:46 = UTC 12:07 / 12:46），
而且**两个脚本改的都是 `paper_b/hc/train_d.py`——它正是 9 文件指纹里的一个**：

- `_patch_m6.py`（12:07Z）→ 之后 12:14–12:25Z 的那 2 行；
- `patch_m3c_rank_diag.py`（12:46Z）→ 之后 13:24–13:53Z 的那 8 行。

**这也解释了为什么「162 个 commit × 单文件替换」全都对不上**：那些代码状态**从来没有进过 git**，
它们只存在于当时 3090 的工作树里。所以这 10 行标 `tree_only` 是**准确描述**，不是「待查」。

## 两个必须随表引用的口径

1. **`preds_md5` 是 sha256**。列名沿用旧名以免打断管线，值不是 md5。引用哈希时写 `preds_sha256`。
2. **`usable=no` 的 17 个门格不是样本**。它们与原始格 `preds`/`metrics`/`ckpt` 逐字节相同，只作复现证据。

## 样本说明（论文可直接用）

> 台账里有若干门格（`_gate`），与原始格逐字节相同，**只作复现证据，不计入样本量**。
> 因此 FD001 上 B0 是 **6 个 seed，不是 12 个**。
> **门格数增长时样本量不动**——这就是「门格不计入样本」的操作化定义：**如果 `usable` 哪天跟着门格一起涨，说明去重坏了。**

**目录数、可用格数、A/B 格数一律不写在这里。** 它们随每批门格落盘而变，写进正文必然过期——**每跑一次门格就要手改一遍，而手改一定会漏一处。**

> **唯一真相源**：`FINAL_24H_DECISION.md` 的 §0.7 生成块，由 `python check_decision_numbers.py --emit-tier` 写入，带 `SEED_RESULTS.csv` 的 sha256 与行数。**块对不上 CSV 时 checker 硬失败**，所以那块里的数不可能旧。
>
> **计数键是 `run_dir` 不是 `run_id`**：`run_id` 跨根重复（同一 `(arm, subset, seed)` 会落在 `_gate`/`_verify`/`d`/`d80` 等多个根），按 `run_id` 计数会把不同轮次静默合并成一格。

## 证据面（谁验证过什么）

| 结论 | 可用证据等级 |
|---|---|
| 有门格且四项全同的格 | **A**：同 HEAD，`preds` / `val` / `epochs_run` / `best_ep` 四项一致 |
| 同子集、但该格无门格 | **B**：未验证代码版本，引用必须带限定 |
| `FD002` / `FD003` | 到收口时仍无一格有门格 → 全部 **B** |

**本表只定义「A 是什么、B 是什么」，不写计数。** 逐子集的 A/B 格数见 §0.7 生成块。

**`FD004` 从「全 B」开始变 A，本身证明「全 B 是时间约束、不是数据性质」**——门格一跑就升，数值一个不动。
