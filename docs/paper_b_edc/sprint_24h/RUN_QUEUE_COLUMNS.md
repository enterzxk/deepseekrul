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
| `provenance` | derived | **三个答案，不是两个**：`commit` = 该代码状态就是一个提交；`commit+patch` = 是「某个提交 + 一个**已提交的**幂等补丁脚本」重放出来的；`tree_only` = 追不到；`no_fingerprint` = `metrics.json` 早于指纹字段。**与 `gate_status` 正交** |
| `provenance_detail` | derived | 来自 `CODE_FINGERPRINT_TO_COMMIT.csv` 的 `note`：`commit+patch` 给出提交号与补丁路径，`tree_only` 给出「为什么追不到」 |
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

### 那 10 行的来源：2 行救回，8 行确认追不到

10 行全部落在 `12:14–13:53Z`，分成两个指纹：

```
1883b283…   2 行   12:14–12:25Z   →  commit+patch（已救回）
2d185f6a…   8 行   13:24–13:53Z   →  tree_only（确认追不到）
```

窗口对上两个**就地打补丁**的脚本（本地 20:07 / 20:46 = UTC 12:07 / 12:46），
两个脚本改的都是 `paper_b/hc/train_d.py`——**它正是 9 文件指纹里的一个**。

**为什么一开始「162 个 commit × 单文件替换」全都查不到**：那批代码状态**从来没有变成过 blob**，
所以按 blob 穷举原理上就查不到——**不是查得不够，是查错了空间**。

**为什么 2 行能救回来**：补丁脚本是**确定性的文本替换**，可以直接重放。
`1883b283` = `commit c3f27a52` + 重放 `_patch_m6.py`（另外五个 commit 的 9 文件状态相同，都能复现）。
**前提是补丁脚本本身被提交了**——今晚这两个都在仓库里，所以救得回来。

**为什么 8 行救不回来**：`patch_m3c_rank_diag.py` 自己的 docstring 写着，
3090 上那份 `train_d.py` 是 **1153 行**，而同一 commit 的仓库版是 **1070 行**——
**它带着一堆没有脚本的人手改动**。重放两个已提交的补丁也复现不出来，所以这 8 行是**真正的 `tree_only`**。

**这条纪律可写成**：**幂等补丁 + 重放 = 可恢复的溯源。**「忘了提交」从**不可恢复**降成**可恢复但成本高**，
而**代价的那一半取决于补丁有没有被提交**——补丁只在远端敲一遍、没进仓库，就真的没了。

**`tree_only` 现在的含义是「知道为什么不知道」**，不是「还没查」。

## 门格判据：根名是线索，配对才是判决

`_gate*` 只是一个目录名。**一格是不是门格，取决于它有没有逐字节匹配到一个非门格的双胞胎**，
不取决于它住在哪个目录下。

```
gate_status
  verified              原始格，四项与门格逐位相同（preds sha256 / val / epochs_run / best_ep）
  gate_reproduction     门格本身，与原始格逐字节相同 → 不计入样本
  gate_root_new_sample  住在 _gate* 根下，但复现不了任何正式格 → 它是新样本，计入样本
  unverifiable          没有门格
  duplicate_bytes       归档副本
```

**为什么不能按根名判**：`_gate80` 根下的四格（`B3` s101/s202/s303、`M6` s101 @80ep）是**全新样本**，
任何根下都没有它们的原件。按根名前缀判会把它们当复现吃掉——`usable` 停在 75，而正确值是 79。
反向的错误同样存在：把复现当样本，同一跑会被算两次。

**两个方向都会错，但只有一个方向会喊**——所以检查必须是双向的（`novel` 与 `twin` 同时报），
不能只查「有没有重复计数」。

**这条和 `budget_epochs` 是同一类**：`d80` 根里同时装着 40ep 和 80ep 两种格，
所以**任何按目录名推字段的做法都是错的**，字段必须读 `metrics.json`。

## 两个必须随表引用的口径

1. **`preds_md5` 是 sha256**。列名沿用旧名以免打断管线，值不是 md5。引用哈希时写 `preds_sha256`。
2. **`usable=no` 的门格不是样本**。它们与原始格 `preds`/`metrics`/`ckpt` 逐字节相同，只作复现证据。**具体行数不写在这里**——它每跑一批门格就变一次，写死必然过期。

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
