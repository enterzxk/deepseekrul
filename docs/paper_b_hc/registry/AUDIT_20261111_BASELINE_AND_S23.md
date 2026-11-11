# 审计快照 2026-11-11 23:1x · 基线哈希 v2 复核 + s23 24 run 完成度

维护：实验结果管理员。本文只记录实测事实与判定，不替代 `TASK_BOOK` §1（基线表由主控维护）。

---

## A. `paperb-s23-seeds-20261111` · 24/24 完成度

**结论：24/24 完整，无 NaN、无中断，可以关任务。** 但有两条必须随关单一并记录的口径事实（A.2、A.3）。

### A.1 逐 run 产物

对 `paper_b/experiments/grids/factorial_s23.json` 的 24 条配置（3 tag × 4 子集 × seed 202/303）
逐条检查 `paper_b/results/raw/<tag>/<subset>/s<seed>/`：

| 检查项 | 结果 |
|---|---|
| `metrics.json` 存在 | 24/24 |
| `test_metrics.json` 存在 | 24/24 |
| 每 run 文件集完全一致 | 24/24（`config.json` `history.json` `metrics.json` `model.pt` `test_metrics.json` `test_pred.csv` `val_pred.csv`，7 个） |
| 数值含 NaN / Inf | 0 / 24（递归扫描两个 metrics 的全部嵌套字段） |
| `len(history) == epochs_run` | 24/24 |
| `best_val_rmse == min(history.val_rmse)` | 24/24 |
| 早停与 `patience=12` 自洽 | 21 个 gap=12 早停；3 个跑到 40 epoch 上限（gap<12），无异常终止 |
| `epochs_run` 范围 | 18–40（配置上限 40） |
| 累计 GPU 秒 | 5555.9 s ≈ 92.6 min，2×3090 并行下 ≈50 min 墙钟，与实测一致 |

`test_metrics.json` 的 mtime 全部落在 **22:15:02–22:15:13** 的 12 秒窗口内，成批收尾。

> **更正一处**：燕子盘点报"产物 21:39 落盘"。实测这 24 个 run 的 `test_metrics.json` mtime 是
> **22:15:02–22:15:13**；21:39 附近没有本批写入。进程确实已退出，结论不变。

### A.2 三 seed 汇总的 12 个 run 不属于本 job

`paper_b/results/seed_robustness_raw.csv` 有 36 行（cell `base`/`cand`/`full` × 4 子集 × seed 101/202/303），
但 `factorial_s23.json` 只声明 seed **202/303**。seed 101 的 12 个 run 是**既有产物**：

| seed | metrics.json mtime | 归属 |
|---|---|---|
| 101 | 19:11:51 – 20:31:18 | 早前一次 factorial 启动，非本 job |
| 202 / 303 | 22:15:02 – 22:15:13 | `paperb-s23-seeds-20261111`（本 job，24 个） |

因此：**本 job = 24 run（202/303）**；`RESULTS_INDEX` 里三 seed 均值
（base 13.4181 / cand 12.9294 / full 12.9821）是 **36 run 汇总**，其中 12 个 seed-101 run 属继承。
两者都真实、可复算，但关单时必须写明 job 是 2 seed，不能写成"本 job 产出三 seed"。

复算（独立，从 `seed_robustness_raw.csv` 起算，4 子集平均）：

| cell | 2 seed (202/303) val / test | 3 seed (101/202/303) val / test |
|---|---|---|
| base | 13.311 / 11.780 | 13.418 / 11.655 |
| cand | 13.005 / 11.209 | 12.929 / 11.134 |
| full | 12.936 / 11.612 | 12.982 / 11.512 |

### A.3 证据缺口（不阻塞关单，但必须记）

- **无代码溯源绑定**：24 个 `config.json` 里**没有** `git_sha` / `code_md5` / `split_hash` 字段
  （实测键集：`aux_dual batch cuts d_model data_dir dropout epochs fold global_hi hi_lambda
  hi_mono hi_mono_coef hi_source input_mode instnorm layers legacy_ema legacy_ms_pad legacy_rul lr …`）。
  即这 24 个 run 无法从产物自证用的是哪一版 `train.py`。与 TASK_BOOK X4 的 legacy 标注一致，
  但意味着**它们永远不能与 hc 结果同表**，也不该在论文里当作可复现证据。
- **无 launch/close**：本线不经 hc registry schema，台账里这 24 行是按 `results/raw/` 回填的 legacy 行。

---

## B. 基线哈希 v2 复核：**不通过，现在冻结不了**

### B.1 三个时间点的同一批文件

| 文件 | 22:36 tmh 报告 | 22:49 HEAD `f314268` | 23:03 实测（工作区 / a40） |
|---|---|---|---|
| `hc/config_hc.py` | `64bf6d55…` | `0f865719…` (8931 B) | **`bffe552f…` (9088 B) / `bffe552f…`** |
| `hc/train_hc.py` | `64b90518…` | `64b90518…` (11630 B) | **`3fb05844…` (11871 B) / `3fb05844…`** |
| `hc/run_hc.py` | `959a490f…` | `959a490f…` (9536 B) | **`d2f08425…` (本地) / `d1c4d87e…` (a40)** |
| `training/train.py` | `ef14ee21…` | **`13ec2fbb…` (9239 B)** | `13ec2fbb…` / `13ec2fbb…` |
| `training/metrics.py` | `a799d43f…` | **`39202682…` (2556 B)** | `39202682…` / `39202682…` |
| 其余 9 个 | — | 未变 | 未变，本地 = a40 |

`git status` 当前把 `hc/config_hc.py`、`hc/train_hc.py`、`hc/run_hc.py` 标为 **M（未提交）**，
HEAD 仍是 `f314268`（22:49:05）。

### B.2 判定

1. **TASK_BOOK §1 v2 的三个值全部已失效**：`train.py`、`metrics.py` 在 v2 写入（22:40）之后又被改，
   随 `f314268` 入库；`config_hc.py` 现在已是第三次内容（`bffe552f`）。
   v2 是照 tmh 22:36 的报告抄的，抄完文件还在动。
2. **本地 ≠ a40**：14 个文件里差 1 个 —— `hc/run_hc.py`（本地 `d2f08425` / a40 `d1c4d87e`，a40 mtime 23:01:04）。
3. **tmh 仍在写**：a40 上 `config_hc.py` 22:57:13、`train_hc.py` 22:55:21、`run_hc.py` 23:01:04；
   `registry/` 里 23:00–23:08 还在新增 `HC-SMOKE-…1500Z`、`HC-C-A3p-…1505Z/1508Z`、`HC-C-A3a-…1505Z`。
4. 因此 **v2 作废，本次不登记生效基线**。现在冻结只会冻住一个半成品，
   且此刻跑出的任何 manifest 的 `code_md5` 都对不上。

### B.2b 更新（23:12 复测）

HEAD 已从 `f314268` 走到 **`f83e27e`（23:12:02，"Paper B HC v4: launch guard + code lock + crash-safe close"）**。
`paper_b/hc/` 与 `paper_b/training/`、`paper_b/models/`、`paper_b/configs/` 在 `f83e27e` 上**全部干净**；
11 个 dirty 文件全在 factorial 线（`experiments/seed_*.py`、`results/seed_robustness_*.csv`、`reports/SEED_ROBUSTNESS.md`），不属 hc 代码。

a40 现落后 **3 个文件**：

| 文件 | HEAD `f83e27e` | a40 现值 |
|---|---|---|
| `hc/model_hc.py` | `95e6685ef7f3c12f56ce72bf5ff87866` (4914 B) | `adcbceb901ae59d2bd15bbfc4067fc3f` |
| `hc/train_hc.py` | `c8f3cfe5cf12387019bbf280e4dd7975` (12535 B) | `3fb058441ffe12b0ece1ff38e11a3027` |
| `hc/run_hc.py` | `b3ee799745bb9475d136b8c4aec2b38f` (15191 B) | `d1c4d87e9456cb801d32ce30801ebf6a` |

其余 11 个文件本地 = a40。**v3 候选 = `f83e27e` 的这 14 个 blob**，只等 a40 补齐这 3 个即可冻结。

### B.2c 独立复核（23:25，按任务书 §1「基准取 `git show HEAD:<path>`」）

**HEAD = `0280cb8`（23:21:27，"Paper B HC v4: 15-blob code lock for the a40 host"）。**

用 `git cat-file blob HEAD:<path>` 逐文件取 blob（不经工作树、不经 scp）重算 15 个 md5，
与燕子 23:31 报的表**逐行全等**，且 a40 独立 `md5sum` 也逐行全等 —— **15/15 一致，燕子这份快照成立**。

| 文件 | md5（HEAD = a40） |
|---|---|
| `models/dspt.py` | `fb7011023b284a5f571cba4d4abb6927` |
| `training/data.py` | `c2ee079f6e60a8b0d0eb608676ff871a` |
| `training/train.py` | `13ec2fbb75ed09a389e4e979a82e11e6` |
| `training/metrics.py` | `39202682865d14dc6be5d332ac7a1c16` |
| `configs/protocol.json` | `c55720ffe09deff3a9b59c218a180114` |
| `hc/config_hc.py` | `bffe552fefa49c0e9cff38a31c6cd8ff` |
| `hc/data_hc.py` | `3ad7c111cf0268409bb40d79f7271cee` |
| `hc/model_hc.py` | `26e2d7c3ad963a0dcc2eeda437902f1d` |
| `hc/cycle_encoder.py` | `f0ddc9115ba49627d4804fa6d5752e6d` |
| `hc/train_hc.py` | `c8f3cfe5cf12387019bbf280e4dd7975` |
| `hc/run_hc.py` | `45bedd9791dd1dee9861da6b9eaa3f34` |
| `hc/verify_loop.py` | `111c7be5106a74dc0e1152c68a69308b` |
| `hc/snapshot_val.py` | `be21a3cdc6f7022f3871faaf9d33a69b` |
| `hc/compare_val_snapshot.py` | `daff09811d4dbe6fa74c8bcb6190b2a6` |
| `hc/code_lock.py` | `e30f3748a2e2529b7fdd322e4a32d211` |

`code_lock.json` 本地 = a40 = `663e0c7b0d5fc20e5c35fb28aa75c135`。
其中 `git_sha=a203f4b`（父提交）**不是漂移**：实测 `git diff a203f4b..HEAD --stat` 只有
`registry/code_lock.json | 13 +++++++------`（7 增 6 删），15 个代码 blob 一个没动，燕子这条判断正确。

**但工作树此刻不干净**（23:25 实测）：

```
 M paper_b/hc/model_hc.py     (工作树 c55eb2b4…  vs HEAD 26e2d7c3…)
 M paper_b/hc/run_hc.py       (工作树 d6b6e381…  vs HEAD 45bedd97…)
 M paper_b/hc/train_hc.py     (工作树 1f8d2b84…  vs HEAD c8f3cfe5…)
?? paper_b/hc/run_v0.py
```

即 tmh 正在工作树上做 v5 的两个阻塞项（`--dump-init`、`pe_proj ↔ cycle_enc.proj` 双向 remap）。
燕子报"`git status` 空"在她采样时成立，23:25 已不成立。

### B.2d 判定：**不登记 `0280cb8`**

1. `0280cb8` 是 **v4**，而主控已裁定 v5 还要补两个阻塞项；登记它等于给一个已知会被替换的提交签基线。
2. 工作树已有 3 个文件越过 `0280cb8`，一旦 tmh 提交 v5，a40 立刻又落后 —— 现在冻结必然是空转。
3. 待 v5 定稿提交后，按同一方法（`git cat-file blob HEAD:<path>`）重算 15 个 blob，
   与 tmh 表逐行比对，再走「登记 → 燕子同步复算 → 干净树 smoke」。


> 已存在的 4 个 SMOKE（`…1431Z` failed、`…1436Z`、`…1447Z`、`…1500Z`）都跑在移动中的工作树上，
> 一律只作流水线自检，不作证据。`…1447Z` 已因 `config_hc.py` 脏工作树判 fail。

---

## C. 冻结窗口起点实测（23:43）· 以及一个必须先排除的假警报

### C.1 当前事实

- HEAD = **20b9f3b（23:42:25，"Paper B HC v6: 15-blob code lock"）**
- `paper_b/hc/` **干净**；`paper_b/` 下的 11 个 dirty 文件全在 factorial 线
  （`experiments/seed_report.py`、`experiments/seed_robustness.py`、`reports/SEED_ROBUSTNESS.md`、
  `results/seed_robustness_*.csv`，另有 3 个未跟踪文件），**没有一个在 hc 下**
- `registry/code_lock.json`（HEAD 里那份）：`baseline_version=v6`、`git_sha=6dc83486`、
  `generated_utc=2026-11-11T15:42:25Z`、15 个文件、自身 md5 `723c71fd…`
- **v6 锁 vs HEAD blob：15/15 全等，0 处不符**

### C.2 那个"锁 ≠ 工作树"的警报是 CRLF 造成的

实测：`code_lock.json` 的 `code_md5` 与**工作树字节**比，有 2 处不符：

| 文件 | 锁里 / HEAD blob | 工作树字节 |
|---|---|---|
| `hc/train_hc.py` | `82163825…` | `ec61f61f…` |
| `hc/run_hc.py` | `4e057e90…` | `c76d32b2…` |

但这**不是有人改了文件**：

`
core.autocrlf = true
.gitattributes = 不存在
train_hc.py  blob CRLF 对数=0  worktree CRLF 对数=310
run_hc.py    blob CRLF 对数=0  worktree CRLF 对数=382
blob 转 CRLF 后 md5 = 工作树 md5（两文件都逐字节相等）
去掉行尾差异后，工作树内容 == HEAD blob（两文件都是 True）
`

即：blob 是 LF，Windows 工作树被 `autocrlf` 转成 CRLF，**内容完全相同**，
`git status` 也判 clean。任何在 Windows 上拿"锁里的 md5"直接比"工作树 md5"的做法，
对每个含 CRLF 的文件都会报一次假不符。

### C.3 结论与口径

1. **比锁只能用 blob**：`git cat-file blob <sha>:<path>` 或 Linux 上 `git show <sha>:<path> | md5sum`。
   工作树字节一律不作基准——这一条比"禁用 scp"更靠前，因为它在本地就会骗人。
2. 建议根治：仓库根加 `.gitattributes`（`paper_b/** -text`）让这些文件在 Windows 上不被转换，
   或在 Linux 侧跑 `code_lock --write`。否则每次定稿都要靠人工解释一遍这 2 处假不符。
3. v6 的锁**与提交一致**，可以先排除"锁失效"这个方向，再查是否真有第三方在写。
   冻结窗口内的实测起点就是本节。


---

## D. 基线登记 · v8（2026-11-11 23:55，生效）

主控裁决 v8 通过，本节为台账侧的登记记录。**方法与任务书 §1 一致：全部取 git cat-file blob <ref>:<path> 字节，不用工作树。**

### D.1 提交

| 角色 | SHA | 时间 | 说明 |
|---|---|---|---|
| 代码提交 | `aef910c` | 23:50:59 | three-step shared-init acceptance in verify_loop |
| 锁提交 | `441c92f` | 23:51:01 | 15-blob code lock（lock-only commit），当前 HEAD |

`paper_b/hc/` 工作树干净（`git status --porcelain paper_b/hc` 为空）。

### D.2 锁内容（`registry/code_lock.json` @ `441c92f`，自身 md5 `7b0458bfc7444c61ebe2326d2a27d3f1`）

`schema_version=1`、`baseline_version=v8`、`git_sha=aef910c`（= 被锁代码所在提交，非 HEAD，语义正确）、
`generated_utc=2026-11-11T15:51:01Z`、`source=HEAD`、`definition="md5 of git cat-file blob <ref>:<path> bytes"`。

### D.3 15 个 blob 实测（三处比对全等）

| 文件 | md5 | 字节 |
|---|---|---|
| `paper_b/models/dspt.py` | `fb7011023b284a5f571cba4d4abb6927` | 14040 |
| `paper_b/training/data.py` | `c2ee079f6e60a8b0d0eb608676ff871a` | 14159 |
| `paper_b/training/train.py` | `13ec2fbb75ed09a389e4e979a82e11e6` | 9239 |
| `paper_b/training/metrics.py` | `39202682865d14dc6be5d332ac7a1c16` | 2556 |
| `paper_b/configs/protocol.json` | `c55720ffe09deff3a9b59c218a180114` | 582 |
| `paper_b/hc/config_hc.py` | `bffe552fefa49c0e9cff38a31c6cd8ff` | 9088 |
| `paper_b/hc/data_hc.py` | `3ad7c111cf0268409bb40d79f7271cee` | 4679 |
| `paper_b/hc/model_hc.py` | `c55eb2b430e2d356a809a0ccbbb3f3f0` | 5658 |
| `paper_b/hc/cycle_encoder.py` | `f0ddc9115ba49627d4804fa6d5752e6d` | 3954 |
| `paper_b/hc/train_hc.py` | `bc24115673b8d57bbc47694b5991321b` | 15413 |
| `paper_b/hc/run_hc.py` | `8784311bebdc6891b59058f278e8d1f8` | 17814 |
| `paper_b/hc/verify_loop.py` | `2c15fcb4293e226102e368c717ef3f3a` | 10345 |
| `paper_b/hc/snapshot_val.py` | `be21a3cdc6f7022f3871faaf9d33a69b` | 3669 |
| `paper_b/hc/compare_val_snapshot.py` | `daff09811d4dbe6fa74c8bcb6190b2a6` | 2882 |
| `paper_b/hc/code_lock.py` | `cd492a1cdc6a812120eceea6f69f7a41` | 3208 |

三处比对：

| 比对 | 结果 |
|---|---|
| 锁 vs `git cat-file blob aef910c:<path>` | **0/15 不符** |
| 锁 vs `git cat-file blob 441c92f:<path>`（HEAD） | **0/15 不符** |
| 锁 vs a40 实测 `md5sum`（LF 工作树） | **0/15 不符** |

a40 的 `docs/paper_b_hc/registry/code_lock.json` md5 = `7b0458bfc7444c61ebe2326d2a27d3f1`，与本地一致。

**判定：v8 基线成立，登记生效。** 此后任何 run 的 manifest `code_md5` 只有与本节 15 个值全等才可入账。

### D.4 随登记一并挂起的待办

1. **v9 待办（主控已裁）**：`hc/code_lock.py:50` 的 `check()` 读工作树字节时未做 `.replace(b'\r\n', b'\n')` 规范化，
   在 `core.autocrlf=true` 的 Windows 上会报假不符。修法：读字节后规范化再比，并在 `definition` 里写明
   "本地工作树是 CRLF、a40 是 LF，须按规范化后字节比"。**留到 Gate A 第一波之后，不阻塞当前放行。**
2. **a40 目录残留**：`paper_b/hc/` 下有 6 个 `tmp_*.py`（`tmp_a8 / tmp_a8b / tmp_dump / tmp_patch / tmp_patch2 / tmp_seed`，23:22:57 那批），
   不在 15-blob 锁内，a40 无 git 故不触发 guard。由 tmh 清理，登记于此避免遗忘。
3. 仓库根仍无 `.gitattributes`，CRLF 问题在 v9 之前会持续存在；期间**一律以 blob 为准**。


---

## E. 基线登记 · v9（2026-11-12 00:1x，生效；v8 同时作废）

**§D 的 v8 登记作废**，理由：v8 的 	rain_hc.py remap 用 pop 搬 pe_periods，
A3a 的 state_dict 是 61 键（pe_periods 与 cycle_enc.periods 两个都在），pop 走一个就缺一个，
--init-from 必炸 missing=['pe_periods']。v9 d1e0527 改成 copy 才通过。
**v8 不是"次优"，是共享初始化路径上必然失败的一版**，不能作为任何 run 的基线。

### E.1 提交

| 角色 | SHA | 时间 |
|---|---|---|
| 代码提交 | `d1e0527` | 23:56:12 | fix A3a init remap (copy pe_periods, do not pop) + CRLF-normalised lock check |
| 锁提交 | `6c405ad` | 23:56:13 | 15-blob code lock（lock-only commit），当前 HEAD |

`paper_b/hc/` 工作树干净。

### E.2 锁内容（`registry/code_lock.json` @ `6c405ad`，自身 md5 `c04c958600ca26fed9ea6944cbe12cbc`）

`baseline_version=v9`、`git_sha=d1e0527`、`generated_utc=2026-11-11T15:56:13Z`、`source=HEAD`、
`definition="md5 of git cat-file blob <ref>:<path> bytes"`。

### E.3 相对 v8 只变两个值，其余 13 个不动

| 文件 | v8 | **v9** |
|---|---|---|
| `hc/train_hc.py` | `bc241156…` | **`069f8977bee8cf6aa01953c817d01601`** (15451 B) |
| `hc/code_lock.py` | `cd492a1c…` | **`45916d9a08c88b40072c03f2127cfd07`** (3574 B) |

沿用：`dspt=fb701102`、`data=c2ee079f`、`train=13ec2fbb`、`metrics=39202682`、`protocol=c55720ff`、
`config_hc=bffe552f`、`data_hc=3ad7c111`、`model_hc=c55eb2b4`、`cycle_encoder=f0ddc911`、
`run_hc=8784311b`、`verify_loop=2c15fcb4`、`snapshot_val=be21a3cd`、`compare_val_snapshot=daff0981`。

### E.4 三处比对（全部 0 处不符）

| 比对 | 结果 |
|---|---|
| 锁 vs `git cat-file blob d1e0527:<path>` | **0/15** |
| 锁 vs `git cat-file blob 6c405ad:<path>`（HEAD） | **0/15** |
| 锁 vs a40 实测 `md5sum` | **0/15** |

a40 的 `registry/code_lock.json` md5 = `c04c9586…`，与本地一致。

**判定：v9 基线成立，登记生效。** 自此 Gate A 每个 manifest 的 `code_md5` / `baseline_version` 必须与本节的 v9 全等。

### E.5 待办销账

- §D.4 第 1 条（`code_lock.py` CRLF 规范化）**已由 v9 销账**。
- §D.4 第 2 条（a40 的 6 个 `tmp_*.py`）**仍未清**，a40 `paper_b/hc/` 下仍在。
- §D.4 第 3 条（无 `.gitattributes`）**仍在**：v9 的 `check()` 已在工具内做归一化，
  但人工比对仍须以 blob 为准。

---

## F. 台账缺口（2026-11-12 00:1x 实测）· 需主控裁决单一写入人

### F.1 `runs.csv` 与 registry 实体已脱节

| 项 | 实测 |
|---|---|
| a40 `experiments/hc/` 下的 run 目录 | **18 个**（另有 `tmp/`） |
| a40 `registry/launch/` | **17 个 json** |
| a40 `registry/close/` | **14 个 json** |
| a40 `runs.csv` 数据行 | **32 行** |
| 本地 `runs.csv` 数据行 | **32 行**（与 a40 的 run_id 集合完全相同） |

即：**`runs.csv` 落后于 launch/close 实体**。以下 run 有 launch/close 文件，但**两侧 `runs.csv` 都没有它们**：

| run_id | baseline | close 状态 | epochs | 判定 |
|---|---|---|---|---|
| `HC-C-A3p-FD001-s101-20261111T1625Z` | v5 | **无 close** | — | 残留，superseded（郑总已判） |
| `INIT-DUMP-A3p-FD001-s101` | v5 | **无 close** | — | 作废；且 dump run 本不该有 launch/close |
| `HC-C-A3a-FD001-s101-20261111T1630Z` | v8 | failed | 0 | superseded（`pe_periods` 缺失） |
| `HC-C-A3a-FD001-s101-20261112T0005Z` | v8 | failed | 0 | superseded（同上） |
| `HC-C-A3a-FD001-s101-20261112T0012Z` | v6 | failed | 0 | superseded（同上） |
| `HC-C-A3p-FD001-s101-20261112T0012Z` | v6 | failed | 0 | superseded（同上） |
| `HC-C-A3p-FD001-s101-20261112T0005Z` | v8 | ok | 36 | superseded（v8 基线作废） |
| `HC-C-A3p-FD001-s101-20261112T0030Z` | v9 | ok | 36 | **有效**，待登记 |
| `HC-C-A3a-FD001-s101-20261112T0030Z` | v9 | **无 close**（运行中） | — | 待收尾 |
| `HC-C-A3a-FD001-s101-20261111T1640Z` | v9 | **无 close** | — | 待确认死/活 |

### F.2 未关闭的 launch（4 个）

`A3p-1625Z`（v5）、`INIT-DUMP-A3p`（v5）、`A3a-1640Z`（v9）、`A3a-0030Z`（v9，燕子报 ep015 在跑）。
前两个是残留，后两个需 tmh 确认：**运行中，还是异常退出未写 close**。

### F.3 需要裁决

`runs.csv` 在本轮被并发改过至少两次（我两次读到的行集合不同），且 a40 侧另有一份。
按"一人一文件"，请指定 `runs.csv` 的唯一写入人。
我的建议：**由我按 a40 的 launch/close + `metrics.json` 一次性补齐**（superseded 行标 `status=superseded`、
`registry_note` 写明 baseline 与作废原因），主控与 tmh 不再直接写该文件；
否则每补一次都会被覆盖，Gate A 开始后 manifest 与台账必然对不上。


---

## G. 台账补齐 · runs.csv（2026-11-12 00:2x，唯一写入人执行）

按主控裁决，`runs.csv` 自此只有本管理员写入。本次一次性补齐，**原有 40 行的 status / epochs_run / val_rmse / baseline_version / audit_verdict 字段零改动**（逐行比对 0 处变化）。

### G.1 结果

| 项 | 数值 |
|---|---|
| 补齐前 | 40 行 |
| 新增 | **8 行** |
| 补齐后 | **48 行**，36 列 |
| 新增 superseded（v5/v6/v7/v8） | 9 行置 `retired_version=true` |
| 备份 | `runs.csv.bak-0015` / `runs.csv.bak-0020` |

新增的 8 行：`A3a-1640Z`(duplicate)、`A3a-0030Z`(v9 ok)、`A3p-1625Z`(v5 unclosed)、
`A3p-1640Z`(v9 unclosed)、`INIT-DUMP-A3p`(v5 unclosed)、`A3a-R1-0035Z`(v9 unclosed)、
`A3p-R1-0035Z`(v9 unclosed)、`A3p-0105Z`(v9 unclosed)。

### G.2 当前 Gate 可用行（v9 且未 retired）

| run_id | 状态 | epochs | val_engine_mean | 用途 |
|---|---|---|---|---|
| `HC-C-A3p-FD001-s101-20261112T0030Z` | ok | 36 | **10.8951** | Fourier 主臂 |
| `HC-C-A3a-FD001-s101-20261112T0030Z` | ok | 37 | **11.3167** | 敏感性臂 |
| `HC-SMOKE-FD001-s101-20261112T0045Z` | smoke | 1 | 62.5931 | 链路自检，非证据 |

Δ = 11.3167 − 10.8951 = **+0.4216**，配对关系已在两行的 `registry_note` 互相写明。
其余 4 条 v9 为 `unclosed`（A3p-1640Z、A3a-R1-0035Z、A3p-R1-0035Z、A3p-0105Z），等 close 落盘再改。

### G.3 去重键有个漏洞，必须记下来

主控定的键是 `(baseline_version, variant, subset, seed, init_sha256)`。**这个键抓不到 `A3a-1640Z`。**

实测两个 A3a run 的 `manifest.init_sha256`：

| run_id | init_sha256 |
|---|---|
| `A3a-0030Z` | `336074a6e173d8df70bba5f5053508990318260c793a4c4515ad88033456c3c8` |
| `A3a-1640Z` | `2138ac5189be5a154d123fef68f4524fe00fe2f7b8fbf70f2568fc44a5f06d89` |

两者**不同**——因为 init 是两份文件，payload 里的 `created_utc` 不同；但两份文件里的 `state_dict` 哈希同为 `af23279d…`（60 个张量），权重完全一样。所以按 `init_sha256` 去重，这两个 run 会被判成"不同 init"，重复照旧漏过去。

本次 `1640Z` 标 `duplicate` 是按主控明确裁决执行的，不是键算出来的。

**建议**：manifest 增记 `init_state_dict_sha256`（对 state_dict 逐张量哈希，不含 payload 元数据），
去重键换成 `(baseline_version, variant, subset, seed, init_state_dict_sha256)`。
在此之前，同格重复只能靠人工比对 `ckpt/best.pt` 与 `preds.csv` 的 sha256 发现。
此项进 v10 backlog，不动 v9。


---

## H. 台账冻结记录（2026-11-12 00:5x，主控令）

### H.1 冻结状态

| 项 | 值 |
|---|---|
| 行数 | **50** |
| 列数 | **38**（v1 起 32 列 → 33 `val_metric_kind`/`legacy_metric`/`baseline_version`/`retired_version` → 37 `determinism_only` → 38 `in_gate_a_table`） |
| `check_runs.py` | **rows=50 flags=0** |
| git | 干净；台账提交链 `8436878` → `8798b87` → `b813154` |
| 冻结后动作 | 不再追加登记，直到 W1 格落盘并按预登记清单准入 |

### H.2 Gate A 准入口径（主控 2026-11-12 定）

**准入 = `run_id` 在 `GATE_A_WAVE1_PLAN.md` §3 预登记清单里**，不看 `gate` 标签。
`gate` 字段是 launch 时人填的，会填错；清单是发车前写死的。

因此台账新增 `in_gate_a_table` 列，当前 **50 行全为 false**——W1 格尚未登记，此刻没有任何 run 是 Gate A 证据。
登记时按预登记 `run_id` 逐格比对后置 true，**不用 `init_from` 是否为空做条件**
（W1 的格除 A0 外全部带 `init_from`，骨干共享是刻意的公平性设计）。

`gate` 字段保持原值不动（`0030Z` 两条已按裁决置 `R`），因为它已不承担准入语义。

### H.3 检查器改动披露（主控要求留痕）

`check_runs.py` 初次运行报 **34 flag**，逐条核完：**31 条是检查器自身的规则缺陷，3 条是真数据问题**。

**31 条假阳性（改规则，不改数据）**

1. `R-a` 的 `LEGAL` 枚举缺 2026-11-12 新引入的状态对：
   `superseded/n-a`、`unclosed/n-a`、`running/n-a`、`unclosed|running|ok + determinism_only`、`ok/duplicate`、`failed/fail`。
   主控新定的去重与确定性口径一写进台账，就被判成"不一致"。
2. `R-c` 把两类合法行误判为 legacy 口径：
   - 24 条 Gate B 行**本来就** `legacy_metric=true`，已被下游排除，再报一次没有信息量；
   - a40 上的 hc run `git_sha=WORKTREE` 是**设计如此**（a40 无 git），版本绑定在 `baseline_version` + 15-blob `code_md5`，不在 commit sha 上。

原文件保留为 `tmp/audit_hc/check_runs.py.orig`。**没有为了过检查修改任何一行台账数据**。

**3 条真数据问题（改数据）**

| run | 原值 | 改为 | 依据 |
|---|---|---|---|
| `A3a-0030Z` / `A3p-0030Z` | `gate=A` | `gate=R` | 不在预登记清单内，是注入点对照，非 Gate A 证据 |
| `A3a-1640Z` | `retired_version=true` | `false` | 它是 v9 的**成功** run（11.3167），`retired_version` 只该给版本号复用的 v6/v7 |
| `A3a-R2` / `A3p-R2` | `running` | `running`→aborted 说明 | 跑到 ep025/ep030 被终止、无 close；`determinism_only`，不进任何表 |

### H.4 并发写入的遗留风险

冻结前 `runs.csv` 至少被两个写入方交替改写：我置的 `determinism_only` / `duplicate` 判决曾被回写为 `pass` / `n-a`，
`gate=R` 也曾被改回 `A`。**当前状态已回读确认稳定**（改后 4 秒与再读一致），且已入库。

若后续再出现"读一次一个样"，说明仍有第二个写入方在跑；届时以**最近一次 commit 的 blob** 为准
（`git cat-file blob HEAD:docs/paper_b_hc/registry/runs.csv`），不看工作树。

