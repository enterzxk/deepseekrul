# Code-snapshot audit: M2 (CDE) across its three seeds

日期 2026-11-13 · 执行 tmh · 探针 `docs/paper_b_hc/probes/audit_m2_snapshot.py`

## 结论

**M2 的 s101 不需要重跑。** 两个候选快照之间，M2 的数值路径没有变动：

- `cde.py`（CDE 证据编码器、`q_t`、`cde_head`）在 s101 快照、s202/s303 快照和 HEAD 上是**同一份字节**，md5 `9de62ab1`；
- `train_d.py` 的全部改动落在 `term_grads()` 及其唯一调用点，那是**优化器步进之后的诊断**，只写 `history.json`；
- `SCR_ARMS = ('M7',)`，M2 两侧都是 `scr=False`，新增的 `scr=` / `lam_a=` 参数对本臂是空操作。

两条独立理由同向：燕子的进程启动时刻（18:41:27 / 18:42:45 均早于 18:43:38 的同步）表明三个 seed 实际都 import 了 `a92ebaf1`；本审计则证明，**即使按最初担心的 `66dd13e → 4ea5d5d` 跨两个快照，M2 的路径也不动**。任一成立即可放行，两者同时成立。

## 判定依据（探针逐条打印，全过才 `AUDIT OK`）

| 检查 | 内容 | 结果 |
|---|---|---|
| A | `cde.py` 在 s101 / s202 快照 / HEAD 三处哈希一致 | PASS `9de62ab1` |
| B | `train_d.py` 的 `66dd13e → 4ea5d5d` 改动全部在 `term_grads()` 内或其调用点 | PASS 13 行改动，0 行越界 |
| C | `SCR_ARMS` 不含 M2，两侧 `scr=False` | PASS `('M7',)` |
| D | 该快照之后的提交只碰指纹记录器，不碰训练路径 | PASS |

`term_grads()` 为何不落数值：它在 `sch.step()` 之后调用（`train_d.py` 训练循环末尾），内部用 `torch.autograd.grad(..., retain_graph=True)` 只取范数，不回写 `.grad`、不修改参数、不调用优化器；返回值只用于 `grad_ratio` 与 `history.json`。`_step()` 里 `alive` 项是 `if 'alive' in L` 才加权，而 M2 的 `L` 没有该项。

复跑：`python docs/paper_b_hc/probes/audit_m2_snapshot.py`

## 指纹记录的两处修复（同批提交）

**① 指纹改为 import 时快照。** `code_fingerprint()` 原先在写 `metrics.json` 时才调用（`train_d.py:742`），而函数体读的是**那一刻盘上的文件**。训练期间任何一次同步都会把指纹改写成「收工时的盘」，而不是「启动时加载的代码」——这一整轮对不上的指纹（含那个查无 commit 的 `e89b23b0`）都由这一处产生。现在 `_FP_AT_IMPORT = code_fingerprint()` 在模块加载时求值一次，写盘时用该常量。

**② 逐文件哈希 + 行尾归一化。** `metrics.json` 新增 `code_fingerprint.file_sha256`（文件名 → 哈希字典），聚合值 `files_sha256` 保留以便对比。哈希前把 CRLF 折叠为 LF，即**按 git blob 的字节**哈希：仓库 `core.autocrlf=true`，同一 commit 在 Windows 检出是 CRLF、在 Linux 检出是 LF，直接哈希工作树会让同一份代码产生两个指纹——正是这个字段要防的「变了但说不清为什么」。已在 `cde.py` 上验证：工作树 LF 归一化后的 sha256 与 `git cat-file blob HEAD:paper_b/hc/cde.py` 的 sha256 逐位相同（`750a7ebf…`）。

**口径说明**：`files_sha256` 自本次提交起按 LF 归一化计算。对全 LF 的检出（3090）而言这是恒等变换，值只随文件内容变化；对 Windows 检出，新值会与旧值不同，但旧值本来就不可归因，不作追溯。

## 未覆盖

- 本审计只回答「M2 的数值路径是否跨快照移动」，不评价 M2 的增益是否成立——那要靠三 seed 落盘后的集成比较。
- 指纹修复后，**此前所有已落盘 `metrics.json` 的 `code_fingerprint` 仍按旧口径记录**，不得与新格直接比较；引用时须注明该格写盘时刻。
