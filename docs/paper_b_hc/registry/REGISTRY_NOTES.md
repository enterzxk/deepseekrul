# Registry notes（append-only 记录）

状态：进行中
最后更新：2026-11-11 23:36 (+08)
写入人：tmh（launch/close 与本节）；`runs.csv` 仍由实验结果管理员写

## 1. 规则（2026-11-11 裁决）

- a40 上的 `registry/launch/`、`registry/close/` 是**权威副本**；本地是单向拉取的镜像。
- **只追加，不删除**。作废用 `status=superseded`（或 close 里的 `failed`）表达，不删文件。
- launch 由 `run_hc` / `run_v0` 在进程启动时用 `O_CREAT|O_EXCL` 生成，写完**回读自检**，带
  `written_by` 与 `process_start_utc`。手写或复制上一份改字段的 launch 视为伪造。
- `hypothesis` 必填；填 `(not stated)` 的 run 按 fail 处理。
- 代码指纹：`baseline_version` + `git_sha`（a40 上恒为 `WORKTREE`）+ `worktree_dirty`。
  `worktree_dirty=true` 的 run 不进任何 Gate 表；启动守卫保证这种 run 根本起不来。

## 2. 删除记录（不恢复）

| run_id | 情况 |
|---|---|
| `HC-C-SMOKE-FD001-s101-20261111T1431Z` | a40 上的 launch/close 已被删除，本地仍保留副本；按裁决**不恢复**，仅在此记账。 |

## 3. Run 清单（按时间）

| run_id | 版本 | 状态 | 说明 |
|---|---|---|---|
| `HC-C-SMOKE-…1431Z` | 无 | deleted | 见 §2 |
| `HC-C-SMOKE-…1436Z` | 无 | superseded | 并行线的流水线自检，早于任何基线 |
| `HC-SMOKE-…1447Z` | 无 | superseded | 工作树 `config_hc` 与 HEAD 不一致，manifest 对不上 |
| `HC-SMOKE-…1500Z` | v3 候选 | superseded | HEAD `f314268` 的干净树 smoke，后被 v4/v5 取代 |
| `HC-C-A3a-…1505Z` / `HC-C-A3p-…1505Z` / `HC-C-A3p-…1508Z` | 无 | superseded | 脏树上的 A3a/A3p，作废（主控裁决） |
| `HC-SMOKE-…1522Z` | 中间提交 `2482baa` | superseded | v4 中间态（已被单提交 v5 取代），不作证据 |
| `HC-C-A3p-…1615Z` / `HC-C-A3a-…1615Z` | **v5** | 见 close | 共享初始化对照：A3p 先 dump init，A3a 用同一份 dump 加载 |

## 4. 版本对照

- v3 候选：`f314268`（14 blob）——未登记为正式基线。
- v4 中间态：`2482baa` / `8066f3f`——被 v5 取代，**不登记**。
- **v5**：`3f38cfa`（代码）+ `4c7f238`（15-blob `code_lock.json`）——待管理员登记。
