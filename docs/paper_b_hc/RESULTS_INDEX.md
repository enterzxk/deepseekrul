# DSPT-HC 实验索引

最后更新：2026-11-12 02:15 · 已登记 run 数：50（treatment 44 / determinism 6 / legacy 24）

> 由实验结果管理员维护，**由 `runs.csv` 生成，勿手改**。**提交前必须通过 `tools/check_runs.py`（退出码非 0 拒绝提交）。**

> **两轴语义**：`status`=进程结果（ok/failed/smoke/running/superseded）；`audit_verdict`=证据可用性（pass/fail/suspect/n/a/空）。**可用结果 = `status=='ok' and verdict=='pass'`。**

> **代码基准**：v9 = 代码 `d1e0527` / 锁 `6c405ad`。**只有 v9 的 treatment run 可作 Gate A 证据。**

> **`determinism_only=true`（gate=R）**：同 seed 的重复实验，用于验证可复现性，**一律不进任何 Gate 表**。

> **registry 读取源**：`launch/` 与 `close/` 以 **a40 为唯一权威源**，本地副本是单向镜像，可能滞后。枚举前先同步。

## 当前基线 v9 · treatment（Gate A 证据）

> **A3a/A3p 配对成立**：同代码 v9、同 seed 101、同 init（`336074a6…` 三方一致），唯一差异为注入点。A3a `11.3167`（37 ep）vs A3p `10.8951`（36 ep），**Δ = +0.4216（A3a 差 3.87%）**，单 seed，观察非结论。

| run_id | variant | subset | seed | val_rmse | kind | status | 审计 |
|---|---|---|---|---|---|---|---|
| HC-C-A3p-FD001-s101-20261112T0030Z | A3p | FD001 | 101 | 10.8951 | engine_mean | ok | pass |
| HC-SMOKE-FD001-s101-20261112T0045Z | SMOKE | FD001 | 101 | 62.5931 | engine_mean | smoke | n/a |
| HC-C-A3a-FD001-s101-20261112T0030Z | A3a | FD001 | 101 | 11.3167 | engine_mean | ok | pass |
| HC-C-A3p-FD001-s101-20261111T1640Z | A3p | FD001 | 101 | — | — | failed | n/a |

## 确定性重复（gate=R，不进 Gate 表）

> A3a/A3p 的 R1 重复与首跑**逐位相同（σ=0）**，是今晚最强的可复现性证据。R2 尚无 close，标 `running`。

| run_id | variant | subset | seed | val_rmse | kind | status | 审计 |
|---|---|---|---|---|---|---|---|
| HC-C-A3a-FD001-s101-20261111T1640Z | A3a | FD001 | 101 | 11.3167 | engine_mean | ok | pass |
| HC-C-A3a-FD001-s101-R1-20261112T0035Z | A3a | FD001 | 101 | 11.3167 | engine_mean | ok | pass |
| HC-C-A3p-FD001-s101-20261112T0105Z | A3p | FD001 | 101 | — | — | failed | n/a |
| HC-C-A3p-FD001-s101-R1-20261112T0035Z | A3p | FD001 | 101 | 10.8951 | engine_mean | ok | pass |
| HC-C-A3a-FD001-s101-R2-20261112T0035Z | A3a | FD001 | 101 | — | engine_mean | running | — |
| HC-C-A3p-FD001-s101-R2-20261112T0035Z | A3p | FD001 | 101 | — | engine_mean | running | — |

## 其他 treatment run（非 v9）

| run_id | variant | subset | seed | val_rmse | kind | status | 审计 |
|---|---|---|---|---|---|---|---|
| HC-C-SMOKE-FD001-s101-20261111T1431Z | SMOKE | FD001 | 101 | — | — | failed | n/a |
| HC-C-SMOKE-FD001-s101-20261111T1436Z | SMOKE | FD001 | 101 | 62.5931 | engine_mean | smoke | n/a |
| HC-B-F_single_proj_mean_hi0-FD001-s202-20261111T1334Z | F_single_proj_mean_hi0 | FD001 | 202 | 12.4424 | pooled | ok | pass |
| HC-B-F_single_proj_mean_hi0-FD001-s303-20261111T1334Z | F_single_proj_mean_hi0 | FD001 | 303 | 11.8451 | pooled | ok | pass |
| HC-B-F_single_proj_mean_hi0-FD002-s202-20261111T1348Z | F_single_proj_mean_hi0 | FD002 | 202 | 11.5350 | pooled | ok | pass |
| HC-B-F_single_proj_mean_hi0-FD002-s303-20261111T1348Z | F_single_proj_mean_hi0 | FD002 | 303 | 12.1319 | pooled | ok | pass |
| HC-B-F_single_proj_mean_hi0-FD003-s202-20261111T1356Z | F_single_proj_mean_hi0 | FD003 | 202 | 12.7731 | pooled | ok | pass |
| HC-B-F_single_proj_mean_hi0-FD003-s303-20261111T1357Z | F_single_proj_mean_hi0 | FD003 | 303 | 12.8110 | pooled | ok | pass |
| HC-B-F_single_proj_mean_hi0-FD004-s202-20261111T1414Z | F_single_proj_mean_hi0 | FD004 | 202 | 15.9865 | pooled | ok | pass |
| HC-B-F_single_proj_mean_hi0-FD004-s303-20261111T1411Z | F_single_proj_mean_hi0 | FD004 | 303 | 16.9653 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD001-s202-20261111T1331Z | F_single_proj_basic_hi1 | FD001 | 202 | 10.9385 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD001-s303-20261111T1332Z | F_single_proj_basic_hi1 | FD001 | 303 | 10.7585 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD002-s202-20261111T1343Z | F_single_proj_basic_hi1 | FD002 | 202 | 12.4815 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD002-s303-20261111T1343Z | F_single_proj_basic_hi1 | FD002 | 303 | 12.0307 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD003-s202-20261111T1353Z | F_single_proj_basic_hi1 | FD003 | 202 | 11.7861 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD003-s303-20261111T1355Z | F_single_proj_basic_hi1 | FD003 | 303 | 12.3434 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD004-s202-20261111T1409Z | F_single_proj_basic_hi1 | FD004 | 202 | 16.9353 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD004-s303-20261111T1407Z | F_single_proj_basic_hi1 | FD004 | 303 | 16.7650 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD001-s202-20261111T1329Z | F_full_basic_hi1 | FD001 | 202 | 11.0119 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD001-s303-20261111T1329Z | F_full_basic_hi1 | FD001 | 303 | 10.9252 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD002-s202-20261111T1338Z | F_full_basic_hi1 | FD002 | 202 | 11.7306 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD002-s303-20261111T1338Z | F_full_basic_hi1 | FD002 | 303 | 12.2846 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD003-s202-20261111T1350Z | F_full_basic_hi1 | FD003 | 202 | 11.4674 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD003-s303-20261111T1350Z | F_full_basic_hi1 | FD003 | 303 | 13.1738 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD004-s202-20261111T1402Z | F_full_basic_hi1 | FD004 | 202 | 16.3040 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD004-s303-20261111T1403Z | F_full_basic_hi1 | FD004 | 303 | 16.5887 | pooled | ok | pass |
| HC-SMOKE-FD001-s101-20261111T1447Z | SMOKE | FD001 | 101 | 62.5931 | engine_mean | smoke | n/a |
| HC-SMOKE-FD001-s101-20261111T1500Z | SMOKE | FD001 | 101 | 62.5931 | engine_mean | smoke | n/a |
| HC-C-A3a-FD001-s101-20261111T1505Z | A3a | FD001 | 101 | 10.5455 | engine_mean | ok | fail |
| HC-C-A3p-FD001-s101-20261111T1505Z | A3p | FD001 | 101 | — | — | failed | fail |
| HC-C-A3p-FD001-s101-20261111T1508Z | A3p | FD001 | 101 | 10.8951 | engine_mean | ok | fail |
| HC-C-SMOKE-FD001-s101-20261111T1548Z | SMOKE | FD001 | 101 | 62.5931 | engine_mean | smoke | n/a |
| HC-SMOKE-FD001-s101-20261111T1522Z | SMOKE | FD001 | 101 | 62.5931 | engine_mean | smoke | n/a |
| HC-C-A3a-FD001-s101-20261111T1630Z | A3a | FD001 | 101 | — | — | failed | fail |
| HC-C-A3a-FD001-s101-20261112T0005Z | A3a | FD001 | 101 | — | — | failed | fail |
| HC-C-A3a-FD001-s101-20261112T0012Z | A3a | FD001 | 101 | — | — | failed | fail |
| HC-C-A3p-FD001-s101-20261112T0005Z | A3p | FD001 | 101 | 10.8951 | engine_mean | ok | fail |
| HC-C-A3p-FD001-s101-20261112T0012Z | A3p | FD001 | 101 | — | — | failed | fail |
| HC-C-A3p-FD001-s101-20261111T1625Z | A3p | FD001 | 101 | — | — | failed | n/a |
| INIT-DUMP-A3p-FD001-s101 | A3p | FD001 | 101 | — | — | failed | n/a |

## LEGACY 区（旧口径，禁止与现行行聚合）

| run_id | variant | subset | seed | val_rmse | kind | status | 审计 |
|---|---|---|---|---|---|---|---|
| HC-B-F_single_proj_mean_hi0-FD001-s202-20261111T1334Z | F_single_proj_mean_hi0 | FD001 | 202 | 12.4424 | pooled | ok | pass |
| HC-B-F_single_proj_mean_hi0-FD001-s303-20261111T1334Z | F_single_proj_mean_hi0 | FD001 | 303 | 11.8451 | pooled | ok | pass |
| HC-B-F_single_proj_mean_hi0-FD002-s202-20261111T1348Z | F_single_proj_mean_hi0 | FD002 | 202 | 11.5350 | pooled | ok | pass |
| HC-B-F_single_proj_mean_hi0-FD002-s303-20261111T1348Z | F_single_proj_mean_hi0 | FD002 | 303 | 12.1319 | pooled | ok | pass |
| HC-B-F_single_proj_mean_hi0-FD003-s202-20261111T1356Z | F_single_proj_mean_hi0 | FD003 | 202 | 12.7731 | pooled | ok | pass |
| HC-B-F_single_proj_mean_hi0-FD003-s303-20261111T1357Z | F_single_proj_mean_hi0 | FD003 | 303 | 12.8110 | pooled | ok | pass |
| HC-B-F_single_proj_mean_hi0-FD004-s202-20261111T1414Z | F_single_proj_mean_hi0 | FD004 | 202 | 15.9865 | pooled | ok | pass |
| HC-B-F_single_proj_mean_hi0-FD004-s303-20261111T1411Z | F_single_proj_mean_hi0 | FD004 | 303 | 16.9653 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD001-s202-20261111T1331Z | F_single_proj_basic_hi1 | FD001 | 202 | 10.9385 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD001-s303-20261111T1332Z | F_single_proj_basic_hi1 | FD001 | 303 | 10.7585 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD002-s202-20261111T1343Z | F_single_proj_basic_hi1 | FD002 | 202 | 12.4815 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD002-s303-20261111T1343Z | F_single_proj_basic_hi1 | FD002 | 303 | 12.0307 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD003-s202-20261111T1353Z | F_single_proj_basic_hi1 | FD003 | 202 | 11.7861 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD003-s303-20261111T1355Z | F_single_proj_basic_hi1 | FD003 | 303 | 12.3434 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD004-s202-20261111T1409Z | F_single_proj_basic_hi1 | FD004 | 202 | 16.9353 | pooled | ok | pass |
| HC-B-F_single_proj_basic_hi1-FD004-s303-20261111T1407Z | F_single_proj_basic_hi1 | FD004 | 303 | 16.7650 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD001-s202-20261111T1329Z | F_full_basic_hi1 | FD001 | 202 | 11.0119 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD001-s303-20261111T1329Z | F_full_basic_hi1 | FD001 | 303 | 10.9252 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD002-s202-20261111T1338Z | F_full_basic_hi1 | FD002 | 202 | 11.7306 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD002-s303-20261111T1338Z | F_full_basic_hi1 | FD002 | 303 | 12.2846 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD003-s202-20261111T1350Z | F_full_basic_hi1 | FD003 | 202 | 11.4674 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD003-s303-20261111T1350Z | F_full_basic_hi1 | FD003 | 303 | 13.1738 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD004-s202-20261111T1402Z | F_full_basic_hi1 | FD004 | 202 | 16.3040 | pooled | ok | pass |
| HC-B-F_full_basic_hi1-FD004-s303-20261111T1403Z | F_full_basic_hi1 | FD004 | 303 | 16.5887 | pooled | ok | pass |

## 异常与撤回

- `A3p…0105Z`：被终止，`failed`。`A3p…1640Z`、`A3p…1625Z`：无 close 且无进程，`failed`。
- `HC-SMOKE-…1447Z`：维持 `fail`。
- 今晚三次越权写入已记于审计文档 §12.6/§12.9。
