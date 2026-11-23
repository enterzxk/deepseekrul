# 证据目录说明（B 段判定）

## 权威文件

- `B_VERDICT.json` —— **官方判定记录**，由**冻结判据** `paper_c/vote_b.py`
  （commit `257686c0`，sha256 `52552a1b2ab8f3aaff8c0dd26c3d9c50e8a8d4925b7241c1544cadadc2b4d9db`，
  26336 B）生成。服务器路径 `out_b/B_VERDICT.json`。
  结论：`verdict=fail`、`claim_scope=none`、`k_full=0`、`k_sc=0`、`k_min=4`。
- `B_CELLS.json` —— 30 格无条件点估计（`delta = V0 − CADF`）+ 每格 `‖proj‖`/`‖gate‖`。
- `B_REPORT.json` —— 10 组引擎级配对 bootstrap（10000 次）CI / MDE80 / 胜率。

## 字段改名说明（不改判定）

`within_noise` 在判据代码里就是 `not passes`，字段名误导（会被人当噪声检验引用），
后续版本改名为 `not_passes`（`paper_c/vote_b.py` sha256 `3268df03…37eea5a7`，26537 B）。

**`B_VERDICT.json` 未覆盖**：它必须保持「由冻结 blob 生成」这个性质，且 `FINAL_DECISION`
正在从它逐字段生成，中途覆盖会造成读写竞争。

改名是否影响判定，用同一条命令跑新旧两版并逐字段比对，结果：

- 差异总数 **20** = 10 个单元 ×（1 个键删除 + 1 个键新增）；
- 全部差异都是**键名**，**值逐个相同**（`True→True` … `False→False`）；
- `verdict` / `claim_scope` / `k_full` / `k_sc` / `tier_state` / `power` 全部一致。

比对产物：`B_VERDICT_rename_check.json`（新判据跑出的记录）、`B_VERDICT_rename_diff.txt`（逐字段 diff）。

## 已知口径坑（引用前必读）

- 判定记录里的 `within_noise` / `not_passes` = `not passes`，**不是噪声检验**。
  真正的 seed 噪声对照看 `B_CELLS.json` 的 `sd_seed` 与 `|Δ̄|/sd_seed`：
  9/10 组 ≤ 1.63 倍，唯一例外 `two_phase/FD004` 为 11.12 倍。
- `Δ̄` 是 3 个 seed 的均值，`SE(Δ̄) = sd/√3`；**不得**拿单 seed 的 sd 直接比 `Δ̄`。
- `two_phase/FD004` 是 10 个单元里唯一四条全过（`passes=true`，CI [+0.1698,+1.5114]），
  但 `full` 档要求 FD001+FD004 同时、`sc` 档只认 FD001，**两档都放不下它**；
  且 `p_null=0.097`，10 个单元按零假设期望约 1 个这种量级，**只记形态、不归因**。
