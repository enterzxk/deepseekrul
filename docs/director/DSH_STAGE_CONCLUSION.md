# C-MAPSS RUL 当前阶段结论（DSH 证据闭环交付）

日期：2026-10-22
作者：DSH 原生目标 worker（会话 `codex-rul-native-goal-20261022`，goal `goal-8f36b594…`）
证据路径：`results/dsh/order03/REPORT.md`、`levelb_aggregate.json`、`results/dsh/freeze.json`、`docs/director/DSH_PROJECT_AUDIT.md` + `DSH_AUDIT_ADDENDUM.md`

---

## 1. 一句话结论

在**唯一一次预登记复验**（`split_seed=2027`、seeds 2029/2030/2031、冻结代码 `8fce61ff…`）中，
RACDR（M4：条件匹配收缩参考 + 6 维证据门控）相对**同特征强 Transformer**（M0），
在 FD002 的 matched-cutoff 上 **RMSE −3.09（95% CI [−5.47, −1.00]，不跨零）**，
FD001 为 **−1.16（CI [−3.42, +1.39]，跨零）**；两子集的 Protocol B 都显著，
FD002 的 Protocol C 也显著。**晚报风险同步改善**：severe-late 从 0.231→0.096（FD002）、0.233→0.117（FD001），
late_fraction 从 0.564→0.455、0.550→0.367，均低于 0.52 门禁。

因此：**“候选在六工况 FD002 上有可复现的误差优势，且在本次复验中晚报更少”得到支持；
“候选在单工况 FD001 上显著优于强 Transformer”未得到支持（方向为正，区间跨零）。**
`M4 − M3`（相对无门控条件参考）在开发划分上不显著，机制归因仍不成立。

## 2. 支持与不支持的声明

| 声明 | 证据 | 判定 |
|---|---|---|
| 候选在 FD002 matched-cutoff 优于同特征 Transformer | Level B Δ=−3.09 [−5.47,−1.00]；开发划分 Δ=−1.98 [−4.05,+0.29] | **支持（仅 FD002/本次复验）** |
| 候选在 FD001 matched-cutoff 显著优于强 Transformer | Level B Δ=−1.16 [−3.42,+1.39] | **不支持（方向正、不显著）** |
| 候选降低晚报风险 | severe 0.096 vs 0.231（FD002）、0.117 vs 0.233（FD001）；late 均 <0.52 | **支持（本次复验，n=3）** |
| 收益来自“证据门控” | 开发划分 M4−M3 = −0.98 [−2.10,+0.13]，不显著；“fixed/half 门控”实现是死参数 | **不支持归因** |
| Sensor-ID / Proj_raw / Proj_ref 是创新点 | `use_sensor_id=False`；proj_* 是死参数 704 个 | **不成立，已更正** |
| e4 是工况中心距离 | 实为传感器空间兼容性代理 | **不成立，已改名** |
| FD001/FD002 官方 test 可用 | 代码从不加载 test；FD003/FD004 test+RUL `chmod 000` | **未使用，封存** |

## 3. 复验的门禁结果

| 门禁 | 结果 |
|---|---|
| ① matched RMSE 优于 tf | FD001 8.735 vs 9.876、FD002 9.762 vs 12.859 → **通过** |
| ② FD001 `[0,25)` severe-late ≤ 1.1×tf | matched 0.000 vs 0.000；Protocol B 0.000 vs 0.0196；Protocol C 0.000 vs 0.000 → **通过** |
| late_fraction ≤ 0.52（既有门禁） | FD001 0.367、FD002 0.455 → **通过** |

但“门禁通过”不等于“所有口径显著”：FD001 matched/Protocol C 的 CI 跨零，必须与门禁结果同时报告。

## 4. 可复现性现状（诚实标注）

- **Level B（12 单元）可复现**：冻结 runner `scripts/frozen/run_candidate_v3_8fce61ff.py`（执行名 `levelb_runner_8fce61ff.py`，字节相同）、
  model `8331a355…`、baselines `32d10b2f…`、数据 SHA256、每条 `metrics.json` 内嵌 `code_sha256` / `command` / `dataset`、
  `best_model.pt` 已保存。
- **ORDER_03 旧矩阵（FD002 12 单元、FD001 旧单元）不可精确重跑**：产生 FD002 的 `run_candidate.py` `6715b40…` 已被覆盖，
  服务器无 git；随后代码又经历 `c0bdf27 → 2393dc8e → 9074d6ca` 多次原地修改；旧 `metrics.json` 无代码哈希。
  这些单元的数字**可以复算**（原始 CSV 保留），但**不能重跑复现**。
- 官方 test 从未加载；FD003/FD004 封存未动。

## 5. 已知限制（影响解释，不推翻 Level B）

1. **tf 基线早停偏早**：FD002 三个种子的 `best_epoch` 为 5/6/2（patience=10），基线可能在更晚 epoch 更好；
   开发划分上也有同样现象（3/3/5）。这会放大候选优势，下一阶段必须用更长 patience 或固定 epoch 复核。
2. **不是独立盲测**：Level B 的验证发动机与开发划分重合 FD001 7/20、FD002 6/52；代码/超参在 split 7 上选定。
3. **n=3 seeds、20/52 台发动机**；CI 已按发动机聚类，但仍属小样本。
4. **M1_global / FD001 开发矩阵有缺口**：M1 仅 s2027 完成、FD001 M4 开发划分仅 2 seeds（s2029 缺）；不影响 Level B 结论。
5. **机制归因缺失**：真正实现“可学习固定传感器门控”需要在 forward 中接线 gate_mode；当前源码里的 fixed/half/const 是死参数，
   其数值与 RACDR 逐位相同，不能用于归因。
6. **并发双会话**：另一会话 `desktop-ptc-deepseek-rul` 在本工作区并行改代码/建队列；本结论只采用冻结版 Level B 与原始 CSV 可核数据。

## 6. 下一阶段建议（按优先级）

1. **基线公平性复核（最高优先）**：固定 epoch（如 30）或 patience≥15，在 split 7 与 2027 上重跑 M0 tf 与 M4 各 3 seeds；
   若 tf 优势消失，则重新评估 §1 的 FD002 结论。这是当前最可能推翻结论的单一因素。
2. **真实门控消融**：在 `CandidateModel.forward` 中把 `gate_mode` 接到 `weight`（ones / half / const / evidence），
   先做单测不做的真实训练对照；同时删除 proj_raw/proj_ref 死参数，使参数量对比干净。
3. **回归到单一控制器 + provenance 强制**：一个 launcher、队列文件、每单元写入 `code_sha256/command/split_hash`；
   禁止原地改源码；失败单元标记 `failed` 而不是无限重试；删除 M4c_noid / H2_fixed1 / H2_half 等伪消融。
4. **若继续科学问题**：把“门控 × 工况异质性”的交互作为**新预注册**假设，用 FD001/FD002 之外的开发切分验证；
   在未做之前，只能写“开发集信号”，不能写定论。
5. FD003/FD004 与官方 test 继续保持封存，直到 Director 明确批准。

## 7. 停止条件声明

本阶段不再扩大搜索：不新增 Sensor-ID、图模块、对比学习或大规模超参搜索；不重复跑已作废/重复单元；
不为了“宣布成功”更换种子或 cutoff。若下一阶段基线公平性复核推翻 FD002 结论，按负结果记录并停止扩展。
