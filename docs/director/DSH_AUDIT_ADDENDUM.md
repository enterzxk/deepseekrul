# DSH 审查补遗（第二轮，2026-10-22T17:40–18:15Z）

本文件是 `docs/director/DSH_PROJECT_AUDIT.md` 的**追加**证据，不改写正文（正文由并行会话持续更新，见 §0）。
作者：`codex-rul-native-goal-20261022`（goal `goal-8f36b594-…`）。

## 0. 并发上下文

同一工作区有第二个 DSH 会话 `desktop-ptc-deepseek-rul`（goal `goal-96c7053b-…`），它在 01:35 之后继续修改
`scripts/run_candidate.py`、`src/candidates/model.py` 并补 FD001 单元，同时改写 `DSH_PROJECT_AUDIT.md` 与
`results/dsh/control.json`。本补遗只记录**代码/SHA/原始 CSV 可验证**的事实，避免双写冲突。

## 1. `gate_mode` 是死参数，固定门控消融是假的

- 当前 `src/candidates/model.py`（SHA256 `8331a355f2b927486d90984aa4c1b28b5382fb01c9aedb6a256a6ff792df82b0`）
  新增 `gate_mode ∈ {evidence, ones, half, const}`，但全文件只有构造函数的 3 行（100/118/119/120）引用；
  **`forward()` 从未读取 `self.gate_mode`**。
- 因此 `RACDR_fixed1`（ones）、`RACDR_half`（half）、`RACDR_const`（const）与 `RACDR` 完全等价，
  同 seed 会给出逐位相同结果，**不是门控消融**。
- 真正的“可学习固定传感器门控”尚未实现；要做必须在 forward 里用 gate_mode 覆盖
  `weight = sigmoid(reliability(evidence))`。
- Level B 只用 `RACDR` 默认 `evidence`，所以复验不受此影响；但任何“固定门控 vs 证据门控”的表格现在都无效。

**数值证据（2026-10-22T18:05Z，服务器 metrics.json）**：

| 单元 | val RMSE | matched RMSE | 对照 |
|---|---|---|---|
| `H2_fixed1/FD002/RACDR_fixed1_s2028` | 12.412820 | 14.893930 | 与 `M4_racdr/FD002/RACDR_s2028` **逐位相同** |
| `H2_half/FD002/RACDR_half_s2027` | 12.543752 | 14.962552 | 与 `M4_racdr/FD002/RACDR_s2027` **逐位相同** |
| `M4_ckpt/FD002/RACDR_s2028` | 12.412820 | 14.893930 | 与 M4 s2028 相同（checkpoint 复跑一致） |

即：所谓“fixed1/half 门控消融”就是同一模型重跑，不能用于归因。

## 2. 单工况评估路径的训练/评估特征偏移（已在本 worker 的冻结版修正）

新代码为 FD001 增加 `build_windows_abs` / `build_windows_matched`，但两处自行重算 6 维证据：

| 证据 | 训练 `build_windows` | 新 `build_windows_abs` | 新 `build_windows_matched` |
|---|---|---|---|
| e1 | min(1, early_n/5) | ref_count（等价） | ref_count（等价） |
| e2 | 1 | 1 | 1 |
| e3 | exp(−mean_age/30) | **常量 exp(−2/30)** | exp(−mean_age/30) |
| e4 | exp(−dev) | exp(−dev) | exp(−dev) |
| e5 | 1/(1+ref_std) | 同 | 同 |
| e6 | jitter | **trend** | jitter |

- 训练路径从未改变 → FD001 的 M3/M4 在 Protocol C / matched 评估时输入分布与训练不一致；M0（tf，只用 x）不受影响。
  这会让 FD001 候选被系统性低估，**不能直接与 M0 比**。
- 另：matched 在单工况路径用一次性 `default_rng(cutoff_seed)` 顺序抽样，而 FD002 用 `seed+unit` 逐机抽样，
  跨子集 cutoffs 不可比。
- 修正版：`scripts/frozen/run_candidate_v3_8fce61ff.py`（本地 `results/dsh/server_snapshot/run_candidate_v3.py`），
  抽取 `_single_evidence` 复用训练定义，matched 改为逐机 `seed+unit`；已 `py_compile` 并在 FD001 上跑通。
- 旧 FD001 单元保留在 `experiments/order03/*/FD001/`，标“pre-fix，仅定位”，不删不覆盖。

## 3. 版本漂移时间线（同一文件、不同 SHA）

| 时间（+0800） | run_candidate.py | model.py | 产物 |
|---|---|---|---|
| ~00:47 | `6715b40ab086800fcca8169a1a9d2c03e24a2d85582b5b00e088bcbf0dc5af75` | `95a1c24c8af044e3860647cca81057b6bd1d81be84d5a68ac6ab351b9e56fff9` | FD002 12 单元 |
| ~01:20 | `c0bdf27347ea48a44a403b2d4febdf4f2041318e311f773e7c0b1fd511768762` | 同上 | 加 RACDR_noid/noref（伪消融） |
| ~01:35 | `2393dc8e70d0b85a2e93b48ed93ae7ac97dd0f442fba6d439f4f25f411c47e42` | 未记录 | FD001 首次出数 |
| ~01:45 | `9074d6ca615aa7dc905990a62ccf12052a920379258509a940dcfb4f75c4c21a` | `8331a355…` | FD001 helpers + checkpoint + dead gate_mode |

结论：FD002 与 FD001 的旧单元**跨版本不可比**；只有 Level B（独立冻结目录）内部可比。

## 4. 有效单元判定（截至 18:10Z）

- **作废/重复**：`M4c_noid` 三个 FD002 + 三个 FD001 与 `M4_racdr` 同配置，FD002 指标逐位相同
  （14.962552/12.543752 等）。
- **FD001 的 M4b** 与 M4 同配置：`condition_aware=False` 时 `--ref-mode pop/cond` 分支不生效，
  两者 2027/2028 指标逐位相同；只有 FD002 的 M4b 是真正的“总体参考”消融。
- **仍然缺失**：FD001 M4 s2029、FD001 M3 s2029（旧代码崩溃后未补），M1_global 三 seed 状态待核。

## 5. 对“门控 × 工况异质性”交互结论的约束

并行会话在 `DSH_PROJECT_AUDIT.md` §7 提出“FD001 门控变差、FD002 门控变好”的交互解释。本补遗的支持与限制：

- 方向在使用旧 FD001 数据时成立（M3 12.215 vs M4 12.879，n=2），但：
  1. FD001 的 M3/M4 评估路径有 §2 的特征偏移，M3/M4 同时受影响，二者比较相对公平，但与 M0 的比较不公平；
  2. n=2、无配对 bootstrap，CI 未知（本 worker 对 FD001 旧数据算过：M4−M3 = +0.661 [−1.740, +3.256]，跨零）；
  3. 该交互是在看到 FD001 结果后提出的，属于**事后假设**，不能当作预登记结论。
- 因此正确表述是“**开发集上出现方向相反的交互信号，样本不足，需 Level B 复验**”，而不是“门控收益依赖工况异质性”的定论。

## 6. 冻结与复验（Level B）

- `results/dsh/freeze.json`：runner `8fce61ff…`、model `8331a355…`、baselines `32d10b2f…`、
  数据 SHA256、环境、协议（`split_seed=2027`、seeds 2029/2030/2031、FD001+FD002、M0_tf/M4_racdr、30 epoch）、
  启动器与输出路径。
- 披露：代码与超参在 `split_seed=7` 上选定；本复验是预登记复验，不是独立盲测；结果不得回改候选。
- 与主控“补充预算种子 2027/2028/2029”的口径差异：Level B 预登记冻结的是 2029/2030/2031，
  本 worker 按预登记执行（3 个种子、12 个单元，仍在 18 单元预算内），**请主控在五小时审阅时确认**。

### 6.1 复验结果（12/12，2026-10-22T18:54Z）

| 子集 | matched（M0 → M4） | ΔRMSE [95% CI] | 跨零 | late（M0→M4） | severe（M0→M4） |
|---|---|---|---|---|---|
| FD001 | 9.876 → 8.735 | −1.157 [−3.420, +1.391] | 是 | 0.550 → 0.367 | 0.233 → 0.117 |
| FD002 | 12.859 → 9.762 | −3.092 [−5.465, −1.000] | **否** | 0.564 → 0.455 | 0.231 → 0.096 |

- Protocol C：FD001 −1.006 [−3.180, +1.593]（跨零）；FD002 −1.889 [−3.311, −0.456]（显著）。
- Protocol B：FD001 −2.549 [−4.077, −0.907]；FD002 −2.925 [−5.013, −0.809]（均显著）。
- 门禁：matched 优于 tf（两子集）、FD001 `[0,25)` severe ≤1.1×tf（三协议）、late ≤0.52 全部通过。
- 每条 `metrics.json` 内嵌 `code_sha256=8fce61ff…`；`best_model.pt` 已保存。
- 关键限制：FD002 的 tf 基线 `best_epoch` 仅 2–6（早停偏早），可能低估基线；这是下一阶段第一优先复核项。
  完整结论与下一阶段建议见 `docs/director/DSH_STAGE_CONCLUSION.md`。
