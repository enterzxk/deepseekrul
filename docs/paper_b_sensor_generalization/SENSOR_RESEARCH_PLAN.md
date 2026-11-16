# SENSOR_RESEARCH_PLAN（冻结件）

> study_id: `paper_b_sensor_generalization`
> 分支：`research/paper-b-sensor-generalization`，从 `paper-b-contribution-attribution` 的 HEAD `0e7893e8` 切出。
> 本文件在**任何本轮训练启动之前**冻结；之后只允许追加「是否满足」，不允许改阈值、口径或臂定义。

## 0. 一句话

在冻结的强 DSPT 上，用**训练期传感器增强**（整通道 mask / 残差尺度噪声）检验「传感器冗余与噪声」是否还有可复现的独立增益；增强类最好只能到 ENGINEERING-ONLY，GO 留给 E4/E5 的结构类候选。

## 1. 边界（不可越）

- 只用 lab3090 的两张 RTX 3090；A40 不启动新实验。
- 全部 FP32；**不得**用历史 BF16 结果充当 FP32 对照。
- Cycle 沿用冻结的 `pre_norm`；本轮**不改** `min(cycle/250, 1)` 截断逻辑。
- 官方 TEST 不参与任何设计、选轮、选臂或阈值；只在冻结后一次性评价、只作描述。
- 不把相关滑动窗口当独立设备；统计单位是发动机。
- 记录全部实验，不只保存获胜配置；失败格如实登记。

## 2. 代码身份

| 项 | 值 |
|---|---|
| 分支 | `research/paper-b-sensor-generalization` |
| 切出点 | `0e7893e8`（`paper-b-contribution-attribution` HEAD） |
| 训练入口 | `scripts/run_sensor_aug.py` |
| 增强实现 | `paper_b/sensor_aug/{augmentation,noise_stats,evaluation}.py` |
| 改动文件 | `paper_b/models/dspt.py`（`sensor_keep`：注意力 key-mask、pooling 重归一化、encode 清洗）、`paper_b/training/{data,train,metrics}.py`（`cond_seq`、增强钩子、engine-equal 指标） |
| 代码指纹 | `code_fingerprint()` = 7 个源文件 sha256[:12] + `git rev-parse HEAD`，每格写入 `config.json` |

`data.py` 新增 `cond_seq`（每窗每时间步的工况簇）**只**供噪声增强按步取尺度；回归网络仍只吃窗口末端工况。

## 3. 数据与划分

- 数据：`data/CMAPSS`，FD001–FD004，14 传感器，`span=70`，`rul_cap=125`。
- 划分：`split_units(units, split_seed, fold=0, val_frac=0.2)`，**按发动机分组**。
- **发现阶段 `split_seed=17`**；**确认阶段 `split_seed=29` 预先保留，不得提前用于任何模型选择**。
- 模型种子：`101 / 202 / 303`。
- 验证截点：`round(f·n)`，`f ∈ {0.40, 0.55, 0.70, 0.85, 0.95}`，`n` 为该机自身寿命。这是协议常数，不读官方测试 RUL。
- 官方 TEST：冻结前不打开。

## 4. 模型（冻结的强 DSPT）

```
DSPT(num_sensors=14, d_model=64, num_layers=2, dropout=0.2, max_span=70,
     ms_mode='full', sa_mode='basic', hi_source='hf', hi_mono='correct',
     use_cycle=True, use_pad_mask=True, n_cond=6, mode='dual')
```

训练：AdamW，`lr=1e-3`，`wd=1e-4`，`batch=512`，`epochs≤40`，`patience=12`，按验证 engine-level RMSE 选轮，FP32。

## 5. 实验臂（E0–E2；E3 已删除）

| 臂 | 定义 | 参数 | 与旧 study 的映射 |
|---|---|---|---|
| **E0** | 无增强（基线） | — | G0 |
| **E1a** | 训练期整通道 mask，p=0.10 | 同一传感器整窗全时间步一起遮；raw/偏差/EMA 三通道一起；显式 availability mask；至少留 1 路 | G1 |
| **E1b** | 训练期整通道 mask，p=0.20 | 同上 | G2 |
| **E2a** | 训练期残差尺度噪声，α=0.25 | 同机、周期差 1、同工况、两端有限的高频残差，稳健尺度 σ；`s=min(ασ, 0.20·r)`；加在 x 上、特征重建之前 | G3 |
| **E2b** | 同上，α=0.50 | | G4 |

**E3（mask + noise 组合）删除**：两个单通道在旧 study 均未达预注册门槛，组合没有剩余假设。`run_sensor_aug.py` 在 `task_config` 里硬拒绝 `mask_prob>0 and noise_alpha>0`，防止被重新打开。

**重要映射更正**：旧报告里的「噪声通道 G2 +8.73%、FD001 +15.3%」**不是噪声注入臂**，而是 **mask p=0.20（G2）在噪声压力评测下**的鲁棒性收益。旧 study 的噪声注入臂是 G3/G4，其噪声通道收益为 **−1.47% / +2.62%**。本轮按代码与 `stage1_table.md` 为准，不按旧报告的口头标签。

## 6. 旧结论的引用方式（不重复已完成的工作）

| 旧结论（split_seed=7, FD001/FD004, BF16 训练 + FP32 评价） | 本轮处理 |
|---|---|
| mask 无 clean 增益（G1 −0.14%、G2 −0.39% 宏观） | 直接引用，不重跑 clean 结论 |
| mask 在噪声压力下有信号（G2 +8.73%，FD001 +15.3%，FD004 +2.2% CI 跨 0） | **只在 2 个子集上出现过**，四子集未验证 → E1a/E1b 必须在 split 17 上补 4 子集 |
| 噪声注入臂无增益（G3 −1.47%、G4 +2.62%） | 按本轮 spec 复核一次（E2a/E2b），不假设结论可迁移 |
| 缺失通道 +1.7%~+2.0%，远低于 10% | 引用，不作为候选 |

**为什么仍然跑 E1**：唯一还站着的信号来自 mask 在噪声压力下的鲁棒性，而它只测过 FD001/FD004。把它当作「已判死」会把唯一有信号的臂排除掉。

## 7. 判据（冻结）

主指标：**engine-equal RMSE** = `sqrt(mean_u MSE_u)`，其中 `MSE_u` 是发动机 u 全部验证截点的均方误差。另报 MAE、NASA mean、逐机误差、seed sd。

配对：同 `(subset, seed)` 的 `ΔRMSE = arm − E0`（负 = 更好）。95% CI 用**发动机级** bootstrap ≥10000 次（重采样发动机 ID，保留该机全部截点）。

### 7.1 三档

**GO**（仅结构类 E4/E5 可及）四条同时满足：
1. 四子集合并 `ΔRMSE ≤ −0.3 cycle`；
2. 配对 95% CI 不含 0；
3. ≥3/4 子集方向为负；
4. 在 `split_seed=29` 的确认集上复现（5 seed，同预算）。

**ENGINEERING-ONLY**：clean 通道不显著，但某个鲁棒性通道（缺失 k=1/2/3、噪声 β=0.25/0.5/1.0）稳定改善，且 clean 恶化不超过 1%。**E1/E2 即使有效也只能落这一档**——训练期增强不包装成架构创新。

**STOP**：没有任何臂在 discovery 上达到 GO 或 ENGINEERING-ONLY；或确认集不复现。

### 7.2 护栏

- NASA mean 比值 > 1.10 的臂不得进入候选（即使 RMSE 更好）。
- 单子集显著不算数；不得用「最好的 seed」代表一个臂。
- 不得因为某格难看而剔除；只有查实技术缺陷才作废，作废要重跑并写明原因。

## 8. run_id 与产物

```
run_id = {arm}-{subset}-s{seed}-{config_hash8}
experiments/paper_b_sensor_generalization/runs/{run_id}/{subset}/s{seed}/
    config.json        # 完整配置 + code_fingerprint + git HEAD
    status.json        # RUNNING / COMPLETE / FAILED
    best.pt
    val_pred.csv       # 逐发动机、逐截点：unit, cut, y_true, y_pred
    metrics.json       # engine-equal RMSE/MAE/NASA + pooled + seed 信息
    train.log
```

- 断点续跑：`runs/` 已 COMPLETE 的格不再启动；claim 用 `O_EXCL`。
- 每格训练前写 `config.json`；训练后写 `metrics.json`；失败写 `FAILED` 与 traceback。

## 9. 执行顺序

1. **E0**：4 子集 × 3 seed = **12 格**（本阶段交付）。
2. **E1a/E1b + E2a/E2b**：4 × 3 × 4 = **48 格**（下一阶段，需 E0 先落地做配对基线）。
3. 汇总：engine-level 重算 → 配对 ΔRMSE + CI → 按 §7 判定。
4. 若出现候选：在 `split_seed=29` 上确认（5 seed），再一次性评价 TEST（描述性）。

## 10. 不做的事

- 不重跑 E3 组合。
- 不改 `min(cycle/250,1)`、不改 Cycle 注入位置、不改窗口长度。
- 不新增 Attention / 不改骨干（E4/E5 属于后续阶段，且必须先过文献员的新颖性判断）。
- 不用官方 TEST 做任何选择。

---

## 修订记录

> 冻结后只追加，不改上文。每条写清「改了什么、什么时候、为什么、影响谁」。

### 2026-11-15 18:08 · 指纹口径 7 → 8 文件（不涉及阈值或臂定义）

- **改了什么**：`code_fingerprint()` 从 7 个模型/训练文件扩为 8 个，加入 `scripts/run_sensor_aug.py`；manifest/freeze 同时记 `code_fingerprint`（h8）与 `code_fingerprint_legacy7`（h7）。
- **为什么**：臂的 BASE 配置写在 runner 里。只改 runner、不改模型文件时，旧 7 文件指纹不变，记录会显示「与旧 study 同一份代码」，但协议已经变了（BF16 → FP32）。
- **值**：`h8=515eb76fc396`，`h7=3f3a3e68594e`。旧 study 的 h7 是 `90f20d35953d`，差异来自 `train.py` 的 `fp32` 默认值与入口断言，不改模型数学。
- **影响**：§7 阈值、§5 臂定义、§6 引用边界均未变；已开跑的格不受影响（重建 manifest 后 `run_id`/`config_hash` 全部变化）。
- **做法反省**：本条最初直接改在 §2，违反「冻结后只追加」；§2 已还原为 7 文件表述，改动只留在这里。

### 2026-11-15 18:20 · `build` 改写 arm YAML 的问题（**推迟执行**）

- **问题**：`build` 用 `yaml.safe_dump` 重写 `paper_b/sensor_aug/configs/` 下 6 个已跟踪 YAML，导致跑批期间工作树变脏、`git status` 不可信。
- **计划改法**：把输出改到 `experiments/paper_b_sensor_generalization/shared/frozen_configs/`，已跟踪的那份只作冻结快照。
- **为什么推迟**：`scripts/run_sensor_aug.py` 在 h8 指纹里。12 格 E0 正在跑、manifest 记的是 `h8=515eb76fc396`；此刻改 runner 会让分支头的指纹与 manifest 不一致，重现旧 study 那个「记录与实跑不同」的问题。等 E0 落盘后再改，并同步升 h8。
- **过渡做法**：跑批期间若 `configs/` 显示 diff，`git checkout -- paper_b/sensor_aug/configs/` 还原即可，不影响 manifest。

### 2026-11-15 18:20 · §8 目录结构更正

- **改了什么**：`runs/{run_id}/` → `runs/{run_id}/{subset}/s{seed}/`。
- **为什么**：原文档少写两层，军师与郑总按文档找不到文件。

### 2026-11-15 20:45 · 主指标口径更正（§7 的落地定义）

- **问题**：§7 写的主指标是 engine-equal `sqrt(mean_u MSE_u)`，但实测 `metrics.json` 里 `val_engine_equal_rmse` 是 **pooled 的副本**（FD004/s303 = 13.989556），真正的「逐机 RMSE 再平均」是 `val_engine_mean_roots_rmse`（11.900607）。且在平衡设计（每机正好 5 个截点）下 `sqrt(mean_u MSE_u)` 恒等于 pooled，等于没有定义出独立口径。
- **决定**：主指标改为 **`val_engine_mean_roots_rmse` = `mean_u RMSE_u`**；`val_engine_equal_rmse` / `val_pooled_rmse` / `rmse` 降为辅助，只作交叉核对。
- **理由**：与前几轮 Paper B 口径一致；数值更高更保守；不依赖各机截点数是否均衡。
- **影响**：§7 的阈值形式（−0.3 cycle、95% CI、≥3/4 同号）不变，只是 Δ 改用新口径计算；E0 基线从 `val_pred.csv` 重算即可；尚无任何臂间决策，不影响已作结论。
- **不改代码**：`metrics.py` 在 h8 里，本轮不动；只改「读哪个字段」。

### 2026-11-15 20:45 · runner 改动时机收紧（覆盖 18:20 那条）

- **决定**：**整个发现阶段不改 `scripts/run_sensor_aug.py`**，E1a/E1b/E2a/E2b 与 E0 共用 `h8=515eb76fc396`。
- **理由**：runner 在 h8 里，中途改会让 E0 与后续臂的配对基线不是同一份代码；而指纹机制分不清「只改输出路径」和「改了数学」。为工作树是否干净把整轮拆成两个指纹，不值得。
- **代价**：`build` 之后执行一次 `git checkout -- paper_b/sensor_aug/configs/`，已写进 `FAIR_COMPARISON_AUDIT.md` §7b「已知操作」。

### 2026-11-16 00:25 · §7.1 ENGINEERING-ONLY 前件澄清（E1a 暴露的定义缺口）

- **缺口**：§7.1 原文是「clean 通道**不显著**，但某个鲁棒性通道稳定改善，且 clean 恶化不超过 1%」。E1a 的 clean 是**显著改善**（合并 −0.545，95% CI [−0.819, −0.271]，4/4 同号），字面上不满足「不显著」，于是四个档位一个都不落。
- **澄清**：ENGINEERING-ONLY 的前件改为「**clean 不恶化**（四子集等权恶化 ≤1%）**且**至少一个鲁棒性通道稳定改善（CI 不含 0、≥3/4 同号）」。clean 显著改善不再被排除在外。
- **不改的东西**：GO 门槛（−0.3 cycle、CI、≥3/4 同号、split 29 复现）、鲁棒性判据、以及「增强类不达 GO」的上限，全部不变。
- **时点与对称性**：此澄清在看到 E1a（当时唯一完整臂）之后做出，**必须在报告里披露**；它只改标签归属、不改任何阈值，且对四臂对称适用（包括后续可能失败的臂）。
- **报告要求**：旧 study 的 clean 结论（五组全负，−2.45%~−0.14%）在新协议下**不复现**（E1a 为 −0.545，变好）。报告必须写「不复现」，不得写「重跑确认」，并列出三条协议差异（`fp32=false`、`split_seed=7`、仅 FD001/FD004）。

### 2026-11-16 00:32 · 旧/新口径对照补充（「不复现」的第四个来源）

- **补充**：旧 study 的 clean 增益按 **pooled** RMSE 计算（`stage1_table.md` 的表头写的是 `sqrt(mean_u MSE_u)`），本轮主指标是 **mean-of-roots**。两者对发动机等权的处理不同（Jensen：`mean(sqrt) ≤ sqrt(mean)`），对尾部发动机的敏感度也不同。
- **影响**：E1a 与旧 G1 的方向翻转**至少含口径变化这一项**，不能全部归因于协议（FP32/split/子集数）。报告必须把四条差异一起列出：`fp32`、`split_seed`、子集数、**指标口径**。
- **不改**：本轮主指标仍为 mean-of-roots（§7 修订已定），旧数字不回算成新口径——它们只作历史参照。

### 2026-11-16 00:45 · 口径差异已量化（覆盖 00:32 条的推测）

- **实测**：同一批 E1a 预测换聚合口径，合并 Δ 从 **−0.545（mean-of-roots）** 变为 **−0.432（engine-equal）**，差 **0.113 周期**。
- **结论**：口径差异**只改幅度、不改符号**，不足解释与旧 study 的方向翻转。00:32 条里「至少含口径变化这一项」的推测，应改为「口径贡献约 0.11 周期，不是主因」；翻转仍需记在 `fp32` / `split_seed` / 子集数 / 实现差异上。
- **报告写法**：四条并列，并注明口径差异的量级。

### 2026-11-16 00:45 · 档位与贡献类型分离（经理裁定，覆盖 00:25 条的标签部分）

- **裁定**：E1a 在发现阶段四条统计判据全部通过，因此它是**确认候选**，进入 `split_seed=29`；**最终档位由确认集决定**，现在不写 GO，也不写最终 ENGINEERING-ONLY。
- **分离**：档位反映证据强度；「增强类不包装成架构创新」是**贡献类型**的约束，不等于「有效也不能算」。
- **最终表述**：若确认集复现，写「经确认的工程增益，不构成方法新颖性」；若不复现，写「发现阶段的效应在确认集上未复现」。
- **时点**：本裁定在确认集开跑**之前**做出，不动任何阈值、统计判据或臂定义。

### 2026-11-16 08:05 · 阶段 2 的两条报告规则（不设新阈值）

1. **ENGINEERING-ONLY 没有效应量下限**：四条臂字面上都能进这一档，但 seed 级 sd 从 0.040（E1a）到 0.326（E2b）差 8 倍。报告必须**按强度排序**、逐臂给出 seed sd 与去 FD001 后的合并值，**不得写成四臂等效**。这是报告规则，不改判据。
2. **旧 study 的 G2 只能写「结构性复现」，不能写「逐点复现」**：本轮收益随 β 单调、FD001 最大、FD004 只有 1–2%（与旧「FD004 CI 跨 0」一致），但旧 study 的 β 在现有文件里没有记录，没有任何一个 β 同时对上 +8.73% 与 +15.3%。措辞必须写成「方向与形状复现，逐点数值不可比」。

### 2026-11-16 08:05 · E1b 触发 §7.2 护栏，不作候选（结果，非规则）

- E1b（mask p=0.20）clean 合并 −0.689，是四臂中最大；但 `val_engine_equal_nasa_mean` 比值 **1.141 > 1.10**，来源单一：FD004 上 RMSE 改善 0.553 周期、NASA 均值恶化 77.5%，单格最差 **3.29 倍**。
- 按 §7.2「即使 RMSE 更好也不得进入候选」，E1b 出局。这不是协议变更，是既有护栏第一次生效。
- 报告价值：这是「平均误差变好、尾部风险变差」的实例（少数严重滞后），应作为机制结果写，而不是只当一条失败臂。

### 2026-11-16 08:40 · **撤回上一条**：E1b 并未触发护栏；§7.2 的比值定义要写死

- **错在哪**：08:05 那条把护栏算成「逐格 NASA 比值再平均」（E1b 1.141），而 §7.2 写的是「NASA **mean** 的比值」——应**先把指标聚合、再相除**。按正确读法四臂比值分别是 E1a 0.881、E1b 0.979、E2a 0.941、E2b 0.941，**四臂都不触发护栏**。
- **1.141 的来源**：被一格的分母主导。E0 自己在 FD004/s303 的 NASA 是 4.00，兄弟 seed 是 17.4 / 33.9（差 4–8 倍），所以逐格比值 3.292 主要是除到了一个异常好的基线。绝对值上 E1b 在 FD004 的 NASA 均值 20.82 对 E0 18.43，是 +13.0% 的温和恶化，不是「爆炸」。
- **连带撤回**：「mask 越强 → 平均精度越好、尾部风险越差」的剂量反应说法作废。按正确口径 E1b 的 NASA 比值 0.979 也是变好；只有 FD004 一个子集 1.130 略超线，且由单个 seed 驱动，撑不起机制叙事。
- **E1b 不作候选的理由改为种子稳定性**：seed 级 sd **0.215**（E1a 0.040，差 5 倍），三个 seed 的合并均值 −0.789 / −0.443 / −0.836，跨度 0.39。不是护栏问题。
- **§7.2 定义写死**：`nasa_mean` 比值 = 先按臂聚合成 mean、再相除；不得用逐格比值再平均。这是定义澄清，不改阈值 1.10。
- **另一条要记住的**：E0 自己在 FD004/s303 的 NASA 是 4.00、兄弟 seed 17.4/33.9——NASA 均值是重尾统计量，单 seed 就能主导比值。以后凡报 NASA 比值，必须同时给绝对值和 per-seed 分布。



