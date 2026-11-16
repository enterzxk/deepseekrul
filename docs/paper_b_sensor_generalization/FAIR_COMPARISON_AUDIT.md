# FAIR_COMPARISON_AUDIT：旧 sensor_aug 结论的代码溯源

> 2026-11-15。只读核对，不改旧 study 的任何产物。
> 问题：从旧 study 取回的模型侧实现，是否就是产出旧 G1–G4 数字的那一份？§6「直接引用旧 clean 结论」是否成立？

## 1. 结论（三句）

1. **是同一份代码，有逐格直接记录。** 36 份 `metadata.json` 的 `git_head` **全部**是 `30138074`（2026-11-15 02:11:26），该 commit 的 7 文件指纹 = **`90f20d35953d`**，与取回的代码一致。不是靠时间戳推的。
2. **manifest 里记的 `fda45915b492` 是建 manifest 时（约 01:40–01:50）未提交工作树的指纹**，早于 `30138074`，**不能代表跑数时的代码**。这是记录缺陷（指纹冻结太早、之后没重算），不是代码版本混杂，**不需要重跑**。
3. **§6 可以引用，但必须带三条边界**：`fp32=false`（BF16 训练）、`split_seed=7`、只有 FD001/FD004 两个子集。它不是本轮的 FP32/四子集对照。

## 2. 取回了什么

| 文件 | sha256[:16] | 字节 |
|---|---|---:|
| `paper_b/models/dspt.py` | `7eedabf768307db2` | 18748 |
| `paper_b/training/train.py` | `e9fbcc5b386edf2d` | 17856 |
| `paper_b/training/data.py` | `c58a3db0d419f4f6` | 15117 |
| `paper_b/training/metrics.py` | `7d8bb2add76d8a91` | 4540 |
| `paper_b/sensor_aug/augmentation.py` | `67d755f588b494c1` | 5719 |
| `paper_b/sensor_aug/noise_stats.py` | `1072d2ec657e3e70` | 6526 |
| `paper_b/sensor_aug/evaluation.py` | `643f1dd40770244b` | 12100 |
| **合并（旧 study 的 `code_fingerprint` 口径）** | **`90f20d35953d`** | |

旧 study 仓库工作树与新分支 worktree 的这 7 个文件**逐字节相同**（上表两边一致）。

## 3a. 每格自己的版本记录（直接证据）

`config.json` **没有** `code_fingerprint` / `git_head` 字段（0/36），所以最初只能在 manifest 与时间戳之间推。真正逐格的记录在 `metadata.json`：

```
$ grep -h '"git_head"' runs/*/*/*/metadata.json | sort | uniq -c
     36  "git_head": "301380742cbc08f8d331ab315b1be70cd429ce15"
```

| 项 | 值 |
|---|---|
| 36 格记录的 commit | `30138074`（`sensor-aug: add test summary and the RESULTS_ZH composer`，02:11:26） |
| 该 commit 的 7 文件指纹 | **`90f20d35953d`** |
| 该 commit 之后动过这 7 个文件的提交 | 仅 `46934d5d`(05:40)、`2cc26ebb`(05:41)，两者指纹仍为 `90f20d35953d` |

所以：**36 格 = 单一 commit = 单一指纹 = 取回的那份代码**。时间戳推断只是旁证，不再是主证据。

## 3. 指纹为什么对不上

| 时点 | 记录 | 实际 |
|---|---|---|
| 建 manifest（约 01:40–01:50） | `code_fingerprint=fda45915b492`，`git_head=51307439` | 工作树已有 `paper_b/sensor_aug/`，但 `51307439` 里没有该目录（`git show` 报 MISSING）→ 代码当时**未提交** |
| 01:52:43 `ff262eb1` | — | 首次提交 harness，**`sensor_keep` 在这一提交引入 `models/dspt.py`**；该提交指纹 `4eb45201ac56` |
| 01:58:48 `5fc87694` | — | evaluation 调整，指纹 `87ff6d495804` |
| 02:06:25 `3ab62ead` | — | `train.py` 验证改为 FP32 评测（影响选轮），指纹 `d23f1f837ed6` |
| 02:07:10 `2f663291` | — | 只改 `reporting.py` |
| 02:08:13 `8734e126` | — | `data.py` 测试窗带 cutoff cycle，指纹 **`90f20d35953d`** |
| 03:31–05:32 | — | **36 格实际训练**（首格 config.json 03:31，末格 05:32） |
| 05:40/05:41 | `46934d5d` / `2cc26ebb` | 报告与文档，指纹仍 `90f20d35953d` |

`fda45915b492` 不对应任何一个 commit：它是「建 manifest 那一刻的工作树」，而那一刻的代码**从未训练过任何一格**（第一格 03:31 才启动）。

## 4. 36 格是否同一版本

§3a 的 `metadata.json` 已给出直接记录（36/36 同一 commit）。本节保留时间戳作为旁证。

是。启动时间 03:31–05:32，全部晚于 `8734e126`（02:08:13），因此都加载 `90f20d35953d`。
- 02:06 的 `train.py` 改动（验证 FP32）在**所有格之前**生效，不存在「前半格用旧选轮、后半格用新选轮」。
- 02:08 的 `data.py` 改动只影响 test-role 窗口，不影响 Stage 1 的 validation 数字。

**若任何一格在 02:08 之前启动，本节结论作废。** 实测最早 03:31，不成立。

## 5. 旧数字的协议边界（引用时必须写）

从每格 `config.json` 实测：

```
"fp32": false            ← BF16 训练（验证在 02:06 之后是 FP32 评测）
"split_seed": 7
"subset": FD001 / FD004  ← 只有两个子集
"rul_cap": 125, "cuts": [0.4,0.55,0.7,0.85,0.95]
```

所以旧 clean 结论（五组全负、−2.45%~−0.14%）可以说「在这个协议下没有 clean 增益」，**不能**说「在 FP32、四子集、split 17 下也没有」。这正是本轮 E1a/E1b 要重跑的理由——不是因为代码不可信，而是因为协议不同。

## 6. 我这次移植漏了什么（根因）

`models/dspt.py` 没被搬过来，原因不是「模型侧改动没进 git」——它在 `ff262eb1` 里，一直在 git 里。
根因是我取 diff 时用了 `git diff --stat 51307439 f237e80b | tail -20`，**只看到末尾 20 行**，`paper_b/models/dspt.py` 在列表上方被截掉了；随后按可见文件复制，漏了 models。

已修正：`paper_b/models/dspt.py` 从旧 study 取回（含 `sensor_keep` 的 attention key-mask、encode 清洗、pooling 重归一化），新分支 7 文件指纹与旧 study 一致（`90f20d35953d`）。

**教训**：diff 列表不得用 `tail` 截断后当作完整清单；搬运代码必须用合并指纹验收，而不是「文件看起来齐了」。

## 7. 对 §6 的最终裁定

| 原表述 | 裁定 |
|---|---|
| 「mask 无 clean 增益，直接引用、不重跑」 | **成立**，但必须附 §5 三条边界（BF16 / split 7 / 两子集） |
| 「噪声注入臂无增益（G3/G4）」 | 同上，引用时注明协议 |
| 「G2 的 +8.73% 是噪声鲁棒性、不是噪声注入」 | 成立，见 `SENSOR_RESEARCH_PLAN.md` §5 |
| E1a/E1b 的性质 | **仍为重做**（协议不同），不是「复核同一协议」 |

## 7a. 本轮追加记录（2026-11-15 18:05）

1. **BF16 事故**：E0 的前几格跑在 `fp32=false`（`DEFAULTS` 默认值）上，违反冻结件。根因：build 路径从未设置 `fp32`。已修：`task_config` 写死 `fp32=True`；`build` 拒绝写入任何 `fp32 is not True` 的 manifest；`train_one` 入口对非 True 抛错（`allow_bf16=True` 只作历史复现开关）。
2. **作废格未留档**：那几格被 `rm -rf runs/*` 删除，没有按约定 `mv` 到 `_invalid_bf16/`，终值不可审计，只剩日志逐轮轨迹。这是操作错误，记录在案；它们本就不进任何表。
3. **指纹口径变更**：`code_fingerprint()` 从 7 文件扩到 8 文件（加入 `scripts/run_sensor_aug.py`——臂的 BASE 定义在那里）。新 `h8=515eb76fc396`，`h7=3f3a3e68594e`；旧 study 的 `h7` 是 `90f20d35953d`，差异来自 `train.py` 的 fp32 默认值与入口断言，不改模型数学。

## 7b. 代码身份的读法（以后核对照这两条）

1. **判代码身份看 `h8`，不看 `git_head` 是否等于分支头。** `git_head` 在 manifest 里只是建 manifest 那一刻的 HEAD；之后任何文档提交都会让分支头前进，但 `h8` 不变。`e931cb93` 就是这种提交：head 从 `ec3289a7` 变成 `e931cb93`，`h8` 仍是 `515eb76fc396`。
2. **同一批里各格 `metadata.json` 的 `git_head` 可以不同，但 `h8` 必须相同。** 12 格跑了约 65 分钟，期间若有文档提交，后启动的格会记到更新的 HEAD。这是正常的；判「是不是同一份代码」只认 `h8`。

**已知操作（跑批期间）**：`build` 会重写 `paper_b/sensor_aug/configs/` 下 6 个已跟踪 YAML，工作树会显示 diff。跑完 `git checkout -- paper_b/sensor_aug/configs/` 还原即可，不影响 manifest 与指纹。

**已知操作（每批收尾）**：`train.py` 在 h8 里，不能中途加字段，所以逐格 `metadata.json` 只记 `git_head`、不记 `code_fingerprint`。每批跑完后执行

```
python scripts/backfill_provenance.py --manifest <manifest.json>
```

把 `code_fingerprint` / `legacy7` / `manifest_git_head` / `run_git_head` / `config_sha256` 写进每个 COMPLETE 格的 `provenance.json`（E0 的 12 格已补）。**逐格指纹只能从 `provenance.json` 断言，不要从 `metadata.json` 断言。**

**但 `provenance.json` 里是 manifest 声明的值，不是运行期独立重算的值。** 它只能证明「每格都从同一个 manifest 启动」。闭环需要独立复算：

```
git worktree add --detach <tmp> ec3289a7   # E0 那批的 git_head
git worktree add --detach <tmp2> 3468954e  # E1/E2 那批的 git_head
# 在两个干净 checkout 上各重算 8 文件指纹
```

2026-11-16 实测：`ec3289a7 → h8=515eb76fc396`、`3468954e → h8=515eb76fc396`。所以报告只能写「48 格共用同一 manifest，其声明的 h8 在 E0 与 E1/E2 两个 `git_head` 上分别复算均为 `515eb76fc396`」，**不能**写「逐格重算指纹一致」。

**独立性怎么数（2026-11-16 订正）**：真正的独立实现是**两个**——① detached worktree 干净 checkout 后对文件系统取字节；② `git show <rev>:<path>` 直接从对象库取字节。仓库里的 `scripts/audit_fingerprint.py` 是 ② 的可复跑入口，**不是第三条路径**；服务器上 `/tmp` 的同方法脚本也不算。把同一方法写在两处不能数成两次独立验证——这正是这份审计要防的错误。

```
python scripts/audit_fingerprint.py <repo> ec3289a7 3468954e
ec3289a7  H8=515eb76fc396 H7=3f3a3e68594e
3468954e  H8=515eb76fc396 H7=3f3a3e68594e
```

**备份链（2026-11-15 21:05 恢复演练后）**：本仓库是 `blob:none` 部分克隆。`git bundle --all`（251 MB）与范围包（79 KB）都缺历史 blob（如 `f2d8056`），**在全新目录 clone 全部失败**——bundle 目前不是可用备份。可恢复的是：

```
git archive HEAD -> pb_tree_<sha>.tar.gz        （4.2 MB，当前树）
git format-patch 0e7893e8..HEAD -> 13 个 patch  （提交序列）
```

已演练：tree 包解压到全新目录后，8 文件指纹重算 = `515eb76fc396`。彻底修复需要 `git fetch --refetch origin` 补齐缺失 blob，等 GitHub 链路恢复再做；在此之前不要用 bundle 当唯一备份。

**实验产物备份（2026-11-15 21:21）**：代码只是资产的一半，论文证据是 `experiments/paper_b_sensor_generalization/` 下的 `config.json` / `metrics.json` / `val_pred.csv` / `val_noise_predictions.csv` / `robustness_val.json` / `provenance.json` / `best.pt`。

| 项 | 值 |
|---|---|
| 包 | `/home/user/pb_recovery/pb_experiments_2111.tar.gz`（37 MB） |
| sha256 | `b029a2f5d898eeaeedc89e24c4d332e438b1dd76d3fd0b99b833a2900f9c56d9` |
| 第二份 | `a40:/home/yndx/pb_recovery/`（sha256 逐字节一致） |
| 覆盖范围 | 打包时 17 格（E0 12 + E1 前 5 格），后续每检查点重打 |

**数据侧恢复演练**（从恢复包解出 `E0-FD001-s101` 的 `val_pred.csv` 重算）：

```
n_engines 20   cuts_per_engine [5]
recomputed_mean_roots 8.573980577
recorded_mean_roots   8.573980290883457
abs_diff              2.856e-07        (< 1e-6)
```

**a40 不能做裸仓库**：那台机器没有 `git`（`bash: git: 未找到命令`），所以没有 `git init --bare`，改为文件级复制。a40 是另一台物理机，与 3090 故障、GitHub TLS 都无关，备份目标成立。

## 8. 复核命令

```bash
# 旧 study
python fp_check.py /home/user/sensor_aug_20261114        # → COMBINED 90f20d35953d
# 新分支
python fp_check.py <worktree>/paper-b-sensor-generalization  # → COMBINED 90f20d35953d
# 每格启动时间
ls -l --time-style=+%H:%M runs/*/*/*/config.json | sort -k6
```

## 9. 确认集的评测侧与训练侧产物（2026-11-16）

确认集第一次起跑时整批崩在鲁棒性评测上，暴露了 `shared/` 下两个产物的**性质不同**，
必须分开处置。这条写在这里，是因为它决定了确认集的数能不能和发现集并排比。

**故障现象**：`E0-FD001-s101-ff1a0bc2` 与 `E0-FD002-s101-cd6c016a` 均为
`EVAL_FAILED`，`last_error` 分别是 `KeyError('8')` 和 `KeyError('5')`——
同一个 bug 撞到不同发动机 id，不是两个独立故障。

**根因**：`shared/missing_patterns_val_<subset>.json` 是**按发动机 id 建键**的
（`patterns[str(unit)][pid][:k]`），而它由 `build` 用 `BASE['split_seed']=17` 的验证机生成。
`split_seed=29` 的验证机是另一批发动机，`evaluation.missing_keep` 找不到键就抛 KeyError。
干净指标不受影响（它在崩溃点之前就由训练循环写入），崩的是鲁棒性那半边。

**处置（两类分开）**

| 产物 | 用途 | 是否重建 | 理由 |
|---|---|---|---|
| `missing_patterns_val_*.json` | **评测专用** | **重建为 split_seed=29** | 训练期 mask 由 `mask_prob` 逐样本随机抽通道，不读这个库；按 29 重建**不改变臂定义**，只是让评测在新划分上有定义 |
| `noise_stats_*.json` | **训练期注入** | **按划分重估：29 用 29 训练机** | 规则是「尺度只从**训练机**估计」，在 29 上就是 29 的训练机。`noise_alpha=0.25` 是冻结超参，随划分变的只是它乘的那个 `sigma` |

> **2026-11-16 08:5x 自我更正**：本节初稿写的是「`noise_stats` 不重建、保持 17，因为属臂定义」。
> 那个论证是错的，已推翻。错在把「冻结超参」和「按划分现估的量」混为一谈：
> `noise_alpha=0.25` 是冻结超参，而 `sigma` 是**从训练机估计**的数据量——预注册规则原文
> 就写着「尺度只从训练机估计」。复用 17 那份等于让 29 的验证机进了估计里（见下），
> 既违反规则又造成泄漏。正确做法是按 29 的训练机重估。
> 实现上不需要改 runner：`train.py:_augmentor` 读的是 `cfg['augment']['noise_stats_path']`，
> 所以把 E2a 的 config 指向 `noise_stats_<subset>_split29.json` 即可，h8 不变。

重建后 4 个子集与 17 版的验证机差异：FD001 18/20、FD002 46/52、FD003 18/20、FD004 48/50。
17 版原文件归档在 `shared/archive_split17/`，两版都在，可追溯。

**训练期 σ（E2a）：已按 29 的训练机重估，这是修协议违规，不是改臂**

初稿沿用了 17 那份，理由是「属臂定义」。这个论证错了：`noise_alpha=0.25` 才是冻结超参，
`sigma` 是**从训练机估计**的数据量，预注册规则原文就是「尺度只从训练机估计」。
实测 17 训练机覆盖 29 验证机的比例是 **FD001 18/20（90%）、FD002 46/52（88%）、
FD003 18/20（90%）、FD004 48/50（96%）**——沿用等于让九成被评测发动机参与了臂参数，
不可辩护。

处置：`scripts/make_noise_stats_split.py` 用 `_prepare_split(..., 29)` 的训练数据重估，
写到 `shared/noise_stats_<subset>_split29.json`（**不覆盖** 17 那份，发现阶段 20 格的
`noise_stats_hash` 记的是 `e5521aea…` 等旧值）。四个子集重估后 **训练/验证重叠均为 0**
（脚本里 `assert` 断言）。`train.py:_augmentor` 读 `cfg['augment']['noise_stats_path']`，
所以只改 manifest 里 E2a 那 20 行的路径，runner 与 h8 都不动。

| 子集 | 17 版 hash | 29 版 hash（E2a 确认集现用） |
|---|---|---|
| FD001 | `e5521aeae40eff46` | `c6680c652c3238cf` |
| FD002 | `da9cc13c73144a5c` | `c5b2080a77423293` |
| FD003 | `d13a6a51eccc9dc2` | `8ed36fe8c2ca525e` |
| FD004 | `1e5cb42de17e15f7` | `32f61e3448dc7c58` |

**评测期 σ（全臂）：仍沿用 17，性质不同，不算违规**

`evaluate_run` 里噪声压力评测读的是固定的 `shared/noise_stats_<subset>.json`（17 版），
这条路与训练期注入是两件事：评测期的 σ 只是给**所有臂**施加同一个固定扰动，
不进模型、不进主指标、四臂完全相同，所以臂间 Δ 有效。换成 29 会把 80 行的
`noise_stats_hash` 全改掉、run_id 全变、已完成格作废重跑，而主结论一个字都不变，
代价与收益不成比例。

**因此论文里必须分开写**：

- 训练期 σ（仅 E2a）：按 29 的训练机重估，验证机未参与；
- 评测期 σ（全臂）：沿用 17，四臂同一份，不进主指标；
- **推论**：确认集的**噪声压力绝对量级不能与发现阶段并排比较**；
  但**主指标（clean mean-of-roots ΔRMSE）不涉及这两个产物，跨阶段可比**。

**代码身份不变**：`refresh_eval_patterns.py` 与 `freeze_confirm_extra.py` 都不在 h8 文件列表里，
`scripts/run_sensor_aug.py` 与 `paper_b/` 相对 `3468954e` 零改动，确认集仍为 `h8=515eb76fc396`。

**验收**：两格最终必须是 `status=COMPLETE` 且八件产物齐全
（`config.json`、`metrics.json`、`val_pred.csv`、`best.pt`、`robustness_val.json`、
`val_noise_predictions.csv`、`val_missing_predictions.csv`、`provenance.json`）。

**eval-only 重试在这个 runner 版本里无法到达终态（已实测，会影响复现性）**

初稿写的是「补评测走 `trained=True` 的 eval-only 分支、不重训」。这个方案**技术上不成立**，
实测推翻了它：

- `status='COMPLETE'` 全仓只有**一个**写入点：`train.py:361`，在 `train_one` 内部。
  worker 的成功分支只 `print('DONE')`，**不写 `status.json`**。
- 于是走 `trained=True` 跳过 `train_one` 的格子，`is_complete()` 永远返回 False，
  `next_task()` 每轮都重新返回同一格——**4 秒一轮无限空转**，日志形态是
  `resume evaluation only` → `DONE ... in 4s` → `START` 同一 run_id → 循环。
- 后果：该卡被这一格永久占住，且 `E0-FD002-s101` 被捡起后会同样死循环。

**结论**：任何走过 eval-only 重试的格子在这个版本里都到不了 `COMPLETE`。
要修只能改 `scripts/run_sensor_aug.py`，而那会改变 `h8`，把 60 个发现格与已跑确认格的
指纹声明全部作废。所以处置改为**让它走 `trained=False`**：删掉这两格的
`train_complete.json` / `attempts.json` / `status.json`，重训后由 `train_one` 正常写 `COMPLETE`。

代价与收益：两格各重训约 13 分钟，旧的 `best.pt` 与崩溃期产物被覆盖，得到的是新的模型抽样。
收益是最终八件产物全部出自同一次训练、同一个 `train_one`，不存在「状态是手写的」或
「checkpoint 与评测产物来自不同代码路径」这类瑕疵。**追溯链上不出现同一 `run_id` 下两个模型，
因为旧产物已被覆盖，留下的只有一次完整训练的结果。**

**后来者注意**：如果重跑本轮的确认集，必须确认跑的是带此缺陷的 runner 版本；
任何需要「补评测而不重训」的场景在这个 `h8` 下都不可行。

## 10. 两条分析纪律（2026-11-16，均由确认集并行跑引出）

### 10.1 前缀 glob 会跨分划串台，且错了看不出

`runs/` 现在同时存放发现集（split 17）与确认集（split 29），而 run id 的格式是
`<arm>-<subset>-s<seed>-<config_hash>`——**两个分划的同名格子共享前缀、只有 hash 后缀不同**：

```
E0-FD001-s202-2040584e   split_seed=29（确认集，当时还在训练，只有 2 台机）
E0-FD001-s202-5b89adbc   split_seed=17（发现集，20 台机）
```

`sorted(glob('E0-FD001-s202-*'))[0]` 取到确认集那份，算出的 Δ 是 **−0.1285（n=2）**，
真值是 **−0.5198（n=20）**。**错得完全看不出来**——它照样给出一个合理的数。

**纪律**：任何碰 `runs/` 的脚本只能用两种方式认格——

1. 按 manifest 里的**完整 run_id** 精确匹配；
2. 走 `scripts/_cells.py`：按前缀匹配后读 `config.json` 的 `split_seed` 过滤，
   并**要求恰好一个命中，否则抛错**（不得 `return None` 静默跳过）。

`audit_e0.py` 原本是 `root.glob("E0-*")` 无过滤遍历，改后加 `--split-seed`（默认 17）。
另外四个脚本原本有 `len(hits) != 1 → 放弃`，不会给出错数，但已全部改走 `_cells.py`。

**已公布数字是否需要重算：不需要。** 四个脚本原本的「匹配数≠1 就放弃」保证了它们没给过错数；
`audit_e0.py` 无防护，但它当时跑的时候盘上只有 12 个发现格。改后回归：

```
audit_e0.py        -> cells 12, missing 0, max|mean_roots - recorded| 4.089199308054958e-07
audit_final_table  -> E1a clean merged=-0.5450  neg=4/4
                      per-subset {'FD001':-0.6432,'FD002':-0.2382,'FD003':-0.9553,'FD004':-0.3433}
```

与改前逐位一致。

### 10.2 `COMPLETE` 只表示训练结束，不表示评测完成

`status='COMPLETE'` 由 `train.py:361` 在**训练结束时**写；`evaluate_run`
（写 `robustness_val.json` / `val_noise_predictions.csv` / `val_missing_predictions.csv`）
在它**之后**才跑。所以存在一个窗口：格子已是 `COMPLETE`、八件产物也都在，
但鲁棒性三件仍是**上一个模型**算出的旧文件。

确认集那两格正好是这种情况——eval-only 重试曾用旧 `best.pt` 成功写出过鲁棒性三件，
而训练正在用新的 `best.pt` 重跑。只看状态和齐全度**查不出来**。

**纪律**：每格必须核 mtime 顺序

```
best.pt <= robustness_val.json <= val_noise_predictions.csv
        <= val_missing_predictions.csv <= provenance.json
```

工具：`scripts/audit_cell_integrity.py`（`--manifest` 或 `--all-on-disk`），
同时检查 `status=='COMPLETE'` 与八件齐全。60 个发现格与 80 个确认格都要过这一关，
不能只看总数。

### 10.3 「同一个统计量的值取决于本次还算了什么」

第三个同族缺陷，也是这一轮最贵的一个：**代码不报错，给出一个自洽的错值**。

实例：`rng` 在臂循环外建一次、跨臂共用。后果有两层——

1. 一条臂的 CI 取决于**同一次调用里还跑了哪些臂**；
2. 更隐蔽的一层：同一条臂里多个统计量（主 CI、去 FD001 的 LOO CI、`lr4−mlp` 对比、
   每个 β 的噪声压力 CI）共用一条流，于是 **LOO 的端点取决于主 CI 先抽走了多少个数**。

第 2 层与「跑了哪些臂」无关，纯粹取决于「跑了几个统计量」，所以按第 1 层的修法
（rng 挪进臂循环）**修不干净**。

**纪律**：任何报出的区间/统计量，必须是 **(数据, 种子, 统计量)** 的函数。
实现上把 RNG 建在 bootstrap 函数**内部**，而不是调用点——让共享在构造上不可能，
而不是靠调用点自觉。

**检测方法**（也是这一轮的定位手段）：同一条臂单独跑一次、放进全量里再跑一次，
端点不一致就是它。再用「同一批逐机差分 + 同一个 bootstrap 函数」喂进去比一次，
若逐位相同，则问题不在取数、不在重采样实现，只在流位置。

与 §10.1（前缀 glob 跨分划串台）、§10.2（`COMPLETE` 只代表训练结束）同族：
**都不会抛异常，只会给出一个看起来正常的错值。** 三者的共性判据是
「换个上下文就变」——所以验收必须包含「换上下文复跑」这一项，而不只是「跑通了」。

## 11. E4 Sensor Identity 控制臂（2026-11-16）

**结论先行：E4 只作控制臂，不构成方法贡献。** 它复用模型里已有的 `sa_mode='identity'`
开关，与 Xu et al. 2026 §2.2 式(6) **精确重叠**；本轮不把它写成任何形式的创新。

### 11.1 必须更正的一处描述

`sa_mode='identity'` **不是「去掉 sensor attention」**。源码里它是：

```python
if self.sa_mode in ('identity', 'identity_cond'):
    st = st + self.sensor_id      # (1, 14, 64)，零初始化
st = self.sensor_attn(st)         # 注意力块完整保留
```

即在传感器注意力**之前**加一个零初始化的传感器身份嵌入，注意力照旧。
参数量也对得上：E0 `115,651` → E4 `116,547`，差 **+896 = 14×64**。

所以 E4 与 E0 的差异是「结构 + **0.77% 容量**」，**不是纯结构**。
任何人若把它写成「去掉注意力」，那是错的；审稿人若追问「多出来的参数有没有贡献」，
必须能答出这 896 个参数。

### 11.2 结果（E0 逐机配对、四子集等权、发动机级 bootstrap 10000）

| | 值 |
|---|---:|
| 合并 Δ | **−0.4098** |
| 95% CI | **[−0.7088, −0.1080]** |
| 方向 | 3/4 为负 |
| 去 FD001 | −0.6646 [−1.0283, −0.3023] |
| seed 级 | −0.327 / −0.164 / −0.738，sd 0.296 |

逐子集：**FD001 +0.3544（变差）**、FD002 −0.1938、FD003 −1.0730、FD004 −0.7269。

### 11.3 这个结果真正的信息量

加一个**零初始化**的身份嵌入（第 0 步等价于恒等映射），合并就改善 **0.41 周期**，
量级与 dropout 臂在确认集上的干净集增益（−0.36）**相当**。

两个毫不相干的旋钮给出同一量级的收益，说明：**这个强基线本身有约 0.4 周期的可及空间**，
而不是某个机制特有。这条比单独报「某个模块有效」更有价值，也更诚实——它同时解释了
为什么增强类只能到 ENGINEERING-ONLY：可及空间是基线自带的，不是方法带来的。

**因此 E4 的用途是把「可及空间」标定出来**，不是证明 identity 嵌入有用。

### 11.4 FD001 的反向行为（写稿单列）

FD001 在整轮里反复与其他三个子集反向：

- E4 上它是**唯一变差**的子集（+0.3544）；
- E1a 在发现阶段它最强（−0.643），确认阶段它最弱（−0.189）。

这与确认集那条「子集强弱排序在两个划分之间翻转」是同一现象的另一个侧面：
**子集级排序主要由该划分抽到哪些验证机决定，不是子集本身的属性。**
写稿时单独一段，不要混进汇总表里当趋势。

## 12. E5 的 h8 变更只增不改（2026-11-16）

E5 改了 `paper_b/models/dspt.py`（新增 `SensorInteraction`）与
`paper_b/training/train.py`（`_build` 多传 `lr_mode`），因此 **h8 从 `515eb76fc396`
变为 `574ac4a06ae7`**。这是预注册里写明的、E5 四臂共用的新身份。

### 12.1 变更是「只增不改」，且已实测

`lr_mode='none'` 时不创建任何新模块、不额外消耗 RNG，所以建出的图与旧 h8 完全一致。
实测（`scripts/audit_e5_none.py`，用 `==` 比 `val_engine_mean_roots_rmse`，不是容差）：

先在前 5 格落盘时比过一次（5/5），`E5none` 12 格全部 `COMPLETE` 后**扩到 12/12**：

```
subset    seed           E0       E5none  verdict
FD001      101     8.573980     8.573980  IDENTICAL
FD002      101    10.071332    10.071332  IDENTICAL
FD003      101     8.689980     8.689980  IDENTICAL
FD004      101    11.077094    11.077094  IDENTICAL
FD001      202     7.817681     7.817681  IDENTICAL
FD002      202     9.838620     9.838620  IDENTICAL
FD003      202     8.623595     8.623595  IDENTICAL
FD004      202    11.367865    11.367865  IDENTICAL
FD001      303     8.744210     8.744210  IDENTICAL
FD002      303    10.053853    10.053853  IDENTICAL
FD003      303     9.021650     9.021650  IDENTICAL
FD004      303    11.900607    11.900607  IDENTICAL

compared 12, identical 12
VERDICT: E5 none arm reproduces stage-1 E0 exactly -> it is a true identity baseline under the new h8
```

**12/12，没有任何一格不等。** 这里没有「大部分相同所以放过」的余地：判定用 `==` 精确相等，
任何一格出现 `diff` 都会在 `verdict` 列显式打出来，且结论会自动翻成
「`none` 不是 E0」。既然一格都没有，`none` 与 E0 的同一性就是逐格成立的，不是抽样成立。

**结论：`none` 臂就是「E0 那个模型」，论文里可以这样称呼**，不必退成
「同一架构在新 h8 下重训」。同时这也说明 E5 的 within-pairing 干净——
四臂之间的差异只能归因到 `lr_mode`，不掺「换了 h8」这个变量。

### 12.2 由此产生的操作约束

h8 已经是 `574ac4a06ae7`，所以**E0–E4 的旧格子不能在这棵树上重跑**：
重跑会得到新指纹，与它们 `provenance.json` 里记的 `515eb76fc396` 对不上。
旧记录本身仍然正确（它们当初就是在旧代码上跑的），但「重跑复现」这条路对旧臂已封。

### 12.3 这条也进复现性说明

一个只增不改的代码改动（新增休眠模块 + 多传一个配置键）下，
同一 seed 的 `none` 臂与旧 h8 的 E0 **逐位相同**。
这与确认集那次 `E0-FD001-s101` 重训复现 `7.6728326636558535` 是同一现象的另一例：
**FP32 + 固定 seed 下，这套栈的重训可复现到浮点末位。**
