# DSPT-HC 实验结果登记规范（实验结果管理员 执行手册）

> 目的：让 Paper B / DSPT-HC 的每一次训练、每一个数字**只有一个来源**，防止"结果乱套"——
> 同名实验跑两遍、指标口径混用、文档数字与产物对不上、旧结论被新数字悄悄覆盖。

版本：v1 · 2026-11-11 · 维护人：郑总（主控线）· 执行人：实验结果管理员

---

## 1. 唯一真相源

| 问题 | 唯一答案在哪 |
|---|---|
| 跑过哪些实验 | `docs/paper_b_hc/registry/runs.csv` |
| 某次实验的配置/产物路径 | `docs/paper_b_hc/registry/runs.csv` 的对应行 + `experiments/hc/<run_id>/manifest.json` |
| 某次实验的数字 | `experiments/hc/<run_id>/metrics.json` 与 `preds.csv`（**不是**文档里的数字） |
| 现在该看哪份文档 | `docs/paper_b_hc/RESULTS_INDEX.md` |
| 有没有造假/泄漏 | `docs/paper_b_hc/RESULTS_AUDIT_<日期>.md` |

**规则：文档里的任何数字，必须能在这张表里定位到 run_id 并从产物复算出来。对不上就是 bug，不是"笔误"。**

---

## 2. 目录结构

**a40 工作根目录（唯一）**：`/media/sangfor/vdb/user/zxk/deepseek-rul`（2TB 盘，别用系统盘 `/home/yndx/deepseek-rul`，那里只读）。

```
docs/paper_b_hc/                       # 本地仓库 + 远端 <HC_ROOT>/docs/paper_b_hc/ 各一份，内容同步
  registry/
    launch/<run_id>.json      # tmh 写，实验启动前，独占创建
    close/<run_id>.json       # tmh 写，实验结束后
    runs.csv                  # 管理员写，唯一台账
  RESULTS_INDEX.md            # 管理员写，给人看的索引
  RESULTS_AUDIT_<YYYYMMDD>.md # 管理员写，审计记录

<HC_ROOT>/experiments/hc/              # = /media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc
  <run_id>/
    config.json               # 实际生效配置（不是模板）
    manifest.json             # 代码 hash、数据划分、环境、产物清单
    metrics.json              # 验证/测试指标
    preds.csv                 # 逐样本预测（engine_id, cycle, y_true, y_pred, split）
    train.log                 # 训练日志原文
    ckpt/                     # checkpoint（按磁盘预算决定保留 best 还是 last）
```

台账在本地 `docs/paper_b_hc/registry/runs.csv`；产物在 a40 的 `<HC_ROOT>/experiments/hc/<run_id>/`。
`runs.csv` 里的路径列一律写**a40 绝对路径**，便于从任何机器复核。

---

## 3. run_id 命名（强制）

```
HC-<stage>-<variant>-<subset>-s<seed>-<YYYYMMDDTHHMMZ>
```

- `stage`：`A` | `B` | `C` | `D` | `E`
- `variant`：`A0`–`A5`、`B0`–`B3`、`C0`–`C3`、`V0C`（V0-Cycle）、`V0N`（V0-NoCycle）、`V0H`（V0-NoHI）、`SMOKE`。
  **A3 的子臂单列**：`A3a`（原 250 截断 Fourier）、`A3b`（不截断）、`A3c`（不饱和归一化）、
  `A3p`（frozen pre-norm 注入点对照）。四个都是合法变体，不是拼写错误。
- `hypothesis` **必填**：写"这次实验想证伪/证实什么"，一句话。填 `(not stated)` 的 run 按 `fail` 处理。
- **代码指纹三件套必填**（缺任一即 `fail`）：
  - `baseline_version`：如 `v3`，对应 `TASK_BOOK.md` §1 的版本表
  - `git_sha`：完整 commit SHA；a40 上恒为 `WORKTREE`
  - `worktree_dirty`：布尔值。**`true` 的 run 不进任何 Gate 表**
- `subset`：`FD001`–`FD004`
- `seed`：`101` | `202` | `303`（扩展需主控批准）
- 时间戳用 **UTC**，精确到分钟

示例：`HC-C-A3-FD001-s101-20261111T1402Z`

**同一 run_id 不得重跑。** 重跑必须换时间戳，并在 `registry_note` 里写清"重跑原因 + 前一次的 run_id"。

---

## 3.1 Gate 表准入（按格，不按精确 run_id）

主控裁决 2026-11-12（revision 51）：**准入按「格」判，`run_id` 只作句柄**。
原因是 W1 的 OOM 重试落在 `-W1r2` / `-W1r3` 后缀上，按精确 id 会把真实成功的格错误排除。

### 格键

```
(baseline_version, variant, subset, seed)
```

`baseline_version` **是承重件，不可省**。实测反例：省掉它时 `A3a-FD001` 会按「最早 ok」
选中 pre-versioning 的 `…T1505Z`（`val_engine_mean=10.5455`、`audit_verdict=fail`、已 retired），
而不是 v9 的 `0030Z`（`11.3167`），**差 0.77 RMSE，正好落在敏感性臂的主数字上**。

### run_id 正则（落成规则，不用口头描述）

格的证据必须匹配：

```
^HC-C-{variant}-{subset}-s{seed}-W{wave}(r\d+)?$
```

只写“前缀 `-W1`”会误纳 `-W1foo`；只写“以 `-W1` 结尾”会把 `r2/r3` 排除。


**`wave` / `seed` 是参数，不是字面量。**

| 波次 | `wave` | `seed` 取值 | 例 |
|---|---|---|---|
| Wave 1（Gate A） | `1` | `101` | `HC-C-A1-FD001-s101-W1r2` |
| Wave 2（G2 尺度对齐） | `2` | `101` / `202` / `303` | `HC-C-A2cal-FD001-s202-W2` |

> 写死 `s101` 或 `-W1` 会让 Wave 2 的格**全部无法准入**（seed 202/303 与 `-W2` 都不匹配）。
> 2026-11-12 实测发现该字面量问题，已参数化。

### 选行规则

1. 同一格取 `status=ok` 的 run 作为该格证据；`failed` / `aborted` / `unclosed` **不占位**；
2. 同格有多个 `ok` 时，按 `close.json` 的 `closed_utc` 取**最早**，其余标 `superseded`；
3. 证据行置 `in_gate_a_table=true`，其余行一律 `false`。

### 仍然排除（重要）

`0030Z` / `1640Z` 的 `run_id` 形如 `…-s101-20261112T0030Z`，**不匹配上述正则**，
因此**永远不是 Gate A 证据**，`in_gate_a_table=false`，只进 `CYCLE_DEPENDENCE_STUDY.md` 的 preliminary 小节。

**理由（批次效应）**：`0030Z` 来自 2×2 批次，而 A3a 的 FD002–FD004 来自 W1 批次。
G2 的判据是「四子集 3/4 同号」，一格跨批次会把批次效应混进符号计数。
`baseline_version` 承重那条只解决「选哪条 run」，不能用来把 `0030Z` 提升成 Gate A 证据。

### `A3a/FD001` 不空

主控 2026-11-12 下令补跑 `HC-C-A3a-FD001-s101-W1`，与 A3a 其余三格同批、同一 init
（`init_v9_A3p` 的 remap）。任何“用 `0030Z` 顶替该格”的写法一律作废。

### 时钟规则

**时间戳一律以 a40 为准。** 实测本地与 a40 相差约 9 分钟（本地 01:28:20 ↔ a40 01:37:10）。
`closed_utc` 全部由 a40 写入，格内排序用 a40 时间、内部自洽；跨主机比较必须注明时间来源。

---

## 3.2 Stage D 预留（2026-11-12 13:5x）

**D 目前零 run。本节只预留命名空间与格键，不向 `runs.csv` 写空行**——空行会污染准入统计与行数核对。

| 项 | 约定 |
|---|---|
| `stage` 列 | `D`（现有 C 67 行、B 24 行，按 stage 过滤时必须显式指定，不得混算） |
| run_id | `HC-D-{variant}-{subset}-s{seed}-{YYYYMMDDTHHMMZ}` |
| variant | `D0`–`D5`，**待 `METHOD_SPECIFICATION.md` 的 D1–D5 定稿后锁定**，未定稿前不得占用具体编号 |
| 格键 | 沿用 `(baseline_version, variant, subset, seed)`；`baseline_version` 从 v10 起，**不与 C 的 v9 混** |
| 波次前缀 | `-D1(r\d+)?`，与 C 的 `-W1(r\d+)?` 区分 |
| Gate 归属 | D 的判据另立表（结构化 HI / 到达时间），**不得并入 Gate A 的 24 格主表**；D 行的 `in_gate_a_table` 恒为 `false` |
| 准入 | 沿用 §3.1：格键 + `status=ok` + 最早 `closed_utc`；`failed` 不占位 |

**登记时点**：等 ①`METHOD_SPECIFICATION.md` 的 D1–D5 数学规格定稿、②测试负责人审完数学规格
（重点：窗口开头不得把 `D` 重置为 0 如何在数学上保证）、③第一个 D run 落盘，三者齐了才登记第一行。
在此之前台账的 D 区保持空，`check_runs.py` 不需要 D 行也能过。

**与 C 的分界**：C 的结论（cycle 通路 +10.46%、Fourier 编码 −5.03%、G4 待 L2）只约束 D 的设计选择
（cycle 用 raw 标量、编码形式不作为贡献点），**不构成 D 的任何证据**，也不得与 D 的数字同表比较。

---

## 3.4 Wave 2（G2 尺度对齐）预留：32 格（2026-11-12 18:1x）

**尚未 launch，本节只预留命名与格清单，不向 `runs.csv` 写空行。**

### 3.4.1 32 格 = 每子集 8 格 × 4

| 变体 | seed | 格数 | 说明 |
|---|---|---|---|
| `A1` | 202 / 303 | 8 | 补 seed；s101 已有（不重跑） |
| `A2cal` | 101 / 202 / 303 | 12 | 未截断 raw cycle + 初始化 RMS 标定（ρ=1.0） |
| `A3bcal` | 101 / 202 / 303 | 12 | 未截断 Fourier（原周期组）+ 同标定 |

合计 **32**。`A2raw` / `A3araw` 是对**现有 run 的展示别名**（原 A2 / A3a），
**不重训、不改 run_id、指标不动**；原 28 格全部复用。

`A2cal-sham`（12 格，预注册为「保留边缘分布、打断 cycle–RUL 对应」的置换，同投影同 gain）
只在需要主张「年龄信息」机制时才做，**本批不做**。

### 3.4.2 run_id 约定

```
HC-C-{variant}-{subset}-s{seed}-W2(r\d+)?
```

- `variant` 取 `A1` / `A2cal` / `A3bcal`；重试后缀 `r2` / `r3`（上限 r3），与 Wave 1 的 `-W1` 区分。
- **准入规则同 §3.1**：格键 `(baseline_version, variant, subset, seed)` + `status=ok` + 最早 `closed_utc`；
  `failed` 的重试不占位。`baseline_version` 随 v10 锁。

> 该前缀是台账侧的提案，**开跑前需主控确认**；若改用时间戳形式，按 §3 的通用 run_id 规则即可，
> 格清单不变。

### 3.4.3 A1 s101 的 canonical run（逐子集，已核）

| 子集 | canonical run_id | status | manifest |
|---|---|---|---|
| FD001 | `HC-C-A1-FD001-s101-W1r2` | ok | ✅ |
| FD002 | `HC-C-A1-FD002-s101-W1r2` | ok | ✅ |
| FD003 | `HC-C-A1-FD003-s101-W1r2` | ok | ✅ |
| FD004 | `HC-C-A1-FD004-s101-W1r3` | ok | ✅ |

**只有这四条有 `manifest.json`**；同格其余 run（FD001/2/3 的 `-W1`、`-W1r3`，FD004 的 `-W1`、`-W1r2`）
目录里只有 `config/launch/train.log`（FD004 的 `-W1r2` 另有一份 `close.json`），**没有 manifest**，
是中断或失败尝试，按准入规则不占位。

### 3.4.4 「seed 复用同一 init」的依据（写入台账，供后续引用）

**结论：Wave 2 的 seed 202/303 复用 s101 的同一份 init 文件，不重新生成。**

依据三条，均为实测：

1. **文件名里的 `s101` 是构造种子，不是训练种子。**
   `wave1_init_A1_s101.pt`（file sha256 `0e65e4e8…`）被 4 个子集共用；它的 **58 条骨干键**
   与 A3p dump（`init_v9_A3p_s101.pt`，sha `336074a6…`）的骨干**逐张量 `max|diff| = 0.000e+00`**。
2. **按 seed 重抽会换一套骨干**，把「init 噪声」混进「seed 极差」，正好废掉共享 init 的意义。
   共享 init 的目的是让 seed 之间只差**训练随机性**（数据顺序、dropout），
   因此论文里 seed 极差只能表述为「**训练随机性的下界**」。
3. **交叉验证**：A1/A2/A4/A5 五份 init 的 58 个骨干张量两两 10 对全部 `max_diff = 0`，
   差异只落在 `cycle_enc.proj.*` / `cycle_enc.knots` / `pe_proj.*` 这 5 个处理键上。

### 3.4.5 已知缺陷：`wave1_init_manifest.json` 已过期（开跑前必须重生）

实测 `experiments/hc/tmp/wave1_init_manifest.json` 记录的 `file_sha256` 与磁盘现值：

| 变体 | 记录值 | 实际值 | 判定 |
|---|---|---|---|
| A1 | `ef13324b…` | `0e65e4e8…` | **不符** |
| A2 | `6c510b8d…` | `bd16ae45…` | **不符** |
| A3p | `336074a6…` | `336074a6…` | 相符 |
| A4 | `1b237dc8…` | `16e8ce98…` | **不符** |
| A5 | `add896df…` | `47421c6f…` | **不符** |

权重未变（骨干仍逐位相同），但**字节变了**（00:43 那次重建）。因此：
**manifest 的 file_sha256 一律以重生后的为准；在重生之前，不得用它作为 init 的身份证明。**
以 A3p 那份为锚（唯一相符）可交叉确认权重未漂移。

---

### 3.4.6 Wave 2 批一结果（FD001，2026-11-12 19:5x）

| run_id | status | ep / best | `val_engine_mean` | manifest |
|---|---|---|---|---|
| `HC-C-A1-FD001-s202-W2` | ok | 34 / 22 | **11.353972188685571** | ✅ |
| `HC-C-A1-FD001-s303-W2` | ok | 36 / 24 | **11.442163300493988** | ✅ |
| `HC-C-A2cal-FD001-s101-W2` | **failed** | 0 / — | — | ❌（有 `close.json`，无 manifest/metrics） |

**A1 FD001 三 seed 带宽（后续 A2cal vs A3bcal 的噪声底）**

| seed | run | `val_engine_mean` |
|---|---|---|
| 101 | `HC-C-A1-FD001-s101-W1r2` | 11.2806 |
| 202 | `HC-C-A1-FD001-s202-W2` | 11.3540 |
| 303 | `HC-C-A1-FD001-s303-W2` | 11.4422 |

**跨度 0.162（相对 s101 为 1.43%）。** 三 seed 共用同一份 init（§3.4.4），
所以这个跨度是**训练随机性的下界**，不是 init 噪声。

**失败格原因（已登记，非环境问题）**

`RuntimeError: init_from mismatch: missing=['cycle_gain']`——`train_hc` 内联了一份与
`scale_align.load_shared_init()` 重复的 remap，那份没有排除 `cycle_gain`。
cal 臂建出的模型带 `cycle_gain`、init 文件里没有，严格断言即炸。
**影响面**：批二 5 格全是 cal 臂，会以同一原因全灭。

**预留**：`HC-C-A2cal-FD001-s101-W2r2`（v11 修复后的重试，失败那条**不覆盖**）。
按 §3.1 的格键 + 最早 `closed_utc` 规则，W2r2 成功即成为该格证据，`-W2` 的 failed 行不占位。

---

## 3.5 Wave 2 分析规则预注册（**先于结果登记**）

**登记时刻：2026-11-12 19:54:20（本地）/ 12:03:11 UTC（a40）。**
此时批二 6 格**尚未落盘**（`A2cal-FD001-s101-W2r2` 等均未 launch），故本节为**真预注册**，不是事后追认。

### 3.5.1 seed 是区组因子，不是配对设计

同一 seed 的 `A2cal` 与 `A3bcal` **只共享数据顺序**（permutation 用独立 generator、同 seed）；
但两臂模型构建时抽的参数个数不同，**dropout 的 RNG 流从第一次前向起就分叉**。
因此：

> **seed 是区组（blocking）因子，不是配对（paired）设计。**
> 不得按 seed 做配对检验并宣称方差被消除。

### 3.5.2 主分析与次要分析（写死，不得事后更换）

| 层级 | 方法 | 用途 |
|---|---|---|
| **主分析** | **3v3 非配对均值差 + 95% CI** | 唯一可写进结论的检验 |
| 次要分析 | 按 seed 的配对差 | **只作探索**，不得单独支撑结论 |

### 3.5.3 可辨阈值与功效（预注册，不得按结果调整）

基于 A1 FD001 三 seed 实测（11.2806 / 11.3540 / 11.4422，§3.4.6）：

- 训练 seed 级 σ ≈ **0.0809**，极差 **0.1616 RMSE（1.43%）**；
- 3v3 非配对均值差的 SE ≈ **0.0661**；
- G2 的 1% 阈值在基线 ≈10.6 上约 **0.106 RMSE**，仅 **≈1.60σ**。

由此写死两条：

1. **单 seed Δ < 0.162 不可解释**（低于训练 seed 噪声的极差）。
2. **CI 跨过阈值时，只能写「功效不足」，不得写「无差异」**（1.60σ 的效应量本就不足以判定）。

### 3.5.4 与既有噪声量的区分（防止混引）

| 量 | 含义 | 值 | 引用限制 |
|---|---|---|---|
| 训练 seed 级 σ | 换数据顺序 + dropout | ≈0.0809（FD001） | 本节用 |
| 投影抽样噪声 σ_Δ | **固定 seed**、只重抽投影 | 0.0378 | **不得与上一行互换** |

### 3.5.5 生效范围

本节对 Wave 2 的 `A2cal` vs `A3bcal` 比较（FD001 起，后续 FD002–FD004 同规则）生效。
**FD002–FD004 的 seed 带宽尚未测**，其可辨阈值需在该子集三 seed 齐了之后按同一方法补登，
**补登前不得用 FD001 的 0.162 跨子集套用**。

---

## 4. launch 文件 schema（tmh 在启动前写）

```json
{
  "run_id": "HC-C-A3-FD001-s101-20261111T1402Z",
  "stage": "C",
  "variant": "A3",
  "subset": "FD001",
  "seed": 101,
  "span": 70,
  "hypothesis": "一句话：这次实验想证伪/证实什么",
  "gate": "A",
  "host": "a40",
  "code_md5": {"paper_b/hc/model_hc.py": "…", "paper_b/hc/train_hc.py": "…"},
  "git_sha": "本地提交 SHA（若无提交写 WORKTREE）",
  "data_split": {"split_seed": 7, "fold": 0, "val_frac": 0.2, "val_engine_ids_hash": "…"},
  "budget": {"epochs": 40, "patience": 12, "est_gpu_min": 25},
  "launched_utc": "2026-11-11T14:02:00Z"
}
```

`hypothesis` 与 `gate` 必填。**没有判定门的实验不许启动**（见 AGENTS.md §14）。

---

## 5. close 文件 schema（tmh 在结束后写）

```json
{
  "run_id": "…",
  "status": "ok|failed|killed",
  "closed_utc": "2026-11-11T14:31:00Z",
  "epochs_run": 31,
  "best_epoch": 19,
  "metrics": {"val_rmse": 0.0, "val_score": 0.0, "val_mae": 0.0, "val_bias": 0.0},
  "artifacts": ["config.json", "metrics.json", "preds.csv", "ckpt/best.pt", "train.log"],
  "notes": "异常、OOM、早停原因等"
}
```

测试集指标只有在 `FREEZE` 冻结后才允许写；冻结前 `metrics` 里**不得出现 test_* 字段**。

---

## 6. runs.csv 列定义（固定，不许加列前先问主控）

```
run_id, stage, variant, subset, seed, span, gate, hypothesis,
host, git_sha, code_md5, split_hash, launched_utc, closed_utc, status,
epochs_run, best_epoch,
val_rmse, val_score, val_mae, val_bias,
test_rmse, test_score, test_mae, test_bias,
ckpt_path, log_path, preds_path, manifest_path,
registry_note, audited_by, audit_verdict
```

- `status`：`running` | `ok` | `failed` | `killed` | `superseded`
- `audit_verdict`：`pass` | `suspect` | `fail` | 空（未审）
- 指标一律**留 4 位小数**，单位与 `metrics.json` 一致；不要在这里做二次计算。

---

## 7. 登记流程

1. 扫 `registry/launch/` 新文件 → 在 `runs.csv` 追加行，`status=running`，`audit_verdict` 留空。
2. 扫 `registry/close/` → 补 `closed_utc / status / epochs / metrics / 路径`，从 `experiments/hc/<run_id>/` 取真实值。
3. **交叉核对**：`close` 里的 metrics 必须与 `metrics.json` 数值一致（容差 1e-6），不一致 → `audit_verdict=fail` 并记录。
4. 审计通过后，把该 run 写进 `RESULTS_INDEX.md` 对应 Gate 的表格。
5. 每次登记结束，在 `RESULTS_INDEX.md` 顶部更新 `最后更新` 与 `已登记 run 数`。

---

## 8. 审计检查清单（每次至少抽查 3 条，全量审计按 Gate 收口）

1. **产物存在**：`ckpt/`、`preds.csv`、`train.log`、`metrics.json` 都在，非空。
2. **条数对得上**：`preds.csv` 行数 = 预期发动机×截断点数（验证）或官方 test 窗口数（测试）。
3. **能复算**：用 `preds.csv` 的 `y_true/y_pred` 重算 RMSE/MAE，与 `metrics.json` 差 < 1e-6。
4. **无泄漏**：验证/测试的发动机 ID 与训练集**零交集**；`val_engine_ids_hash` 与实现一致。
5. **无未来信息**：窗口内不含 t 之后的测量；Cycle 只用已观测值；HI 分支不含绝对 Cycle。
6. **无测试集调参**：冻结前 `metrics` 无 `test_*`；模型选择只用 val。
7. **预算一致**：同阶段各变体的 epochs/patience/batch/lr 与协议一致，差异必须有 `registry_note`。
8. **代码没漂移**：a40 上跑的 `code_md5` 与 launch 声明一致。
9. **种子真实**：同 run_id 的 seed 与 `metrics.json`、日志里的 seed 一致。
10. **重复检查**：同 (stage, variant, subset, seed, span) 是否已存在 `ok` 的 run；重复则标 `superseded` 并说明。

**任何一条不过 → 该 run 标 `suspect` 或 `fail`，并在审计文档写明证据，不得直接改数字。**

---

## 9. 违规与处理

| 情况 | 处理 |
|---|---|
| 无 launch 文件的 run | 一律不登记，标 `unregistered`，通知 tmh 补声明后重跑 |
| 文档数字对不上产物 | 以产物为准，文档标 `已撤回`，写明原因，保留原文 |
| 同 run_id 重跑 | 旧行标 `superseded`，新行写 `registry_note` 指向旧的 |
| 疑似泄漏/未来信息 | 立即 `fail` + 上报主控，**该 Gate 暂停推进** |
| 覆盖他人文件 | 上报主控，按 AGENTS.md §4 处理 |

---

## 10. 写入边界（硬）

- 管理员**只写** `docs/paper_b_hc/registry/runs.csv`、`docs/paper_b_hc/RESULTS_INDEX.md`、
  `docs/paper_b_hc/RESULTS_AUDIT_*.md`。
- **不写** `paper_b/`、`experiments/hc/`、`src/`，**不改** tmh 的 8 份交付文档内容
  （发现问题只在审计文档里记，由 tmh 改）。
- 不跑训练、不下载数据、不动 Paper A 任何文件。
- 台账只追加，不删行；作废用 `status=superseded`。
