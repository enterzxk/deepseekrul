# DSPT-HC 任务书（Paper B · 结构化健康状态与运行周期协同建模）

> 本文件是 tmh 的唯一执行契约。原文来自用户 2026-11-11 21:50 的任务书，本文保留其全部硬性要求，
> 并补上本轮实测到的环境事实与路径修正。**冲突时：本文的环境事实与边界 > 用户原文中的路径假设。**
>
> 版本：v1 · 2026-11-11 · 维护人：郑总（主控线）

---

## 0. 目标与定位

在**现有 DSPT 骨干不变**的前提下，研究如何用**结构化健康指标（HI）显式表示累积退化**，并把
**绝对运行周期（Cycle）当作年龄先验**，联合建模退化速度与剩余使用寿命。

- 内部代号：**DSPT-HC**（Dual-Span Spatio-Temporal Transformer with Health–Cycle Coupling）。代号不是论文名。
- 目标期刊：IEEE TIM。
- 阶段定位：**探索阶段**。不要求一开始就投稿级，但必须形成可证伪的机制创新 + 可靠证据 + 公平对照。
- **本任务不启动、不改动、不覆盖 Paper A（区间预测/保形）任何成果，也不覆盖旧 DSPT 的可信结果。**

### 三条已有问题（必须正面回答，不许绕开）

- **P1：HI 没真正参与 RUL 计算。** 旧模型 `RUL = reg_head(h_B)`，HI 只是 `hi_head(h_f)` 的辅助损失；
  删掉 HI 头 RUL 路径照样成立 → HI 不是独立退化状态。
- **P2：Cycle 可能是年龄捷径。** 原模型用绝对周期 Fourier 编码，周期集合 `[25, 50, 125, 250]`，
  输入把 cycle 裁剪到 ≤250。需要判定：Cycle 贡献多少？是否过度依赖年龄？Fourier 是否优于简单年龄编码？
  为什么去掉 Cycle 后部分模型严重退化？Cycle 能否只影响**未来退化速度**而不直接决定 RUL？
- **P3：MS / SA / HI 的消融存在负向贡献。** 历史消融里 FULL 不稳定优于内部 Base，HI 加到 MS+SA 后
  部分子集变差。这些历史结果是探索性的、协议不一致，**不能当最终结论**。本轮禁止无目标堆模块。

---

## 1. 环境事实（2026-11-11 实测，勿凭记忆改）

| 项 | 事实 |
|---|---|
| a40 连接 | `ssh a40`（HostName `172.25.52.97`，User `yndx`，Port 22，密钥 `~/.ssh/id_ed25519`，已配好免密） |
| **a40 工作根目录（唯一）** | **`/media/sangfor/vdb/user/zxk/deepseek-rul`**（2TB 盘 `/dev/vdb`，挂载点 `/media/sangfor/vdb`，可用 1.9T，yndx 可写无需 sudo） |
| 环境脚本 | `/media/sangfor/vdb/user/zxk/env.sh` —— 跑任何东西前 `source` 它（设 HC_ROOT / HC_OUT_ROOT / TORCH_HOME / PATH） |
| a40 Python | `/home/yndx/miniconda3/envs/a40_env/bin/python` → Python 3.9.25，torch 2.5.1+cu121，CUDA 可用 |
| a40 GPU | NVIDIA A40，49140 MiB，基本空闲 |
| a40 磁盘 | 系统盘 `/` 只剩 124G，**已弃用**；新工作全部在 2TB 盘，剩 1.9T |
| **旧目录（只读）** | `/home/yndx/deepseek-rul` —— 旧实验与旧代码，**只读参考，不得在其下跑任何新东西、不得写入** |
| 本地数据 | `data/CMAPSS/train_FD00{1..4}.txt`、`test_FD00{1..4}.txt`、`RUL_FD00{1..4}.txt` 齐全 |
| a40 数据 | `<工作根目录>/data/CMAPSS/`（14 个文件已就位） |
| GitHub | `https://github.com/enterzxk/deepseekrul`，Paper B 分支 `punet-interval`，交接提交 `95e729e` |

**目录树（2TB 盘）**

```
/media/sangfor/vdb/user/zxk/
├── env.sh                      # source 它
├── cache/torch/                # TORCH_HOME，避免写系统盘
├── logs/                       # 长任务 nohup 日志
└── deepseek-rul/               # = HC_ROOT
    ├── paper_b/  src/  scripts/   docs/  data/CMAPSS/
    └── experiments/hc/<run_id>/    # 产物，格式见 RESULTS_REGISTRY.md
```

**冒烟已验证（2026-11-11 22:0x）**：新目录下 `python tools_smoke.py` → `SMOKE OK`，
params 117955、pred (8,1)、hi (8,)、hi_seq (8,70)、前向 403 ms、显存 43.3 MB、torch 2.5.1+cu121。

**关键差异（务必知道）**：a40 旧目录的 `paper_b` 是**混合状态**（3 旧 + 2 新），不是"整体旧版"：
`dspt.py`（`341c2e…`）、`train.py`（`cc158c…`）、`protocol.json`（`54e74d…`，epochs 36 / patience 6）是旧版，
但 `data.py`、`metrics.py` 在迁移前已被覆盖成新版。**只看目录名或抽查一两个文件必判错 → 一律逐文件比 md5，基准永远取本地仓库。**
新目录已按本地仓库覆盖上传。

**基线版本制（2026-11-11 23:10 起）**：基准**写死 commit SHA，不写 `HEAD`**——写成 `HEAD` 时下一次提交会静默改变基准，今天哈希表已因此漂了四次。

| 版本 | commit | 说明 |
|---|---|---|
| v1 | `95e729e` | 交接提交（旧） |
| v2 | `bc75aa6` | 三 seed 结论（旧） |
| **v3** | **`f3142687ed6c04efcfa35173772f532b3ebb3c38`** | R1/R2 修复 + `paper_b/hc/` 首次入库（**当前基准**）；**14 个 blob** = 5 frozen + 9 hc |
| **v4** | **`0280cb8`**（链：`f314268`→`f83e27e`→`a203f4b`→`0280cb8`） | launch 独占创建 + 回读自检 + crash-safe close + `code_lock.py` + `worktree_dirty`/`baseline_version` 落 manifest + A3p 的 250 assert（在 `model_hc.py:64-73` 子类 override，**未动 `dspt.py`**）+ verify_loop 的 A3p 等价断言 + commit guard + shared-init 控制；**15 个 blob** |
| ~~v5~~ | `4c7f238`（父 `3f38cfa`） | **未通过验收，作废**：①dump payload 缺 `code_md5`/`git_sha`/`baseline_version`（`train_hc.py:170-172`）；②`--dump-init` 写完不退出、继续训练（`:177`）。待 v6 |
| v6–v8 | `20b9f3b` / `dc77575` / `aef910c`+`441c92f` | 过渡版本：v6 锁、v7 两提交结构、v8 三步验收；均已被 v9 取代 |
| **v9** | **代码 `d1e0527` / 锁提交 `6c405ad`** | **当前有效**：A3a init remap 修复（copy `pe_periods`，不 pop）+ CRLF 归一化（本地 `--check` 首次通过）+ 15 blob 0 不匹配 |
| v10+ | 每次提交新版本 | 必须在本表加一行，写清改了什么 |

**代码指纹的组成**：`FROZEN` 5 个（`dspt.py`、`data.py`、`train.py`、`metrics.py`、`protocol.json`）+
`OWN` 9 个（`config_hc`、`data_hc`、`model_hc`、`cycle_encoder`、`train_hc`、`run_hc`、`verify_loop`、
`snapshot_val`、`compare_val_snapshot`）。
**`paper_b/hc/__init__.py` 不纳入代码指纹**（空文件，无逻辑），按数量核对时不要把它算进去——v3 = 14，v4 = 15。

> **勿引用**：`2482baa`、`8066f3f`、`2ae9b5a` 三个 SHA 曾被本地 `reset` 重写，不在任何分支的祖先链上。
> 它们现在还能 `cat-file` 到，但 reflog 过期后会被 `gc` 清掉；按它们复核会直接失败。引用一律用上表的可达 SHA。

**代码锁规则（2026-11-11 定，取代"冻结窗口"的临时做法）**

**版本号唯一性（2026-11-12 补充，硬规则）**：一个版本号**只能对应一种代码状态**。
- 同一版本号出现两把不同的锁，**视为版本号复用，该版本号下所有 run 一律无效**；
- 已确认复用：**v6**（两把锁）、**v7**（`70c8779` 的锁 `git_sha=20b9f3b`；`dc77575` 的锁是 from-index、`git_sha=None`）→ **v6/v7 标 retired，不得引用**；
- 以后复用版本号即判该版本作废，不追溯，但该版本下的 run 全部降为 `unknown`。

一个版本 = **两个提交**：
1. **代码提交**：改 15 个受锁文件里的任意一个；
2. **锁提交**：只含 `docs/paper_b_hc/registry/code_lock.json`，由 `python -m paper_b.hc.code_lock --write --version vN` 生成。

要点：
- 锁只覆盖 15 个 `.py`，`code_lock.json` 自己**不在**覆盖范围，所以提交锁不会让它自己失效，提交后 `--check` 仍通过。
- `lock.git_sha` 记的是**被锁代码所在的 commit**，不等于 HEAD；**不要为了让它等于 HEAD 再生成一次**（会无限递归）。
- 任何动了受锁文件的提交，**必须**跟一个锁提交，否则 a40 上 `run_hc` 拒绝启动。
- 锁文件里的 `definition` 字段写的是 `md5 of git show HEAD:<path> blob bytes`，**这里的 HEAD 指锁自己的 `git_sha`**，不是会移动的 HEAD。比对时 `rev` 一律从锁文件里读。

规矩：
- 每个 run 的 `manifest.json` 写 `baseline_version`（如 `v3`）+ `git_sha`（完整 SHA）+ `worktree_dirty`。
- **`worktree_dirty=true` 的 run 不进 Gate 表**；`run_hc` 启动前查脏，非空直接拒绝启动。
- **a40 无 git**，`git_sha` 在远端恒为 `WORKTREE`，**不能**拿它去 `git show`。a40 侧由 `paper_b/hc/code_lock.py`
  生成并校验哈希清单；本地核对用 `git show f3142687ed6c04efcfa35173772f532b3ebb3c38:<path> | md5sum`。

**基准 md5（当前唯一有效 = v9；下方 v2/v3 说明仅作历史）**：

> **v2 变更（2026-11-11 22:40，主控裁定）**：`train.py`、`metrics.py` 的哈希因 B5 更正而变——
> 只删了"权重 9 倍"这句已被证伪的表述，改的是 docstring/注释文本，R2 逻辑零改动（`git diff` 已核）。
> **不回滚**：保住旧哈希等于把错误说法留在冻结代码里。
> 旧值仅作历史记录：`train.py=07e7379dc2a8ae93c0f828efa8694b6c`、`metrics.py=87e8245b49f4d23f61c6dd019958aa24`。
> 新值由两条独立测量互证：tmh 本地/远端逐文件 `md5sum`；燕子独立在 a40 采样三次（22:39 起稳定不变）。
>
> **v3/v4 状态（2026-11-11 23:15，主控）**：v2 已被提交 `f314268` 取代，表中 `train.py`/`metrics.py`
> 两行作废（现为 `13ec2fbb75ed09a389e4e979a82e11e6` / `39202682865d14dc6be5d332ac7a1c16`）。
> **基准 SHA 写死，不写 `HEAD`**：比对一律 `git cat-file blob <锁里记的 git_sha>:<path>`，上传用同一 blob；禁止直接 scp Windows 工作树（CRLF 会污染字节）。今晚 HEAD 在 `f314268 → 0280cb8 → 4c7f238 → 20b9f3b → dc77575 → 441c92f → 6c405ad` 之间动了七次，任何绑在 `HEAD` 上的表述都会过期。
> ——CRLF/LF 差异会改 md5，2026-11-11 22:49 那次漂移就是这么来的。
> 当前等 `v4` 定稿，定稿后本表整体重写并登记。

> **v9 定稿（2026-11-12 00:10，主控）**：v9 是当前**唯一**基线——代码提交 `d1e0527`、锁提交 `6c405ad`、
> `baseline_version=v9`。取哈希一律 `git cat-file blob d1e0527:<path> | md5sum`，上传用同一 blob，
> **禁止 scp Windows 工作树**（CRLF/LF 会改 md5）。v8 及更早全部 superseded：v8 的 `pop pe_periods`
> 在共享初始化路径上必炸，不能作任何 run 的基线。v9 相对 v8 只变两个：`train_hc`→`069f8977…`、
> `code_lock`→`45916d9a…`（remap 改 copy-never-pop + CRLF 归一化）。
> 三方独立复算 0/15 不符（锁 vs `d1e0527`、锁 vs `6c405ad`、锁 vs a40 实测）；a40 `code_lock.json`
> md5 = `c04c9586…`，`code_lock --check` = `CODE LOCK OK`。

| 文件 | md5（v9） |
|---|---|
| `paper_b/models/dspt.py` | `fb7011023b284a5f571cba4d4abb6927` |
| `paper_b/training/data.py` | `c2ee079f6e60a8b0d0eb608676ff871a` |
| `paper_b/training/train.py` | `13ec2fbb75ed09a389e4e979a82e11e6` |
| `paper_b/training/metrics.py` | `39202682865d14dc6be5d332ac7a1c16` |
| `paper_b/configs/protocol.json` | `c55720ffe09deff3a9b59c218a180114` |
| `paper_b/hc/config_hc.py` | `bffe552fefa49c0e9cff38a31c6cd8ff` |
| `paper_b/hc/data_hc.py` | `3ad7c111cf0268409bb40d79f7271cee` |
| `paper_b/hc/model_hc.py` | `c55eb2b430e2d356a809a0ccbbb3f3f0` |
| `paper_b/hc/cycle_encoder.py` | `f0ddc9115ba49627d4804fa6d5752e6d` |
| `paper_b/hc/train_hc.py` | `069f8977bee8cf6aa01953c817d01601` |
| `paper_b/hc/run_hc.py` | `8784311bebdc6891b59058f278e8d1f8` |
| `paper_b/hc/verify_loop.py` | `2c15fcb4293e226102e368c717ef3f3a` |
| `paper_b/hc/snapshot_val.py` | `be21a3cdc6f7022f3871faaf9d33a69b` |
| `paper_b/hc/compare_val_snapshot.py` | `daff09811d4dbe6fa74c8bcb6190b2a6` |
| `paper_b/hc/code_lock.py` | `45916d9a08c88b40072c03f2127cfd07` |

> R1 行为不变已证明：改前/改后四子集 32/32 字段逐元素相等、`max|diff|=0`、
> 两个 npz 的 sha256 同为 `38032744387b71be50410aac2a4c23d52454d6f72cf504ad9bab5b153a66f11f`；
> 省略 `cuts` 确实 `raise ValueError`（主控独立复验通过）。
> R6 划分 hash 也已复验一致：FD001/FD003 `f81d7f2496c494ca`、FD002 `cacf4ea1d7cf96b6`、FD004 `eb1b0f00047d393e`。

**Python 依赖**：`sklearn 1.6.1`、`pandas 2.0.3`、`numpy`、`torch 2.5.1+cu121` 均可用。
若遇 `ImportError: libstdc++.so.6: version GLIBCXX_3.4.29 not found`，
用 `source /media/sangfor/vdb/user/zxk/env.sh`（其中已设 `LD_LIBRARY_PATH=<env>/lib`）。

### 1.1 路径修正（用户原文写的路径与实际不符，以本表为准）

| 用户原文 | 实际情况 | 本轮使用 |
|---|---|---|
| 核心模型 `src/candidates/dspt.py` | 存在，但那是**旧版/简化版**（8018 B，无 cond、无 FastSensorAttention） | **不修改**，仅作历史参照 |
| 训练代码 `docs/paper_b_code/run_dspt.py` | **该目录不存在**（本地与 a40 都没有）。真实运行器在 `paper_b/training/` | 见下行 |
| 研究历史 `docs/paper_b_results/` | **该目录不存在**。历史在 `paper_b/reports/` 与 `paper_b/results/` | 只读 |

**实际要冻结的"原 DSPT"（Live baseline）**：

- 模型：`paper_b/models/dspt.py`（14040 B，含 `FastSensorAttention`、`cond` 支持、`build_dspt`）
- 数据/窗口：`paper_b/training/data.py`（`prepare_train` / `prepare_test` / `split_units` / `fit_hi`）
- 训练：`paper_b/training/train.py`（`train_one` / `predict` / `set_seed`）
- 指标：`paper_b/training/metrics.py`
- 协议：`paper_b/configs/protocol.json`（**冻结**：split_seed 7、fold 0、val_frac 0.2、
  val_cuts `[0.4,0.55,0.7,0.85,0.95]`、rul_cap 125、span 70、seeds `[101,202,303]`、
  epochs 40、patience 12、batch 512、lr 1e-3、wd 1e-4、d_model 64、layers 2、dropout 0.2、hi_lambda 0.5）

> `paper_b/models/dspt.py` 与 `src/candidates/dspt.py` **md5 不同**（`FB7011…` vs `634EB8…`）。
> 任何"原 DSPT 基线"必须明确声明用的是哪一份；本轮基线一律用 `paper_b/models/dspt.py`。

### 1.2 代码同步纪律

**基线提交（2026-11-11 22:50，唯一代码基准）**：`f3142687ed6c04efcfa35173772f532b3ebb3c38`（短 `f314268`，父 `bc75aa6`）。
- 所有 run 的 `git_sha` 必须写这个 SHA；它之后的改动算未提交工作树，`manifest.json` 要写 `worktree_dirty=true` 并列出差异文件 md5。
- 管理员核对 `code_md5` 时基准取**本提交的 blob**：`git show f314268:paper_b/training/data.py | md5sum`。
  不取任何时刻的工作树快照——否则哈希每十分钟过期一次（已连续发生三次）。
- 在本提交之前的 SMOKE（`T1431Z` 失败、`T1436Z` 通过）跑的是更早的工作树，`code_md5` 与基准不一致，
  二者都标 `not evidence`，**不作为证据引用**；正式放行前需在 `f314268` 上重跑一次 SMOKE。

- a40 无 git，本地与远端**必然漂移**。每次上传后必须 `md5sum` 逐文件核对，把 hash 写进 `manifest.json`。
- 上传只允许写 `<工作根目录>/paper_b/hc/`、`<工作根目录>/experiments/hc/`、`<工作根目录>/docs/paper_b_hc/`。
- **禁止**写 `/home/yndx/deepseek-rul/`（旧目录，只读）；禁止改 `paper_b/models/`、`paper_b/training/`、
  `paper_b/configs/`、`experiments/punet_*`、`docs/tables/`。
- 跑训练一律用绝对路径 `/home/yndx/miniconda3/envs/a40_env/bin/python`，或先 `source /media/sangfor/vdb/user/zxk/env.sh`。
- 长任务用 `nohup ... > /media/sangfor/vdb/user/zxk/logs/<run_id>.log 2>&1 &`，日志别写系统盘。

---

## 2. 不可触碰边界（违反即任务失败）

1. 不改 `paper_b/models/dspt.py`、`paper_b/training/*.py`、`paper_b/configs/protocol.json`。
2. 不改 `src/candidates/dspt.py`、`src/candidates/dspt_vanilla.py`、`src/candidates/model.py`。
3. 不写 `docs/tables/`、`experiments/punet_*`、`paper_b/reports/`、`paper_b/results/`。
4. 不覆盖 Paper A 任何文件（区间预测 / 保形 / MUP / `docs/manuscripts/`）。
5. 不 force-push，不改写已提交结论；结论被推翻只追加"已撤回 + 原因"。
6. 不编造任何实验数据、指标、日志、引用；没跑就写"未跑"。
7. **不用官方 TEST 做任何设计选择**（选周期形式、选窗口、选超参、选模型都不行）。
8. 不做不可逆删除；`rm -rf` 一律禁止，清理只动自己的 `experiments/hc/tmp/`。

新代码只允许落在：`paper_b/hc/`（代码）、`experiments/hc/`（产物）、`docs/paper_b_hc/`（文档）。

### 2.1 冻结文件的显式白名单（X1 裁决，2026-11-11）

§2 的"不改 `paper_b/training/*.py`"与 §5.2 的授权修复冲突，现给出**穷举白名单**，白名单之外一律仍冻结：

| 文件 | 允许的改动 | 授权编号 |
|---|---|---|
| `paper_b/training/data.py` | `prepare_train` 的 `cuts` 默认值 → `None` + 首行 `raise`（行为保持） | R1 |
| `paper_b/training/metrics.py` | **只允许新增** `per_engine_rmse` / `per_engine_rmse_map`；现有函数一行不许动 | R2 |
| `paper_b/training/train.py` | 选轮改用发动机级主指标、落盘双口径字段、修正过期 docstring | R2 / R4 |
| `paper_b/models/dspt.py` | **无**。一行都不许动 | — |
| `paper_b/configs/protocol.json` | **无**。协议常量不许动 | — |

任何超出上表的改动都算违反边界；如确需新改动，先报主控加白名单，不许先改后报。

---

## 3. 四个模型版本

### V0 原始 DSPT（历史强基线）
保留修复后的原 DSPT。必须同时给出三份，且**评估协议完全一致**。

**骨干固定（X2/X3 裁决，2026-11-11）**：`ms_mode='full'`、`sa_mode='basic'`、`use_pad_mask=True`、
`d_model=64`、`layers=2`、`dropout=0.2`、`span=70`。即 §5 要求保留的「MS 多尺度卷积 + Temporal Transformer +
SA 跨传感器注意力 + Learned Sensor Pooling」全部在场。**Stage C 的 A0–A5 与 V0 三份共用这一个骨干**，
任何变体不得改骨干；改了骨干就换一张表、换一个结论范围。

| 变体 | ms_mode | sa_mode | hi_source | use_cycle |
|---|---|---|---|---|
| **V0-Cycle** | full | basic | `hf`（HI 辅助损失开） | **True** |
| **V0-NoCycle** | full | basic | `hf` | **False** |
| **V0-NoHI** | full | basic | `none` | True |

> **V0-NoHI ≡ A3p**：两者是同一配置（full / basic / none + 原 Fourier cycle，frozen pre-norm 注入点），**只跑一次、双向引用**，
> 不得重复训练，也不得当成两个独立证据。（2026-11-11 修订：原写 A3，但 A3a 是 post-norm 变体、与 frozen 不等价；
> A3p 才是与 `build_dspt(use_cycle=True)` 逐位等价的注入点，故改为 A3p。）
> V0-Cycle 与 A3p 的差 = HI 辅助损失的作用；
> V0-NoCycle 与 A1 的差 = HI 辅助损失的作用（在无 cycle 条件下）。
>
> factorial 选出的候选骨干 `single_proj / basic / hf` **不属于 Stage C**。它丢了 MS 多尺度，
> 与 §5 冲突；若后续要在它上面做，必须单列一张表并写明"候选骨干上的 cycle 作用"，**不得与 V0 同表**。

### V1 DSPT + Structural HI
保留 DSPT 时空编码器；把原 HI 辅助头升级为**结构化累积退化状态**；**主 RUL 输出必须经由 HI 状态生成**。
初始版本**不输入绝对 Cycle**。

### V2 DSPT + Structural HI + Cycle
在 V1 上加**独立 Cycle 分支**，研究不同周期表达与 Cycle 影响退化速度的方式。
**Cycle 分支不得绕过 HI 直接回归 RUL。**

### V3 DSPT + Health–Cycle Coupling
增加**退化证据感知的年龄先验调节**：按传感器退化状态、历史退化速率、可观测信息，决定 Cycle 先验在
未来退化预测中的作用。**只有 V1、V2 给出可靠证据，才进入 V3。**

---

## 4. 阶段 A：文献与创新审查

精读以下 6 篇（**必须拿到真实可访问来源，读正文而非摘要；查不到就写查不到，禁止编造**）：

1. Knowledge-guided RUL prediction for multi-sensor devices under a data-model interactive framework，MSSP 2026
2. A hybrid prognosis method based on health indicator and Wiener process: multi-sensor monitored aero-engine，EAAI 2025
3. ST-KDFormer，IEEE TIM 2025
4. HDA-GNN，IEEE TIM 2025
5. SFI-STHGAT，IEEE TIM 2026
6. STGATN，IEEE TIM 2025

重点核查已有工作是否已经做过：HI 作为直接 RUL 预测状态 / 单调损伤增量 / 失效阈值到达预测 /
Wiener 与 First-Hitting-Time / 未来退化速度估计 / Cycle 年龄先验 / 年龄与真实退化测量解耦 /
工况感知退化速度模型。

**独立审查本方法与 MSSP 2026、EAAI 2025 的区别。不得把已有 HI+FHT 方法当全新思想。**

交付 `docs/paper_b_hc/RELATED_WORK_GAP.md`：写**值得创新的科学问题**，不是模块清单。

---

## 5. 阶段 B：冻结骨干 + 公平实验协议

**最重要原则：不能为了加结构化 HI，把原 DSPT 重设计成另一个模型。**

必须保留：原始三通道特征（raw / raw−baseline / causal EMA）、MS 多尺度卷积、Temporal Transformer、
SA 跨传感器注意力、Learned Sensor Pooling、原始时空特征维度与训练配置。允许修复**已证实**的代码错误。

新模型**只能在共享表示之后**增加结构化退化状态与动力学预测模块。禁止同时改输入编码、传感器建模、
损失和预测路径，再把成绩变化归因于 HI。

### 训练协议（冻结）
- Train / Validation / Official Test 三段；C-MAPSS FD001–FD004。
- 官方训练数据按**发动机 ID** 划分 80% Train / 20% Validation（split_seed 7, fold 0）。
- **≥3 个随机种子**（101/202/303）。同一阶段内所有候选：相同划分、相同基础编码器、相同验证截断、
  相同训练预算（epochs 40 / patience 12）。
- 官方测试集**不参与任何设计选择**。历史测试集已被多次观察，报告里必须写明这一限制，
  **不得宣称"完全独立测试验证"**。
- **样本单位**：Validation 以**发动机**为统计单位；截断规则只用训练数据确定；
  **不得用官方测试 RUL 标签的经验分布构造验证截断点**。
- 训练可用多个因果窗口，但**模型选择必须是统一的发动机级评价**。
- **窗口**：先固定 `L=70`（协议已定），隔离 HI/Cycle 结构变化；核心方法确定后，再在 Validation 上
  按子集搜索 `L=[30,50,70,100,130]`，各子集允许不同最优窗口。

### 5.1 评估口径（2026-11-11 审计后由主控确定，全部变体统一）

依据用户任务书「Validation 以发动机为统计单位」「模型选择必须采用统一的发动机级评价」。

| 用途 | 指标 | 定义 |
|---|---|---|
| **主指标（选 best epoch、判 Gate、排名次）** | 发动机级平均 RMSE `val_engine_mean_rmse` | 先算每台验证引擎自身 RMSE，再对引擎**等权**平均 |
| 次报指标（只用于与历史数字对照） | 全局池化 RMSE `val_pooled_rmse` | 所有 (引擎 × 截断点) 混算，即 `train.py:101-103` 现有口径 |

硬性要求：
1. **所有变体（含 V0 三份）一律用主指标选 best epoch**；V0 若历史用的是池化口径，必须在新口径下重跑，
   不得新旧混用。最终对比表的 seed 集必须一致（同 seed 才可比）。
2. 两个口径**都记录、都报告**，并在报告里说明二者关系：
   **池化 = 先平方后平均，发动机级 = 先平均后平方**（Jensen），因此池化值**恒 ≥** 发动机级平均；
   冻结 val 下每台引擎恰好 5 个窗口，**每台引擎在池化里的权重相同（1/E）**，
   不存在"长引擎行数多所以权重大"。真正的差异是**平方误差质量**不同：按 cap=125 复算每台引擎的 Σy²，
   FD004 最长/最短 = 3.94、FD001 3.28、FD002 3.17、FD003 2.55 —— 少数高误差引擎对池化值影响更大。
   **禁止写"权重 9 倍"**（那是未封顶 (1-f)·n 标签下的数，不是本协议的数；2026-11-11 已由测试负责人证伪并更正）。
3. `train.py:153` 的字段名 `val_engine_rmse` **实际是全局池化 RMSE**，命名误导。读数据一律以字段定义为准，
   不要按名字理解；hc 层输出用上面的新字段名。
4. val 数字一律标注 **select-on-val**（按验证集挑最优轮），不得当作无偏估计写进论文。

### 5.2 必守的已知陷阱（2026-11-11 审计确认）

| 编号 | 陷阱 | 要求 |
|---|---|---|
| **F1 / R1（P0）** | `data.py:281` `prepare_train` 的 `cuts` 默认 `(0.5,0.75,0.9)` ≠ 协议 `[0.4,0.55,0.7,0.85,0.95]`。省略即静默用错验证集 | 改为 `cuts=None` 且显式 `raise`；这是**行为保持的缺陷修复**（现有调用都显式传值），允许改，改后必须证明 val 张量与改前逐元素一致 |
| **R2** | 池化口径 ≠ 发动机级 | `metrics.py` 只允许**新增** `per_engine_rmse()`，不改现有函数；`train_one` 同时记录两个口径 |
| **R3** | select-on-val 乐观偏差 | 报告必须标注 |
| **R6** | 划分漂移 | 每个 launch 写 `val_engine_ids_hash`，基准：FD001 `f81d7f2496c494ca`、FD002 `cacf4ea1d7cf96b6`、FD003 `f81d7f2496c494ca`、FD004 `eb1b0f00047d393e`（定义：`md5(",".join(sorted(val_ids)))[:16]`）。不一致即 fail |
| **R7** | FD001 与 FD003 的 val 划分完全相同（同为 1..100、同 RNG） | 不是泄漏，但**四子集不是四次独立重复**，论文不得这么写 |
| **Cycle 饱和** | `data.py:186` `min(cyc/250, 1.0)` 使 cycle≥250 完全饱和；FD004 有寿命 435 的引擎 | A3 内部必须含「不截断 / 不饱和归一化」对照臂 |
| **pe 被 LayerNorm 部分抵消** | `dspt.py:243-244` pe 加到 pooled 后立刻 `self.norm` | "cycle 影响多大"只能靠**干预实验**测，不能看 pe 范数 |
| **截断标签饱和** | `rul_cap=125` 使 val 标签恰为 125 的占比：FD001 7.0% / FD002 8.5% / FD003 11.0% / **FD004 20.4%** | 报告里说明；必要时补一版「剔除饱和点」的敏感性分析 |
| **val / test 不可比** | val 是完整 run-to-failure 轨迹，test 是截断轨迹 | 两套数字不得混表 |
| **off-protocol 产物** | `paper_b/experiments/probe_selection_metric.py:45` 省略了 `cuts`，一直在旧的 `(0.5,0.75,0.9)` 上跑，它产出的 `results/selection_metric_probe.csv` 是 off-protocol 的 | **不得作为证据引用**；要复现必须先补 `cuts` 再重跑，并在报告里标明是重跑版 |

### 5.3 Gate A 的执行分级（算力调度）

72 格（6 变体 × 4 子集 × 3 seed）不必一次铺满。按下面两步，**最终对比必须 seed 对齐**：

1. **第一波（方向判断）**：6 变体 × 4 子集 × **1 seed（101）** = 24 run，约 10 GPU-h。
   目的只是判「Cycle 有没有增益、Fourier 是否优于简单年龄」，不据此下最终结论。
2. **第二波（确认）**：对 A0 / A1 / A3 以及第一波里出现显著差异的变体，补到 **3 seed**（101/202/303）。
3. 最终 Gate A 表**只放同 seed 集的对比**；不同波次的数字不得拼进同一张表。

**X4 裁决（2026-11-11）**：正在跑的 `paperb-s23-seeds-20261111`（24 run）用的是 22:04 之前的 `train.py`
（池化选轮），与新主指标口径不同。它属于 Paper B v2 factorial 线，**标记为 `legacy metric`**，
**不得与 hc 的任何结果同表、不得作为 V0 或 Gate A 的证据**。若日后需要它的数字，只能在新口径下重跑。

---

## 6. 阶段 C：Cycle 年龄先验研究（先不开发新 HI）

保持原 DSPT 骨干与直接 RUL 回归头不变，**只改 Cycle 路径**：

| 编号 | 变体 |
|---|---|
| A0 | Cycle-only baseline |
| A1 | Sensor-only DSPT |
| A2 | Sensor + Raw Cycle |
| A3 | Sensor + 原 Fourier Cycle |
| A4 | Sensor + Log Cycle |
| A5 | Sensor + 非周期单调样条年龄编码 |

A3 内部必须对比三种：原始 250 截断 / 不截断 Cycle / 合理归一化但不饱和的 Cycle。

**注入点必须分表（2026-11-12 定，实现事实决定的硬约束）**
`model_hc.py` 里只有 **A3p** 走 pre-norm（`DSPT_HC_Frozen.encode` 在 `self.norm` 之前加 cycle），
**A2/A3a/A3b/A3c/A4/A5 全部走 post-norm**（`DSPT_HC.encode` 在 `super().encode()` 之后加）。因此：
- **编码形式比较**只在 post-norm 家族内部做：`A1`（无 cycle 基线）+ `A2`(raw) / `A3a`(Fourier) / `A4`(log) / `A5`(spline)；
- **注入点比较**单独做：`A3p` vs `A3a`（同 Fourier、同共享初始化，只差位置）；
- **A3p 不得与 A2/A4/A5 比较编码形式**，任何表里出现这种比较即判该表作废；
- 不把其他臂改造成 pre-norm：`dspt.py` 不可改，要统一就得在子类复制 `encode` 主体，破坏"只 import 不复制"原则，代价不值。

**A3a vs A3p 必须显式共享初始化（2026-11-11 补充，否则对照无效）**：
`dspt.py` 里 `pe_proj` 在第 119 行创建，**早于** layers(125) / sensor_attn(133) / cond_emb(139) /
reg_head(146) / hi_head(149)。A3p 走 `use_cycle=True` 会抽这一次 `Linear(8,64)`，A3a 走
`use_cycle=False` 不抽，于是 119 行之后所有模块的 RNG 流整体错位——**同 seed 不够，主干权重会整体不同**。
做法：先建 A3p，把 `pe_proj.weight/bias` 搬进 A3a 的 `cycle_enc.proj`（同为 `Linear(8,64)`，
输入同为 `cat([sin,cos])`），其余键原样载入，再同 seed 训练。
`verify_loop.py` 的验收改成**三步**（2026-11-11 修订：原来要求的"同输入输出逐位相等"不可能成立——
A3a 的 cycle 项注入在 post-norm、A3p 在 pre-norm，二者按构造就不同）：

1. **共享张量 0 差**：两侧 state_dict 里所有共享键逐元素相等（`max|Δ|=0`）；
2. **循环项同值**：同一 `cat([sin,cos])` 输入下，A3p 的 `pe_proj` 与 A3a 的 `cycle_enc.proj` 输出逐位相等；
3. **置零投影后 ≤1e-6**：把两侧的循环投影权重置零，前向输出差 `≤1e-6`——这一步证明**差异只来自注入点**。

三条全过才算共享初始化成立。真实初始化下记录的 `A3a − A3p` 的 Δ 作为 **treatment**，不是噪声。

**`--dump-init` 的验收（v9 起是纯导出模式，2026-11-12 更正）**
v9 的 `run_hc.py:242-249` 里 dump 是 export-only：写 payload → 打印 → `return`，**不写 launch、不写 close、不建 run 目录、不训练**。
所以验收改成三条，与实现一致：
1. payload 含 `variant / subset / seed / span / created_utc / code_md5 / git_sha / baseline_version` **八个字段**；
2. 终端打印 `init sha256=...`，且该值与文件实际 sha256 一致；
3. 消费方（A3a）的 `manifest.init_sha256` 与它相等。

链条是「dump payload 自带溯源 → A3a manifest 记录同一 sha256」，**registry 里不需要有一条 dump 记录**。
（v5 的 `INIT-DUMP-A3p` 有 launch，是因为老代码还走训练路径、训到了 ep020；v9 不会再有这种记录。）
**不要再要求 dump 写 `close.json` 或检查 `train.log` 里有没有 `ep001`**——导出模式下根本没有这两个东西。

**每个变体都要记录 9 项**：①四子集 RMSE ②NASA Score ③MAE ④Bias ⑤不同 RUL 区间表现
⑥不同发动机寿命分位数 ⑦不同截断长度 ⑧Cycle-only 相对 Sensor-only 的收益
⑨固定传感器证据下**干预 Cycle** 的预测变化。

**禁止用官方 TEST 调周期形式。** 最终必须回答：**Cycle 到底帮模型理解退化，还是主要提供寿命年龄先验？**

交付 `docs/paper_b_hc/CYCLE_DEPENDENCE_STUDY.md`。

---

## 7. 阶段 D：结构化 HI 模型

### D1 结构化累积损伤
- 编码器输出 `z_t = DSPT_Encoder(X_1:t)`，**该分支不输入绝对 Cycle**。
- 必须因果（因果注意力或等价机制），t 时刻不能看到未来测量。
- 增量网络：`delta_t = softplus(f_theta(z_t))`；累积：`D_t = D_{t-1} + delta_t`，`D_0 = 0`；`HI_t = 1 - D_t`。
- 固定失效阈值 `D* = 1`；非负增量保证单调；用训练发动机完整 run-to-failure 轨迹约束终点 `D_T → 1`。
- **失效终点只是训练监督目标，不得作为推理输入。**

### D1 必须解决"完整历史"问题
禁止在每个窗口开头把 D 重置为 0、再把窗口内累积量当全寿命损伤。
必须用**发动机级因果状态更新**，或设计有明确初始状态估计与一致性约束的窗口状态空间模型。
同一发动机同一时刻，不能因窗口长度不同得到不合理的完全不同损伤尺度。

### D2 未来退化速度
`v_t = Smooth(delta_1:t)`；`v_(t+k|t) = softplus(g_phi(D_t, v_t, z_t, k))`；
`D_(t+k|t) = D_t + Σ_j v_(t+j|t)`；由阈值到达时间给 RUL。

### D3 可微失效阈值
设计**可微软阈值到达**，让 RUL 误差能反传到损伤增量与未来速率模型；不得只用不可导硬 argmin 作唯一训练算子。
训练后可用明确阈值到达规则推理，并核对软/硬到达时间的系统偏差。
设定合理预测时间上限，上限内未达阈值的样本必须有定义明确的损失与输出处理。

### D4 训练损失
`L = L_RUL + λ_T L_terminal + λ_S L_smooth`（初始版不加一堆辅助损失）。
- `L_RUL`：基于阈值到达时间的 RUL 误差
- `L_terminal`：完整训练发动机失效终点的损伤锚定
- `L_smooth`：抑制退化速率高频波动

必须**检查各损失梯度贡献**，避免量纲不同导致辅助任务失效。
对 `RUL=125` 封顶标签：必须明确它是**截断监督目标**，不等于真实退化状态；**原始 RUL 与封顶标签两种数据都要保存**。

### D5 四组核心对照
| 编号 | 变体 |
|---|---|
| B0 | 原 DSPT，无 HI |
| B1 | 原 DSPT，HI 辅助损失 |
| B2 | DSPT + 结构化 HI + 固定未来速率 |
| B3 | DSPT + 结构化 HI + 可变未来速率 |

B0–B3 同编码器基础、同划分、可比训练预算。**B2/B3 不允许绕过 HI 的直接 RUL 预测头。**
**只有 B3 在稳定性与预测能力上可接受，才进入 Cycle 融合。**

交付 `docs/paper_b_hc/STRUCTURAL_HI_RESULTS.md`。

---

## 8. 阶段 E：Cycle 与退化速度协同

在 B3 基础上：

| 编号 | 变体 |
|---|---|
| C0 | 不使用 Cycle |
| C1 | Cycle 直接拼到状态特征 |
| C2 | Cycle 只作为未来退化速度的条件信息 |
| C3 | 退化证据感知的 Cycle 调节（**核心候选**） |

C3 建议形式：
```
v_(t+k|t) = softplus( f_phi(D_t, v_t, z_t, k) + g_t · a_psi(cycle_t, k) )
g_t       = sigmoid( q_omega(D_t, v_t, z_t) )
```

硬约束：
- Cycle 不得通过独立 RUL Head 绕过 HI；
- 退化状态分支不得直接使用绝对 Cycle；
- 只用当前**已观测** Cycle，不用未来真实信息；
- 年龄调节项必须能被**单独消融**；
- 与普通 Cycle 拼接、固定门控、直接 Fourier 编码**公平比较**；
- 报告参数量与计算时间。

### 机制验证（5 问，必须回答）
1. 同年龄、不同退化状态，模型能否区分 RUL？
2. 相近退化程度、不同年龄，Cycle 是否有额外价值？
3. 传感器证据强时，年龄门控是否降低？
4. 传感器退化信息弱时，年龄先验是否更有价值？
5. 这种行为是否真由模型学到，而不是人为解释？

**不得仅凭注意力/门控权重可视化宣称因果性。**

交付 `docs/paper_b_hc/HEALTH_CYCLE_COUPLING.md`。

---

## 9. 研究关卡（Gate）

### Gate A：周期机制成立
确定 Cycle-only / Sensor-only / 不同周期编码的主要差异与稳定性。
**若 Cycle 增益可被简单线性年龄输入完全解释，就不得把 Fourier 编码当创新点。**

### Gate B：结构化 HI 可用
- HI 在时间上满足结构单调性；
- 损伤尺度与训练失效终点合理对齐；
- B2/B3 预测路径**确实依赖 HI**；
- 多 seed 下 RMSE / NASA Score 可接受；
- 至少拿到明确的退化状态机制证据。

若 HI 结构性能明显弱于 B0/B1，先分析信息瓶颈是否过强；**一次针对性修复仍无收益即停止该方案**。

### Gate C：Cycle–HI 协同有效
C3 相对 C0/C1/C2 有可靠增益，或在**预先定义的关键失效模式**上稳定改善，且整体性能退化不可接受地大 → 不通过。
**不以单一子集、单一 seed 的最优值判断创新成立。**

### Gate D：论文级验证（前三关过了才做）
四子集完整实验、多种子统计、窗口敏感性、近期强基线公平对比、传感器噪声与缺失鲁棒性、
年龄扰动与退化状态分析、参数量/FLOPs/推理时间、条件允许时 N-CMAPSS 独立验证。
据证据决定是否进入 IEEE TIM 写作。

---

## 10. 交付物（8 份，全部落在 `docs/paper_b_hc/`）

| # | 文件 | 归属 | 对应阶段 |
|---|---|---|---|
| 1 | `RELATED_WORK_GAP.md` | tmh | A |
| 2 | `METHOD_SPECIFICATION.md` | tmh | B/D/E |
| 3 | `CYCLE_DEPENDENCE_STUDY.md` | tmh | C |
| 4 | `STRUCTURAL_HI_RESULTS.md` | tmh | D |
| 5 | `HEALTH_CYCLE_COUPLING.md` | tmh | E |
| 6 | `ABLATION_RESULTS.md` | tmh | D/E |
| 7 | `REPRODUCIBILITY_AUDIT.md` | tmh 起草 + 管理员复核 | 全程 |
| 8 | `PAPER_B_RESEARCH_DECISION.md` | tmh 起草 + 主控签发 | 收口 |

每份文档开头必须有：`状态：未开始 / 进行中 / 已完成`、`最后更新`、`证据 run_id 列表`。

### 每次实验必须留存
Git commit SHA（本地）/ 上传文件 md5（a40）、完整配置 JSON、随机种子、数据划分（含发动机 ID 列表或
其 hash）、checkpoint 路径、训练日志、预测 CSV（逐样本）、指标 JSON。
**没有 run_id 的结果不算结果。**

---

## 11. 与实验结果管理员的分工（防乱套）

- **tmh** 只写：`paper_b/hc/**`、`experiments/hc/**`、8 份交付文档、`registry/launch/<run_id>.json`。
- **实验结果管理员** 只写：`docs/paper_b_hc/registry/runs.csv`、`docs/paper_b_hc/RESULTS_INDEX.md`、
  `docs/paper_b_hc/RESULTS_AUDIT_*.md`。
- 启动实验**前**先写 `docs/paper_b_hc/registry/launch/<run_id>.json`（独占创建，已存在即视为重复）；
  结束后写 `registry/close/<run_id>.json`。
- 管理员据 launch/close 文件登记进 `runs.csv`，并抽查产物真实性（checkpoint 存在、预测条数、指标可复算）。
- **任一方不得改对方的文件**。细则见 `docs/paper_b_hc/RESULTS_REGISTRY.md`。

**违反写入边界的处理（2026-11-11 裁决）**：`docs/` 下没有文件锁，并发写会丢内容（已实测出现 `ReplaceFileW EIO` 写冲突）。因此：
1. 越权写入的那次修改**视为无效**，由文件所有者以自己那一版为准覆盖；
2. 覆盖后在审计文档追加一行：时间 / 被改文件 / 越权方 / 如何处理；
3. 需要别人文件的内容时，**只发替换文本，不落笔**；
4. 不靠重试解决写冲突——重试只能掩盖丢内容，§11 的「一人一文件」是唯一解。

---

## 12. 汇报与验收

- 每完成一个 Gate，写一段 ≤15 行结论：过了/没过、证据 run_id、下一步。
- 决策分级：实现细节/实验参数/排期 = L1 自决；**放弃技术路线、判断创新性、跨 Gate 推进** = 上报主控（L2），
  **不得转问用户**。
- 失败必须分类后记录：假设错 / 设计不足 / 优化错 / 数据协议坏 / 无真实增益 / 算力不足。
- 验收以磁盘产物为准：文件存在 + 文档里的数字能在 `runs.csv` 与预测 CSV 里对上。
