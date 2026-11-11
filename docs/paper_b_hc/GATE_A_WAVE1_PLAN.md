# Gate A 第一波：24 格清单与并行方案

状态：待发车（未启动任何训练）
最后更新：2026-11-12 00:35 (+08)
基线：v9（代码 `d1e0527`，锁 `6c405ad`）
依据：任务书 §5.3（6 变体 × 4 子集 × seed 101 = 24 run，只判方向）

## 1. 变体集

`{A0, A1, A2, A3p, A4, A5}`。**A3p ≡ V0-NoHI**（同配置），只跑一次、V0 表双向引用。
A3a 是敏感性臂，不进第一波（已有 0030Z）；A3b/A3c 不在第一波。

## 2. 初始化策略（公平性的关键）

6 个变体的骨干必须同权重，否则跨变体差里混入初始化噪声（牢莫在 A3a/A3p 上已证）。
做法：以 `init_v9_A3p_s101.pt`（sha256 `336074a6…`）为**骨干基准**，为每个变体生成一份
自己的 init 文件：骨干键从 A3p 复制，变体特有的循环模块保留自身初始化（形状不同，无法共享）。

| 变体 | 循环模块 | init 文件 |
|---|---|---|
| A3p | `pe_proj` Linear(8,64) | `experiments/hc/tmp/init_v9_A3p_s101.pt`（现成，sha `336074a6…`） |
| A1 | 无 | `wave1_init_A1_s101.pt`（仅骨干） |
| A2 | `ScalarCycle` Linear(1,64) | `wave1_init_A2_s101.pt`（骨干 + 自身 Linear(1,64)） |
| A4 | `ScalarCycle(log)` Linear(1,64) | `wave1_init_A4_s101.pt`（骨干 + **与 A2 同一份** Linear(1,64) 抽样） |
| A5 | `SplineAge` Linear(9,64) | `wave1_init_A5_s101.pt` |
| A0 | `CycleOnly`（无骨干） | 无 init（自身 seed 初始化，**参考线，不进 treatment 比较**） |

生成器放 `experiments/hc/tmp/make_wave1_inits.py`（不在 15-blob 锁内，不碰 `paper_b/hc/`），
产物 sha256 写入各自 manifest 的 `init_sha256`。

**注入点（2026-11-12 主控要求，必须写死）**

| 变体 | 模型类 | 循环项注入位置 |
|---|---|---|
| A3p | `DSPT_HC_Frozen` | `DSPT.encode` 内部、`self.norm` **之前**（pre-norm），与 `build_dspt(use_cycle=True)` 逐位等价 |
| A1 / A2 / A4 / A5 | `DSPT_HC` | `encode()` 返回之后（post-norm） |
| A0 | `CycleOnly` | 无编码器，直接线性回归年龄 |

因此 A3p 与 A1/A2/A4/A5 之间同时差「编码形式」和「注入位置」两个变量，
Δ(A2→A3p) 不能单独归因给 Fourier；G2 必须拆成两张表（见 §9），
拆表需要 post-norm 的 Fourier 臂 **A3a**，故第一波从 24 格扩为 28 格。

两条额外约束（2026-11-12 主控补充）：

- **A2 / A4 共用同一次 `Linear(1,64)` 抽样**：两者只差 `log1p` 变换，若循环投影各自随机，
  A2−A4 的差里会混入投影初始化噪声。
- **A0 是参考线**：它没有共享骨干，不参与「cycle 编码优劣」的 treatment 比较，只用来回答
  「这个 benchmark 能不能仅靠年龄解」。

## 3. 24 格清单

run_id 规则：`HC-C-{variant}-{subset}-s101-W1`（确定性、可预登记、天然防重名）。

| # | variant | subset | seed | init | run_id |
|---|---|---|---|---|---|
| 1 | A0 | FD001 | 101 | — | HC-C-A0-FD001-s101-W1 |
| 2 | A0 | FD002 | 101 | — | HC-C-A0-FD002-s101-W1 |
| 3 | A0 | FD003 | 101 | — | HC-C-A0-FD003-s101-W1 |
| 4 | A0 | FD004 | 101 | — | HC-C-A0-FD004-s101-W1 |
| 5 | A1 | FD001 | 101 | wave1_init_A1 | HC-C-A1-FD001-s101-W1 |
| 6 | A1 | FD002 | 101 | wave1_init_A1 | HC-C-A1-FD002-s101-W1 |
| 7 | A1 | FD003 | 101 | wave1_init_A1 | HC-C-A1-FD003-s101-W1 |
| 8 | A1 | FD004 | 101 | wave1_init_A1 | HC-C-A1-FD004-s101-W1 |
| 9 | A2 | FD001 | 101 | wave1_init_A2 | HC-C-A2-FD001-s101-W1 |
| 10 | A2 | FD002 | 101 | wave1_init_A2 | HC-C-A2-FD002-s101-W1 |
| 11 | A2 | FD003 | 101 | wave1_init_A2 | HC-C-A2-FD003-s101-W1 |
| 12 | A2 | FD004 | 101 | wave1_init_A2 | HC-C-A2-FD004-s101-W1 |
| 13 | A3p | FD001 | 101 | init_v9_A3p | HC-C-A3p-FD001-s101-W1 |
| 14 | A3p | FD002 | 101 | init_v9_A3p | HC-C-A3p-FD002-s101-W1 |
| 15 | A3p | FD003 | 101 | init_v9_A3p | HC-C-A3p-FD003-s101-W1 |
| 16 | A3p | FD004 | 101 | init_v9_A3p | HC-C-A3p-FD004-s101-W1 |
| 17 | A4 | FD001 | 101 | wave1_init_A4 | HC-C-A4-FD001-s101-W1 |
| 18 | A4 | FD002 | 101 | wave1_init_A4 | HC-C-A4-FD002-s101-W1 |
| 19 | A4 | FD003 | 101 | wave1_init_A4 | HC-C-A4-FD003-s101-W1 |
| 20 | A4 | FD004 | 101 | wave1_init_A4 | HC-C-A4-FD004-s101-W1 |
| 21 | A5 | FD001 | 101 | wave1_init_A5 | HC-C-A5-FD001-s101-W1 |
| 22 | A5 | FD002 | 101 | wave1_init_A5 | HC-C-A5-FD002-s101-W1 |
| 23 | A5 | FD003 | 101 | wave1_init_A5 | HC-C-A5-FD003-s101-W1 |
| 24 | A5 | FD004 | 101 | wave1_init_A5 | HC-C-A5-FD004-s101-W1 |
| 25 | A3a | FD001 | 101 | init_v9_A3p | HC-C-A3a-FD001-s101-W1 |
| 26 | A3a | FD002 | 101 | init_v9_A3p | HC-C-A3a-FD002-s101-W1 |
| 27 | A3a | FD003 | 101 | init_v9_A3p | HC-C-A3a-FD003-s101-W1 |
| 28 | A3a | FD004 | 101 | init_v9_A3p | HC-C-A3a-FD004-s101-W1 |

> 第 25–28 格是 2026-11-12 主控要求新增的 **post-norm Fourier 臂**（A3a），
> 用于把 G2 拆成「同注入点比编码形式」与「同编码形式比注入点」两张表。
> A3a 的 init 用 A3p 那份（`init_v9_A3p_s101.pt`），载入时 `pe_proj → cycle_enc.proj` remap，
> 与 `0030Z` 的做法一致。

## 4. 并行方案

- 显存：A40 49140 MiB。实测单 run 约 **6.9 GB**（`0030Z`/`0105Z`），双 run 并行时 14.4 GB
  （≈7.2 GB/run）。3 路按实测约 **21 GB**，余 ~28 GB；按保守 14 GB/run 则是 42 GB，
  仍 < 49 GB。两种口径都放得下，计划采用实测口径。
- **并发度 3**（`cuda:0`）。每路独立进程，`cudnn.benchmark=True` 保持默认。
- **R0 必须单跑**：`HC-C-A3p-FD001-s101-W1` 先在并发度 1 下跑完并核对逐位值，
  再开 3 路跑其余 27 格。`cudnn.benchmark=True` 下并发可能改变算法选择，
  把 R0 放在并发里会把「链路问题」和「并发算法选择」混在一起。
- 墙钟：单 run 实测 200–440 s。R0 单跑约 3.5 min；其余 27 格 / 3 路 = 9 波 × ~7 min
  ≈ **65–80 min**，加启动开销约 85 min。
- 调度：`experiments/hc/tmp/run_wave1.sh` 用 3 个槽位的信号量（`flock` 或 FIFO），
  逐格调用 `run_hc`，每格独立日志 `logs/wave1/<run_id>.log`。

## 5. 失败与重试（不污染台账）

1. `run_hc` 在异常路径也会写 `close.json`（`status=failed`），launch/close 成对，**不删**。
2. 失败格重跑一律**换新 run_id**：`…-W1r2`、`…-W1r3`，绝不复用旧 id（`O_EXCL` 会拒）。
3. 台账唯一键 = `(baseline_version, variant, subset, seed, init_sha256)`；同一格出现多个完成 run 时，
   保留第一个 `ok`，其余标 `duplicate`。
4. 重试只针对失败格；成功格不重跑。若某格连续 2 次失败，停该格并报错，不无限重试。
5. 发车前预检：`code_lock --check` 绿、`git status -- paper_b/{hc,training,models,configs}` 空、
   GPU 空闲、`init` 文件 sha256 已记录。

## 6. 未做

- 未生成 init 文件（等发车指令）。
- 未启动任何训练。

## 9. Gate A 可执行判据（预注册，阈值写死，不含 X）

主指标一律 `val_engine_mean_rmse`（发动机级等权平均，select-on-val）；单 seed，只判方向，
不作显著性声明。所有比较基于 24 格同骨干、同 seed、同预算（40 epoch / patience 12）。

定义：`Δ(A→B) = (RMSE_A − RMSE_B) / RMSE_A`，正值表示 B 更好。

| # | 判据 | 阈值（写死） | 结论 |
|---|---|---|---|
| G1 | Cycle 是否有用：A2 相对 A1 | 四子集 `Δ(A1→A2)` 均值 **≥ 1.0%**，且无子集 **< −2.0%** | 过 → Cycle 有增益；否则「cycle 无可测增益」 |
| G2a | **同注入点**（post-norm）比编码形式：A3a 相对 A2 | `Δ(A2→A3a)` 均值 **≥ 1.0%**，且至少 **3/4** 子集同号 | 过 → 在 post-norm 下 Fourier 优于线性年龄 |
| G2b | **同编码形式**（Fourier）比注入点：A3p 相对 A3a | 只报 `Δ(A3a→A3p)` 的均值与四子集符号，**不设增益阈值** | 用于把 G2 的联合差分解成「形式」与「位置」两部分 |
| G3 | 年龄捷径检查：A0 相对 A1 | `Δ(A1→A0)` **≤ −10.0%**（A0 明显更差） | 不过 → benchmark 可由年龄单独解，cycle 结论作废 |
| G4 | 干预方向：`cycle_intervention` | 所有带 cycle 的臂 `mean_abs_delta ≥ 0.5` 且 `frac_negative ≥ 0.5`；A1 必须 `mean_abs_delta = 0` | 不过 → 该臂的 cycle 路径未被模型使用，该臂结论作废 |
| G5 | 方向稳定性 | 四子集上 G1/G2a 的符号至少 **3/4** 一致 | 不过 → 只作子集条件陈述，不进主表 |

**G4 的方向约定（写死）**：`cycle_intervention` 的 `delta` 是**加在 raw cycle 上的 +10**，
`mean_delta = mean(pred(c+10) − pred(c))`。**负值表示预测 RUL 随年龄增加而下降**，这是物理正确方向；
`frac_negative ≥ 0.5` 表示至少一半窗口朝正确方向移动。`mean_abs_delta` 只衡量响应幅度，不含方向。

补充规则：

- **R0 发车即验**：`HC-C-A3p-FD001-s101-W1` 必须复现 `HC-C-A3p-FD001-s101-20261112T0030Z`
  （同 init sha `336074a6…`、`val_engine_mean_rmse = 10.8951`）。不一致 → **立即停波**，先查
  init/数据/代码指纹，不允许继续跑完 24 格。
- **R1 单 seed 限制**：以上只判方向；「显著」一律留给 Gate A 第二波的 3 seed。
- **R2 不挑子集**：G1–G5 必须报四子集全表，禁止只报最好子集。
- **R3 A0 排除**：A0 只进 G3，不进 G1/G2 的 treatment 比较。
