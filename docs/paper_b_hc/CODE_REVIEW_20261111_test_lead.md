# 代码逻辑审计报告 · paper_b/hc（测试负责人 · 第一轮）

- 状态：**已完成（针对下列固定哈希的版本）**
- 最后更新：2026-11-11 22:10 (+08)
- 审计对象：`paper_b/hc/` 全量 8 个文件 + `paper_b/training/` 三处改动（tmh 22:04–22:06 落地）
- 审计方式：只读代码审查 + 本地只读探针（`tmp/audit_hc/test_hc_smoke.py`）+ `git diff` 对照 HEAD + 数据契约复算
- 未修改任何 `paper_b/hc/**`、`experiments/hc/**` 文件；本文件是唯一写入

> ## ⚡ 最新状态（每次更新置顶，主控请先读这一块）
>
> **更新时间：2026-11-12 04:00 (+08)**
>
> | 事项 | 状态 | 证据位置 |
> |---|---|---|
> | **G2 判读规则以预注册为准** | **已提交** `747b033`（Windows 03:16:48 +0800 ≈ a40 03:25:48），**早于 A3a-FD001 的任何结果** | `docs/paper_b_hc/ANALYSIS_RULE_GATE_A.md` |
> | 符号约定「正 = B 优于 A」 | **已统一并落盘** | `CYCLE_DEPENDENCE_STUDY.md` 开头约定块 + 本文 §12.16；判定器代码即此约定 |
> | `NOISE1` 投影噪声 | **已完成**：`\|Δ\| = 0.030123 RMSE` | 本文 §12.16；run `HC-C-A2-FD001-s101-NOISE1` |
> | **σ 记 2.80，不记 3.5** | `σ_p = 0.030123/1.128 = 0.0267`；`σ_Δ = √2·σ_p = 0.0378`；`0.106/0.0378 = 2.80σ`。**阈值处单次检出概率约 50% → G2 报区间，不报 pass/fail** | 本文 §12.18.1；预注册文件 §4 |
> | **判定器位置（v10 决定）** | **留在 `tools/`，不进 `paper_b/hc/`、不进 `OWN` 列表**。理由：评估器必须能独立演化；冻结面应是「产生结果的代码」，不是「评估结果的代码」。溯源靠快照里的 sha256（粒度比锁更细） | `tools/gate_a_judge.py`；主控裁定 |
> | 骨干同源 | **通过**：10 对 × 57 键 × 0 不匹配 | 本文 §12.12 |
> | G1（A1→A2，四子集） | **PASS**：+6.10 / +13.52 / +11.46 / +10.74%，均值 +10.46% | `tools/gate_a_judge.py` |
> | G3（A1→A0） | **PASS**：四子集 −166.8 / −122.7 / −232.4 / −181.0% | 同上 |
> | **G2（A2→A3a）** | FD001 = **−6.84%**（符号反向）；**但 FD001 的通道惰性、A3a≈A1，且惰性随子集变化** | 本文 §12.18 / §12.18.2 |
> | 进度 | **14 格 ok** / 剩 14 格（含在跑），串行 | `cell_lock.py --status` |
>
> **两条必须一起读的限定**：
> 1. **σ = 2.80**（不是 3.5）。按 3.5σ 会以为「阈值很稳」；按 2.80σ，真实效应恰为阈值时单次检出概率约 **50%**。且 A3a 四格共用同一份 init，投影噪声跨子集**完全相关**，`σ/√4` 不成立。**G2 报区间，不报 pass/fail。**
> 2. **σ 的适用性有限**：`σ_p = 0.0267` 是 **A2 通道**的敏感度。A3a 的投影输出 std 只有 A2 的 1/158（0.440 vs 69.665），且 FD001 上 A3a 退化成 A1（+0.32%）。**该 σ 不能直接迁移到 A3a**，见 §12.18。

## 0. 固定版本（本报告的结论只对这套哈希成立）

| 文件 | md5 前 12 | 备注 |
|---|---|---|
| `paper_b/hc/config_hc.py` | `d08e56156690` | |
| `paper_b/hc/cycle_encoder.py` | `f0ddc9115ba4` | |
| `paper_b/hc/data_hc.py` | `995498011015` | |
| `paper_b/hc/model_hc.py` | `7a7ec9e48902` | |
| `paper_b/hc/train_hc.py` | `1e5b6a5fda22` | |
| `paper_b/hc/run_hc.py` | `df664a1aa8bb` | |
| `paper_b/hc/snapshot_val.py` | `74ba7daff1b0` | |
| `paper_b/hc/compare_val_snapshot.py` | `daff09811d4d` | |
| `paper_b/training/data.py` | `c2ee079f6e60` | 已含 R1 修复 |
| `paper_b/training/metrics.py` | `87e8245b49f4` | 已含 `per_engine_rmse` |
| `paper_b/training/train.py` | `07e7379dc2a8` | 已改发动机级选轮 |
| `paper_b/models/dspt.py` | `fb7011023b28` | 未动 |
| `paper_b/configs/protocol.json` | `c55720ffe09d` | 未动 |

> 审计过程中代码仍在变动（`metrics.py` 22:05、`train.py` 22:06、`data.py` 22:04、新增 `compare_val_snapshot.py` 22:04）。
> 若哈希已变，结论需重跑本报告 §3 的探针后再引用。

## 1. 结论摘要

| 级别 | 编号 | 问题 | 后果 |
|---|---|---|---|
| **P0** | A1 | hc 训练器仍按**池化 RMSE** 选 best epoch，并把该值命名为 `val_engine_rmse` | V0（frozen，已改发动机级）与 A0–A5（hc，池化）**选轮口径不同**，任何跨变体对比无效 |
| **P0** | A2 | `A3p` 控制臂文档声称可用，实际 `hc_config('A3p')` 抛 `KeyError` | A3a 的注入点与 V0-Cycle 不同，却缺控制实验；"Fourier 是原 DSPT 形式"无法成立 |
| **P0** | A3 | 任务书 §1 的五个冻结 md5 已被 22:04–22:06 的改动**作废**，a40 副本必然更旧 | 第一批 run 的 `manifest.json:code_md5` 与基准对不上，证据链在起点断开 |
| P1 | B1 | R6 的 `val_engine_ids_hash` 未写入 manifest，也未对四个基准值做断言 | 划分漂移不会被自动拦住（数据侧我已复算，四个 hash 全对） |
| P1 | B2 | `run_hc` 无 launch 记录、无重复 run_id 守卫，`exist_ok=True` 静默覆盖 | 同 run_id 重跑会静默替换证据，正是本项目已经踩过的坑 |
| P1 | B3 | `cfg` 携带 epochs/patience/batch/lr/wd，但 `train_hc` 读模块常量 | `config.json` 可能与实际训练预算不符（单一事实源被破坏） |
| P1 | B4 | `train_hc.py:8` 声称已由 `verify_loop.py` 验证，该文件不存在 | "忠实转录已核对"目前是**无证据声明** |
| P1 | B5 | "长引擎在池化口径下权重可达短引擎约 9 倍"**数值错误** | 该句已扩散到任务书 §5.1、`metrics.py`、`train.py` 文档；写进 Methods 会被审稿人抓 |
| P2 | C1–C7 | 干预实验越界输入、manifest 自述不全、分位聚合口径不一致、`dict.get` 急切求值、>250 处守卫失明、历史探针会抛错、R1 改前快照只能从 git 取 | 局部正确性问题，不影响主结论但会污染报告 |

## 2. 逐条证据

### A1（P0）hc 选轮口径与 frozen 主指标不一致

- 契约：任务书 §5.1「**所有变体（含 V0 三份）一律用主指标选 best epoch**」，主指标 = `val_engine_mean_rmse`；§5.1-3 明说 `val_engine_rmse` 这个名字是池化值、**hc 层必须用新字段名**。
- frozen 侧已改对：`paper_b/training/train.py:147` 选轮用 `per_engine_rmse`，`:167` 同时写 `val_engine_rmse`(兼容别名)、`val_pooled_rmse`、`val_engine_mean_rmse`。
- hc 侧没跟：
  - `train_hc.py:130-132` `val_now()` → `rmse(p, va['y'])`（池化）；
  - `train_hc.py:171` `if vr < best - 1e-4` → 用池化值选轮；
  - `train_hc.py:181` `m.update(val_engine_rmse=rmse(pv, va['y']), ...)` → **池化值 + 被明令禁止的字段名**；
  - `run_hc.py:128` `close.json` 把 `val_engine_rmse` 当头条指标上报给管理员台账。
- 后果：V0-Cycle（走 frozen `train_one`）与 A3a（走 `train_hc`）在同一 seed 下会因为选轮标准不同而停在不同 epoch，二者 RMSE 之差里混入了口径差，Gate A 的"Fourier vs raw"比较被污染。
- 修复（改 3 处，约 6 行）：
  1. `from ..training.metrics import per_engine_rmse`；
  2. `val_now()` 返回 `(pooled, engine_mean, preds)`，选轮用 `engine_mean`；
  3. 落盘 `val_engine_mean_rmse`（主）+ `val_pooled_rmse`（次），`close.json` 上报主指标。
- 验证：同一 run 的 `metrics.json` 里 `val_engine_mean_rmse <= val_pooled_rmse` 恒成立（Jensen），且 `history.json` 每轮两列都在。

### A2（P0）A3p 控制臂不可达

- `model_hc.py:13-15` 文档：「the frozen *injection point* … is available as variant `A3p`」；`model_hc.py:84-85` 也确实有 `if v == 'A3p'` 分支。
- 但 `config_hc.py:62-80` 的 `VARIANTS` 没有 `A3p`；`hc_config()` 第一步就 `variant(variant_name)` → 抛错。实测：

  ```
  FAIL a3p_control: KeyError("unknown variant 'A3p'; known: ['A0','A1','A2','A3a','A3b','A3c','A4','A5']")
  ```

- 为什么这是 P0 而不是小事：`dspt.py:239-244` 把 Fourier 项加在 **LayerNorm 之前**（`pooled + pe_proj(pe)` 然后 `self.norm`）；`model_hc.py:34-38` 的 A3a 把它加在 `encode()` 返回的**post-norm** 表示上，且不再归一化。两者**形式同、注入点不同、尺度不同**。没有 A3p 就无法排除"差异来自注入点而非编码形式"，而 Gate A 的第一条判据正是"Fourier 是否优于简单年龄编码"。
- 修复：`VARIANTS['A3p'] = dict(label='frozen injection point', cycle_mode='none', cycle_source='clipped', cycle_only=False)`（`build_hc` 对 A3p 忽略 `cycle_mode`），并在 Gate A 前跑一次 1 seed FD001 对照，把 Δ 写进 `CYCLE_DEPENDENCE_STUDY.md`。

### A3（P0）冻结 md5 基准已失效

- 任务书 §1 表：`data.py 9441399…`、`train.py 4853414…`、`metrics.py 87c6637…`。
- 现在：`data.py c2ee079f6e60`、`train.py 07e7379dc2a8`、`metrics.py 87e8245b49f4`（22:04–22:06 改动）。
- `run_hc.py:46-51` 会把当前 md5 写进 `manifest.json`，管理员按任务书基准核对时必然不一致；反过来，如果 a40 上还是旧文件，则 manifest 会记录一套**从未在 a40 上跑过**的哈希。
- 修复：改动冻结文件后必须（1）更新任务书 §1 表，（2）重传 a40 并 `md5sum` 核对，（3）在 `RESULTS_REGISTRY.md` 记一条"基线哈希变更"。**在这一步完成前不要发第一个正式 run。**

### B1（P1）R6 划分指纹缺失

- `data_hc.split_fingerprint`（`data_hc.py:87-95`）给的是 `sha1("train_ids|val_ids")`；任务书 §5.2 R6 要求的是 `md5(",".join(sorted(val_ids)))[:16]`，基准 FD001 `f81d7f2496c494ca`、FD002 `cacf4ea1d7cf96b6`、FD003 `f81d7f2496c494ca`、FD004 `eb1b0f00047d393e`。
- 我已按 `split_units` 源码复算，四个值与基准**逐字一致**（数据侧没问题），但代码不做断言 → 将来 `split_seed`/`fold` 漂移不会被拦。
- 修复：manifest 增加 `val_engine_ids_hash`，与常量表比对，不一致直接 `raise`。

### B2（P1）缺 launch/close 协议实现

- 任务书 §11：启动前独占创建 `registry/launch/<run_id>.json`（已存在即重复），结束后写 `registry/close/<run_id>.json`。
- `run_hc.py:88-90` 只做 `out.mkdir(parents=True, exist_ok=True)`，随后无条件覆盖 `config.json`/`train.log`；`close.json` 写在 run 目录里（`:135`），靠人工复制到 registry。
- 后果：同一个 run_id 重跑不会报错，只会把旧证据静默替换。本项目已因"未设种子的整批重跑"付出过一次代价（D023），这条守卫必须进代码而不是靠纪律。
- 修复：`os.open(launch_path, O_CREAT|O_EXCL)`；已存在则拒绝；`out` 非空且无 `--force` 时拒绝。

### B3（P1）配置双源

- `config_hc.hc_config` 把 `epochs/patience/batch/lr/weight_decay` 写进 cfg（`:115-116`），`train_hc.py:23` 却 `from .config_hc import BATCH, EPOCHS, LR, PATIENCE, WEIGHT_DECAY`，训练只用模块常量。
- 现在两者相等，所以不炸；但 `config.json` 声称是"effective config"（`run_hc.py:11`），一旦协议改了 epoch 数，manifest 会说谎。
- 修复：`train_hc` 从 cfg 读，或加一行 `assert (cfg['epochs'], cfg['batch'], ...) == (EPOCHS, BATCH, ...)`。

### B4（P1）`verify_loop.py` 不存在

- `train_hc.py:6-8`：「The transcription is verified against the frozen implementation by `paper_b/hc/verify_loop.py`」。该文件不在 `paper_b/hc/`（现有 8 个文件，见 §0）。
- 我逐行比对过 `train_hc.py:135-177` 与 `train.py:106-158`：loss 三项、系数、clip、cosine、早停阈值、best-state 拷贝逻辑**确实一致**。但"一致"目前是我的阅读结论，不是仓库里的可复跑证据。
- 修复：补 `verify_loop.py`（构造 8×70×14 的假数据，两个 trainer 各跑 1 epoch，比对 loss 序列与最终参数），或删掉该句声明。**不能保留无证据的"已验证"。**

### B5（P1）"9 倍权重"是错的

- 出处：`RESULTS_AUDIT_20261111.md:165` → 任务书 §5.1 注 2 → `metrics.py:30-34` docstring → `train.py:104-107` docstring。
- 事实 1：冻结 val 构造下**每台引擎恰好 5 个窗口**（我按 `make_val_windows` 的 `round(f*n)`+去重复算，FD001/2/3/4 全部 `windows_per_engine=[5]`）。因此
  `pooled² = (1/E)Σ_e MSE_e`，**每台引擎权重相同 = 1/E**，不存在"长引擎行数多所以权重大"。
- 事实 2：实测探针 `metric_identity`：4 台引擎、1 台误差 60，`pooled=31.097`、`rms_of_mse=31.097`（完全相等）、`engine_mean=17.906`。即池化 = 逐引擎 RMSE 的**均方根**，不是加权平均。
- 事实 3：真正存在的是"平方误差质量"差异。按冻结 cap=125 复算每台引擎的 Σy²：FD004 `max/min = 3.94`、FD001 3.28、FD002 3.17、FD003 2.55。"9 倍"只有在**不封顶**（用 (1-f)·n 原始标签）时才成立（FD004 435/128 引擎算得 9.28）。
- 正确表述（建议直接替换）：池化 RMSE 与发动机级平均 RMSE 的差别是**先平方后平均 vs 先平均后平方**（Jensen），池化恒 ≥ 发动机级平均；在冻结 cap 下，最长 FD004 引擎的平方误差质量约为最短的 3.9 倍，因此少数高误差引擎对池化值影响更大。**不要写"权重 9 倍"。**
- 影响：这是 Methods 里一句话的事，但写错会被审稿人用同一个数据反驳。

### C 组（P2）

| 编号 | 位置 | 问题 | 建议 |
|---|---|---|---|
| C1 | `train_hc.py:66-68` | 干预实验在**裁剪之后**加 delta：A3a 的 cycle 已 `min(c,250)`，+10 后得到 260，是训练分布里从未出现的输入 | 在 raw 空间加 delta，再经 `cycle_vector` 重新映射 |
| C2 | `run_hc.py:119` vs `:135` | `manifest.artifacts` 在 `close.json` 写入**之前**生成 → 自述清单漏掉 `manifest.json` 与 `close.json` | 先写 close，再生成 manifest；或把两个名字补进列表 |
| C3 | `train_hc.py:100-102` | 寿命三分位用 `sqrt(mean(rmse_i²))`（又是 RMS），与新主指标口径不一致；20 台引擎被切成 6/6/8 | 改用等权平均或 `np.array_split`，并在报告里注明每组引擎数 |
| C4 | `train_hc.py:43` | `d.get('cycle_hc', d['cycle'])` 的默认值会被**急切求值**，缺 `cycle` 键时即使有 `cycle_hc` 也抛 KeyError | 显式 `d['cycle_hc'] if 'cycle_hc' in d else d['cycle']` |
| C5 | `data_hc.py:57-62` | 守卫用 `min(c/250,1)` 比对，**所有 >250 的窗口都映到 1.0** → 恰好在"无截断年龄"真正生效的区间失明 | 增加逐引擎单调性/偏移断言（`np.all(np.diff(per_engine_raw) == 1)`） |
| C6 | `experiments/probe_selection_metric.py:45` | 省略 `cuts`，R1 修复后必抛 `ValueError`；且 `:51` 把池化值命名为 `eng` | 补 `cuts=tuple(PROTOCOL['val_cuts'])`；重命名变量并注明口径 |
| C7 | `snapshot_val.py:4-6` | 文档要求"改前跑一次、改后跑一次"，但改动已在工作树里，"改前"只能从 `git show HEAD:paper_b/training/data.py`（HEAD=`49ac685`）取 | 用 git 取旧版建临时模块跑 before；否则这段证明无法执行 |

## 3. 已通过的检查（同样重要）

1. **边界未被越界**：审计时刻五个冻结文件与任务书 §1 表逐字节一致（22:04 前）；`paper_b/hc/**` 只 `import` 冻结模块，无写入。
2. **协议守卫可跑**：`assert_protocol_matches_taskbook()` 通过（epochs=40、hi_mono_coef=0.1、cuts 五点、seeds 101/202/303）。
3. **R6 划分正确**：四个子集 `val_engine_ids_hash` 与基准逐字一致；`FD001 == FD003`（同为 `f81d7f2496c494ca`）复现，符合任务书 R7 的"不是四次独立重复"。
4. **R1 修复行为保持**：`git diff` 显示 `data.py` 只改了默认值 `(0.5,0.75,0.9)` → `None` 并加了 `raise`，**没有任何计算路径改动**；所有现存调用（`train.py:85`、`data_hc.py:78`、`snapshot_val.py:45`）都显式传 `cuts`，故张量不变是静态可证的。
5. **val 原始 cycle 重建公式与冻结实现一致**：`_val_raw_cycle` 的 `round(f*n)`、去重、`cyc[c-1]` 与 `make_val_windows:215-226` 逐条对应。
6. **不碰测试集**：`run_hc`/`train_hc`/`data_hc` 无 `prepare_test` 引用；Stage C 无 test 阶段。
7. **A1 的 None 分支安全**：`needs_cycle=False` 时传 `cycle=None`，`dspt.forward:252-253` 会补零向量，不会崩。

## 4. 未覆盖 / 需要下一轮

- 本地 Python 无 `torch`、`sklearn`（`HAS_TORCH False`），**前向与训练未在本地执行**。模型构建类问题（如 `DSPT_HC` 构造参数、`hi_seq` 形状）只能靠阅读判断，必须在 a40 上跑 A0–A5 各 1 epoch 冒烟后才算"能跑"。
- 未验证 `hi_head` 在 `DSPT_HC` 下 `hi_seq` 的 mask 对齐（继承自 frozen，风险低）。
- 未做同 seed 两次重复的噪声底线测量。鉴于 `train_hc.py:122` 与 `train.py:94` 都设了 `cudnn.benchmark=True`，**同 seed 不保证逐位可复现**。Gate A 第一波是单 seed 判断方向，建议先对 1 个 cell 跑两遍，测 `max|Δpred|` 噪声底线（与 MCD80 的做法一致）。

## 5. 需要主控（L2）裁决的契约冲突

| 编号 | 冲突 | 建议 |
|---|---|---|
| X1 | 任务书 §2.1「不改 `paper_b/training/*.py`」 vs §5.2「允许改 `data.py` cuts 默认值、`metrics.py` 只允许新增 `per_engine_rmse`」 | §2.1 加一条显式白名单，否则 tmh 的改动处于"已违反 §2.1"的状态 |
| X2 | §6 要求「保持**原 DSPT 骨干**」，但 `config_hc.BACKBONE` 是 `single_proj/basic/hf`（factorial 选出的候选，丢掉 MS 多尺度），而 §5 又要求"必须保留 MS 多尺度卷积" | 明确 Stage C 到底在哪个骨干上做；若用候选骨干，则 Stage C 结论只能表述为"候选骨干上 cycle 的作用"，且不能与 V0 基线同表 |
| X3 | §3 的 `V0-NoHI` 与冻结默认 `hi_source='none'` 重叠 → `V0-Cycle` 与 `V0-NoHI` 可能是同一个模型 | 给出 V0 三份的精确 `(ms_mode, sa_mode, hi_source)` 三元组 |
| X4 | 正在跑的 `paperb-s23-seeds-20261111` 用的是 22:04 之前的 `train.py`（池化选轮） | 其 24 个 run 标记为 legacy 口径，或在新口径下重跑；不得与 hc 结果同表 |

---

# 第二轮复核（2026-11-11 22:40）

对 tmh 22:07–22:35 修复的独立验证。方法同第一轮：读文件 + `git diff` + 数据契约复算，不采信转述的 grep 快照。

## 6.1 已关闭

| 编号 | 证据 | 判定 |
|---|---|---|
| A1 | `train_hc.py:160-163` 返回 `(pooled, engine_mean, p)`；`:203` 用 `eng` 选轮；`:215` 落盘 `val_engine_mean_rmse`+`val_pooled_rmse`；`:219` 加 `select_on_val=True`；`run_hc.py:193` `main_metric='val_engine_mean_rmse'`；全树已无 `val_engine_rmse` 字段 | 关闭 |
| A2 | `config_hc.py:97-99` 新增 `A3p`（`frozen_injection=True`, `cycle_source='clipped'`）；`model_hc.py:102-103` 按 flag 走 `DSPT_HC_Frozen` | 关闭（等价性见 6.2） |
| B1 | `data_hc.py:106` 写 `val_engine_ids_hash`；`run_hc.py:126-130` 比对 `SPLIT_HASH_REF` 后 `SystemExit`；launch/close/manifest 三处都带 | 关闭 |
| B2 | `run_hc.py:89-99` `O_CREAT\|O_EXCL`；`:146-148` 目录非空且无 `--force` 拒绝 | 关闭 |
| B3 | `train_hc.py:131-138` 从 cfg 读并断言与协议常量一致；`budget_override` 显式豁免 | 关闭 |
| B4 | `verify_loop.py` 存在；monkey-patch 目标正确（`FT.prepare_train` 是模块全局名，`DH.build` 是同一模块对象）、同 seed、dropout=0、2 epoch、比较 loss 序列+参数字典键集 | 关闭（待 a40 实跑） |
| C1 | `train_hc.py:74-94` 在 `cycle_raw` 上加 delta 后经 `cycle_vector` 重映射 | 关闭 |
| C2 | `run_hc.py:202-214` manifest 最后写，含 `close.json` | 关闭 |
| C3 | `train_hc.py:117-121` `np.array_split` + 等权平均 | 关闭 |
| C4 | `train_hc.py:50-52` `_cycle_of` 不再急切求值 | 关闭 |
| C5 | `data_hc.py:31-33,44-45` 原始 cycle 严格递增断言 | 关闭 |

## 6.2 指定审计项：A3p 是否等价于 frozen 的 pre-norm 注入 —— **成立**

三步论证，逐环闭合：

1. **标量**：`data_hc.cycle_vector('clipped')` 给的是 `min(c,250)`（单位 cycles）；`DSPT_HC_Frozen.encode`（`model_hc.py:61-63`）做 `clamp(c,0)/250` → `min(c/250,1)`；frozen `DSPT.encode`（`dspt.py:240`）再 `*250` → `min(c,250)`。与 frozen 数据层「存 `min(c/250,1)` 再乘 250」**逐位同值**（c≤250 时是同一个除法，c>250 时两边都得 1.0）。
2. **注入点**：`DSPT_HC_Frozen` 只覆写标量换算，`super().encode` 未改 → 仍在 `pooled + pe_proj(pe).unsqueeze(1)` 之后 `self.norm(pooled)`，即 frozen 的 pre-norm 位置。
3. **形式与预算**：两者都用 `cat([sin(ang), cos(ang)])` 同序、同一周期库 `(25,50,125,250)`；A3a 的 `FourierCycle.proj` 与 A3p 的 `pe_proj` 同为 `Linear(8,64)` = 576 参数，参数量相等。

结论：**A3a 与 A3p 是单因子（注入点 pre-norm vs post-norm）对照，可以放行。** A3a 的编码器本身与 frozen 的 `pe` 计算数值同一，差异只来自「加在 norm 前还是 norm 后」，这正是要测的东西。

建议把这条从论证升级为可跑断言：在 `verify_loop.py` 增加一段——`build_dspt(use_cycle=True, ...)` 与 `DSPT_HC_Frozen(...)` 同参构造，断言 `state_dict` 键集相同、同输入 forward 的 `max|Δprediction| == 0`。这样 B4 一次覆盖「循环转录一致」+「注入点等价」，Gate A 的对照臂就有机器可验的底座。

## 6.3 第二轮新发现

| 级别 | 问题 | 证据 | 建议 |
|---|---|---|---|
| **P1** | `V0_TRIPLES` 是**死配置**：只有定义，全仓库无消费者；`hc_config` 不接受 `V0C/V0N/V0H`；`BACKBONE` 硬编码 `hi_lambda=0.0`，而 V0C/V0N 需要 `hi_source='hf'` + `hi_lambda=0.5`，当前没有任何路径能产出 | `config_hc.py:51-58`；`grep V0_TRIPLES` 仅命中定义处 | 二选一：①把 V0C/V0N 加进 `VARIANTS` 并允许每变体覆盖 `backbone`（含 `hi_lambda`）；②明确 V0 走 `paper_b/training/train.py::train_one`，并在 `RESULTS_REGISTRY.md` 写明两套产物格式差异（V0 无 launch/close/manifest）。**不选就仍然是「声称有 V0、实际跑不出来」** |
| P2 | A3p 的 `config.json` 写 `cycle_mode='none'`，实际用 Fourier（frozen 注入）。管理员读 config 会误判成「无 cycle」 | `config_hc.py:155` | 写 `effective_cycle_mode`，或对 `frozen_injection` 写 `'fourier_frozen'` |
| P2 | `scaled500` 注释残留，`cycle_vector` 不支持该 source | `config_hc.py:72` vs `data_hc.py:75-82` | 删注释或补实现 |
| P2 | 训练抛错时没有 `close.json`：registry 留下 launch 无 close，且同 run_id 永久被 `O_EXCL` 拒绝 | `run_hc.py:163-184` 无 try/except | 包 try/except，失败时写 `status='failed'` + 异常摘要，保持 launch/close 成对 |
| P2 | `train_hc.py:220` 的 Jensen 断言隐含「每引擎窗口数相同」（val 恰好 5/引擎才成立）。将来若 val 构造改成长短不一，断言会误杀已训完的 run | `train_hc.py:220-221` | 注释写明前提，或按同权重口径比较 |
| P2 | `manifest.artifacts` 仍不含 `manifest.json` 自身（计算在写入前） | `run_hc.py:211` vs `:214` | 补一行或接受 |
| P2 | `SMOKE` 变体与 `--smoke` 标志两套机制并存，`smoke_of` 未传到 cfg | `config_hc.py:100-104`；`run_hc.py:116,141-143` | 删变体，只留 `--smoke` |

## 6.4 仍未验证（a40 必做）

1. `python -m paper_b.hc.verify_loop` → 必须打印 `VERIFY_LOOP PASS`（本地无 torch，未跑）。
2. A0–A5 + A3p 各 1 epoch `--smoke`，确认前向/反传/产物链路。
3. 同 seed 两次重复的噪声底线：`cudnn.benchmark=True` 下同 seed 不保证逐位复现，Gate A 第一波是单 seed 判方向，必须先知道 `max|Δpred|` 有多大（做法同 MCD80 的两遍抽样噪声底线）。
4. 更新后的冻结 md5 表与 a40 上 `md5sum` 逐一核对（第一轮 A3）。

---

# 第三轮复审（2026-11-11 22:45，只读，按当前哈希）

复审范围：A1 / A2 / B2 / B3 / B4。方法同前：读文件 + `git diff` + 本地探针（`tmp/audit_hc/test_hc_smoke.py`）。

## 7.1 本次复审固定的哈希（第三版）

| 文件 | md5 |
|---|---|
| `paper_b/training/train.py` | `13ec2fbb75ed09a389e4e979a82e11e6` |
| `paper_b/training/metrics.py` | `39202682865d14dc6be5d332ac7a1c16` |
| `paper_b/training/data.py` | `c2ee079f6e60a8b0d0eb608676ff871a` |
| `paper_b/models/dspt.py` | `fb7011023b284a5f571cba4d4abb6927` |
| `paper_b/configs/protocol.json` | `c55720ffe09deff3a9b59c218a180114` |
| `paper_b/hc/config_hc.py` | `64bf6d5543f7fae6a8cd05cf254b2bcd` |
| `paper_b/hc/data_hc.py` | `3ad7c111cf0268409bb40d79f7271cee` |
| `paper_b/hc/model_hc.py` | `adcbceb901ae59d2bd15bbfc4067fc3f` |
| `paper_b/hc/train_hc.py` | `64b90518c42cff6e36074f3967d51caa` |
| `paper_b/hc/run_hc.py` | `959a490fce77b419bbb96cb822517742` |
| `paper_b/hc/verify_loop.py` | `bf3022ba60a804bdbd7b2ead0a38577c` |

> 注意：22:36–22:37 之后 `train.py`/`metrics.py` 又变过一次。若管理员按更早的 `ef14ee21…`/`a799d43f…` 定基线，**已经过期**。

## 7.2 复审矩阵

| 项 | 结论 | 关键证据 |
|---|---|---|
| **A1** 主指标口径 | **通过** | `train_hc.py:160-163` 返回 `(pooled, engine_mean, p)`；`:203` 用 `eng` 选轮；`:215-219` 落 `val_engine_mean_rmse`+`val_pooled_rmse`+`select_on_val=True`；`run_hc.py:193` `main_metric='val_engine_mean_rmse'`。frozen 侧 `train.py:103-115` 同样用 engine-mean 选轮。两侧口径现已一致 |
| **A2** A3p 控制臂 | **通过** | 探针实测 `hc_config('A3p')` 返回 `frozen_injection=True, cycle_source='clipped'`；`model_hc.py:102-103` 据此走 `DSPT_HC_Frozen`；与 A3a 共享同一 clipped 标量，单因子（注入点）对照成立（等价性论证见 §6.2） |
| **B2** launch/重复守卫 | **通过** | `run_hc.py:89-99` `O_CREAT\|O_EXCL`；`:146-148` 目录非空且无 `--force` 拒绝；`:163` launch 在训练前写入 |
| **B3** 配置单一事实源 | **通过** | `train_hc.py:131-138` 从 cfg 读五个预算项并与协议常量断言；`budget_override` 显式豁免并写进 config.json |
| **B4** 循环转录已验证 | **通过** | `verify_loop.py` 存在且逻辑正确（monkey-patch 目标正确、同 seed、dropout=0、2 epoch、比较 `train_loss` 序列 + 参数字典键集）。22:37 的 `train.py` 改动只涉及 docstring / `val_metrics` 签名 / history 字段，**训练循环体（loss、clip、optimizer、schedule）未变**，故基准仍然有效。仍未在 a40 实跑 |
| **B5**「9 倍」表述 | **已改对** | `metrics.py:30-45` 与 `train.py:104-113` 已替换为正确表述：每引擎恰好 5 窗口 → 池化等权 1/E；差异是聚合次序（Jensen，池化恒 ≥ 发动机级平均）；平方放大尾部。与我第二轮的实测一致 |

本地探针（当前代码）：

```
PASS protocol_guard
PASS a3p_control  frozen_injection=True  cycle_source=clipped
PASS data_contract  FD001/2/3/4 hash = f81d7f2496c494ca / cacf4ea1d7cf96b6 /
                    f81d7f2496c494ca / eb1b0f00047d393e；windows_per_engine=[5] 全同
variants: A0 A1 A2 A3a A3b A3c A3p A4 A5 SMOKE
所有变体 backbone 均为 full/basic/none（X2 裁决已生效）
```

## 7.3 仍未关闭

| 级别 | 问题 | 证据 | 影响 |
|---|---|---|---|
| **P1** | **V0 三份仍无代码路径**：`V0_TRIPLES` 只有定义，`VARIANTS` 里没有 `V0C/V0N/V0H`，`hc_config` 会 KeyError；且 `BACKBONE` 硬编码 `hi_lambda=0.0`，V0C/V0N 需要的 `hf + 0.5` 无处可配 | `config_hc.py:51-58`；探针 `V0 in VARIANTS: []` | Gate A 表里的 V0 行目前跑不出来（A3p 可充当 V0-NoHI，但 V0-Cycle / V0-NoCycle 不行） |
| **P1** | **基线未提交，md5 表第三次过期**：`git log -1 = bc75aa6`，`data.py/metrics.py/train.py` 仍是 modified，`paper_b/hc/` 仍是 untracked。`run_hc.env_info()` 记录的 git SHA 指向**不含本次改动**的 commit | `git status`；`git diff --stat` 三个文件 79 插入 | 今天已经出现三次「表刚写就过期」。建议先提交一个 baseline commit，再把 md5 钉到该 commit 的 blob 上；否则管理员的复核永远追不上工作树 |
| P2 | `history.json` 字段名两侧不一致：frozen 用 `val_rmse`（实为 engine-mean）+ `val_pooled_rmse`，hc 用 `val_engine_mean_rmse` + `val_pooled_rmse` | `train.py:148-149` vs `train_hc.py:198-199` | 将来合并曲线会错列 |
| P2 | A3p 的 `config.json` 写 `cycle_mode='none'`，实际用 Fourier（frozen 注入） | `config_hc.py:155` | 审计者读 config 会误判 |
| P2 | 训练抛错时无 `close.json`，registry 留 launch 无 close，同 run_id 永久被拒 | `run_hc.py:163-184` 无 try/except | 失败 run 在台账里无法收口 |
| P2 | `manifest.artifacts` 不含 `manifest.json` 自身；`SMOKE` 变体与 `--smoke` 双机制并存；`scaled500` 注释残留 | `run_hc.py:211`；`config_hc.py:72,100-104` | 清洁度 |

## 7.4 复审结论

**A1/A2/B2/B3/B4 五项全部通过（只读复审），可以进入 SMOKE。** 进 Gate A 前只剩两件事：①V0 三份的代码路径要定（否则 Gate A 表缺 V0 行）；②baseline 先提交、哈希钉到 commit，再让管理员定基线。

SMOKE 的验收点（我收到 run_id 后按这个核）：`launch.json` 与 `close.json` 成对且都在 registry；`manifest.code_md5` 与 §7.1 表逐字一致；`metrics.json` 同时含 `val_engine_mean_rmse` 与 `val_pooled_rmse` 且前者 ≤ 后者；`preds.csv` 行数 = 验证窗口数；`history.json` 两列齐全；`close.json.main_metric == 'val_engine_mean_rmse'`。

---

# 第四轮：SMOKE 验收与比值争议（2026-11-11 22:50）

## 8.1 「3.94 vs 4.86」争议已解决：是总体不同，不是算错

tmh 复算得 4.16/4.28/3.79/4.86，我此前报 3.28/3.17/2.55/3.94。两种都复现了，差别只在**引擎总体**：

| 子集 | 验证引擎（round，封顶） | 全体引擎（round，封顶） | 验证引擎（不封顶） |
|---|---|---|---|
| FD001 | **3.26** | 4.16 | 4.37 |
| FD002 | **3.17** | 4.28 | 4.85 |
| FD003 | **2.54** | 3.79 | 3.85 |
| FD004 | **3.93** | 4.86 | 9.79 |

- 我的数 = `Σ_c min(n-c,125)²` 在**验证引擎**上的 max/min（我第一版用 `(1-f)·n` 近似，差 0.01–0.02，可忽略）。
- tmh 的数 = 同一公式在**全体引擎**（train∪val）上的 max/min。
- 两者都对。但 docstring 讨论的是 **validation pooled metric**，总体应是验证引擎 → 用 **3.26/3.17/2.54/3.93**，或者干脆不写比值（只留 Jensen 机制）。写"全体引擎 4.86"会把训练集的长寿引擎算进来，与 val 指标不对应。

## 8.2 SMOKE 验收：**条件不通过（哈希不符）**

产物侧是齐的，但 `launch/close` 记录的 `code_md5` 与当前仓库不一致：

| 文件 | SMOKE 记录（a40 上实际跑的） | 当前本地（§7.1） |
|---|---|---|
| `model_hc.py` | `99888134af4d…` | `adcbceb901ae…` |
| `train_hc.py` | `1e1580b189b6…` | `64b90518c42c…` |
| `run_hc.py` | `9cb09d6010be…` | `959a490fce77…` |
| `training/train.py` | `ef14ee21cf8d…` | `13ec2fbb75ed…` |
| `training/metrics.py` | `a799d43f1c35…` | `39202682865d…` |

`config_hc.py` / `data_hc.py` / `cycle_encoder.py` / `data.py` / `dspt.py` / `protocol.json` 六项一致。

结论：**冒烟证明的是 22:31 那一版管线能跑，当前版本从未跑过。** 两个 run 的 launch 都在 22:32/22:36，本地 `model_hc` 22:34:30、`train_hc` 22:31:01 都已早于冒烟，说明 a40 在冒烟时用的是旧副本（或上传未覆盖到 hc 目录）。Gate A 前必须用当前哈希**重跑一次 SMOKE**。

## 8.3 冒烟记录自身有一处矛盾

`T1431Z`（崩）与 `T1436Z`（过）的 `code_md5` **完全相同**，其中 `model_hc.py` 都是 `99888134…`。但 close 注记说崩溃原因是 `model_hc.encode` 缺 `unsqueeze(1)`，而修复后重跑通过。同一个哈希不可能既缺 `unsqueeze` 又能跑通。三种解释：①修的是另一个文件而注记写错；②记录哈希时 a40 已是修复版、崩溃实际另有原因；③记录被手工改过。**不论哪种，这次冒烟的可追溯性不成立**，以一次干净重跑为准。

## 8.4 失败 run 的 close 是手工补的——正好证实 P2

`T1431Z` 的 `close.json` 有 `status=failed`、`artifacts` 只有三件、无 manifest。`run_hc` 在训练抛错时不会走到写 close 的分支（`run_hc.py:163-184` 无 try/except），所以这条 close 是人工补的。单次可以人工兜，**Gate A 一批 24 个 run 不能靠人工兜**；批量前把 try/except 补上。

## 8.5 通过项（产物与字段侧）

- launch/close 四个文件成对存在，失败 run 有 `failed` 状态与 `retry_of`/`retry_reason` 链。
- 成功 run：`main_metric='val_engine_mean_rmse'`；`val_engine_mean_rmse=62.5931 ≤ val_pooled_rmse=63.6472`（Jensen 断言过）；`artifacts` 八件齐全；`data_split.val_engine_ids_hash=f81d7f2496c494ca` 与基准一致；`git_sha=WORKTREE`（a40 无 git，符合预期）。
- `budget_override` 写进 config，1 epoch 明示"not evidence"。

## 8.6 第四轮结论

- 比值争议：**关闭**。用验证引擎的 3.26/3.17/2.54/3.93，或删掉比值。
- SMOKE：**条件不通过**。产物与字段合格，但代码版本不是当前版本，且两条 launch 的哈希自相矛盾。
- Gate A 放行条件：①用 §7.1 的 11 个哈希重跑一次 SMOKE，`manifest.code_md5` 逐字一致；②`run_hc` 补 try/except 写 failed close；③V0 三份的代码路径定案；④baseline 提交后哈希钉到 commit。

---

# 第五轮：新冒烟 T1447Z 验收（2026-11-11 22:58）

tmh 在 22:47 用当前代码重跑了一次冒烟（`HC-SMOKE-FD001-s101-20261111T1447Z`）。我在 a40 上直接核对，不采信转述。

## 9.1 结论：**通过**

| 验收项 | 实测 | 判定 |
|---|---|---|
| a40 上五个关键文件 md5 | model_hc `adcbceb9`、train_hc `64b90518`、run_hc `959a490f`、train.py `13ec2fbb`、metrics.py `39202682` | 与本地、与 §7.1 逐字一致 |
| `manifest.code_md5` | 14 项全含（含 verify_loop、snapshot_val） | 与 §7.1 一致 |
| `preds.csv` 行数 | 101 = 表头 + 100（FD001 20 引擎 × 5 截断） | 符合 |
| `close.json` 主指标 | `main_metric='val_engine_mean_rmse'` | 符合 |
| Jensen | 62.593074 ≤ 63.647171 | 成立 |
| launch/close 成对 | 两个文件都在 a40 registry | 符合 |
| artifacts | 八件齐全 | 符合 |

**Gate A 的冒烟门通过。** 22:36 那条 T1436Z 记录的是旧代码，已被 T1447Z 取代，不再作为证据。

## 9.2 但 registry 出现主机间分叉（P1）

| 主机 | launch/ | close/ |
|---|---|---|
| a40（权威运行地） | T1436Z, **T1447Z** | T1436Z, **T1447Z** |
| 本地仓库 | **T1431Z**, T1436Z | **T1431Z**, T1436Z |

- 失败 run `T1431Z` 的 launch 与 close 在 a40 上**已不存在**，只留在本地仓库里；而它的 run 目录 `experiments/hc/HC-C-SMOKE-…T1431Z/` 还在 a40 上。
- 通过的新冒烟 `T1447Z` 的记录在 a40 上，本地还没有。

后果两条：

1. **O_EXCL 守卫对 T1431Z 失效**——那条 run_id 在 a40 上重新可用，而失败记录已经从审计链上消失。这跟项目"结论被推翻只追加撤回说明、不删除旧记录"的规则直接冲突。
2. **管理员无法从单一来源登记**：按本地 registry 会漏掉 T1447Z，按 a40 会漏掉 T1431Z。

建议：registry 以 a40 为权威、只追加不删除；把 T1447Z 同步回本地；T1431Z 若要作废，追加一条撤回说明而不是删文件（或恢复其 launch/close）。

## 9.3 Gate A 放行状态

| 门 | 状态 |
|---|---|
| 冒烟门 | **通过**（T1447Z） |
| 代码门（A1/A2/B2/B3/B4） | **通过** |
| V0 三份代码路径 | 未定（P1，`V0_TRIPLES` 仍是死配置） |
| baseline 提交 / 哈希钉到 commit | 未做（P1） |
| `run_hc` 失败路径 try/except | 未做（批量 24 run 前必须补） |
| registry 主机分叉 | 未清（P1，见 9.2） |

---

# 第六轮：按 HEAD blob 复审（2026-11-11 23:05）

主控已提交 baseline：**HEAD = `f3142687ed6c04efcfa35173772f532b3ebb3c38`（22:49:05，父 `bc75aa6`）**。§7.3 与 §9.3 里"baseline 未提交"一条**作废**。

## 10.1 HEAD 与工作树的实际差异

| 文件 | HEAD blob | 工作树 | 状态 |
|---|---|---|---|
| `hc/config_hc.py` | `0f86571997d7…` | `bffe552fefa4…` | 已改（未提交） |
| `hc/model_hc.py` | `adcbceb901ae…` | 同 | 一致 |
| `hc/run_hc.py` | `959a490fce77…` | `d1c4d87e9456…` | 已改（未提交） |
| `hc/train_hc.py` | `64b90518c42c…` | `3fb058441ffe…` | 已改（未提交） |
| `training/train.py` | `13ec2fbb75ed…` | 同 | 一致 |
| `training/metrics.py` | `39202682865d…` | 同 | 一致 |

主控 v3 表里的 `config_hc 0f865719` 与 HEAD blob 一致 ✓；但工作树已经走到 `bffe552f`。**v3 只对 `f314268` 成立。**

## 10.2 A3p pre-norm 等价性复核（主控指定项）：**成立，不退化成无 cycle**

按 HEAD blob 逐环核：

| 环节 | 证据（HEAD 版本） | 结论 |
|---|---|---|
| A3p 定义 | `config_hc.py:97-99`：`cycle_mode='none', cycle_source='clipped', frozen_injection=True` | 标量与 A3a 同为 `min(c,250)` |
| 配置传播 | `config_hc.py:152` `frozen_inj = bool(v.get('frozen_injection', False))`；`:162` 写入 cfg 顶层 | 不会被丢掉 |
| 分支顺序 | `model_hc.py:102-103` **先判** `cfg.get('frozen_injection')` → `DSPT_HC_Frozen`；`:104` 才是 `DSPT_HC(cycle_mode=...)` | `cycle_mode='none'` 永远轮不到，A3p 不会退化成无 cycle 模型 |
| 模型 | `model_hc.py:58` `super().__init__(use_cycle=True)`；`:62` `clamp(cycle,0)/250` | 走 frozen 的 `pe_proj` 路径 |
| 注入点 | `dspt.py:239-244`：`c_abs = cycle*250` → `ang=2πc_abs/periods` → `pooled + pe_proj(pe).unsqueeze(1)` → `self.norm` | **pre-LayerNorm，原位置** |

标量链：`min(c,250)` → `/250` → `min(c/250,1)` → frozen `*250` → `min(c,250)`，与 A3a 输入逐位同值。参数量：`pe_proj` 与 A3a 的 `FourierCycle.proj` 同为 `Linear(8,64)`=576。**A3a vs A3p 是单因子对照，成立。**

## 10.3 B1 复核：已落地，无缺断言；但我上一版报告有一处写错

HEAD 版 `run_hc.py`：

- `:129` `raise SystemExit(f'SPLIT DRIFT on …')` —— 启动前重算并拒绝，比断言更强；
- `:159` launch 的 `data_split.val_engine_ids_hash`；
- `:206` manifest 的 `val_engine_ids_hash`。

**更正**：§7 复审矩阵里我写"launch/close/manifest 三处都带"是错的——`close.json`（`:186-193`）没有这个字段，只有 launch 与 manifest 有。B1 本身不受影响（启动守卫 + 两处留痕足够），但我的表述要改。

## 10.4 工作树三处未提交改动：不触碰已审逻辑

| 文件 | 改动 | 影响 |
|---|---|---|
| `config_hc.py` | 新增 `injection` 字段（A3p 写 `'frozen_pre_norm'`） | 正是 P2-2，纯展示字段，不影响训练 |
| `run_hc.py` | launch 加 `written_by='run_hc'` + `process_start_utc`；写完回读自检（code_md5/written_by 不符即 SystemExit）；`try/except` 写 `status='failed'` 的 close | 主控③ + 我 P2-1；`O_EXCL` 与 R6 守卫未动 |
| `train_hc.py` | 仅新增注释块（字段词表统一） | 无逻辑变化 |

结论：这三处提交后可直接采信，**但必须先提交再跑**；在提交前起的 run 一律按脏树处理。

## 10.5 两条治理建议

1. **基准要写死 commit SHA，不要写"HEAD"**。写成"基准 = `git show HEAD:<path>`"时，下一次提交会静默改变基准；应写成"v3 基准 = commit `f314268` 的 14 个 blob"，新提交显式升 v4 并登记。今天哈希表已经漂了四次，根因是基准绑在可变引用上。
2. **a40 无 git，`git_sha` 恒为 `WORKTREE`**。主控新加的验收项"`launch.code_md5` 与 `git show <run 的 git_sha>:<path>` 一致"在 a40 上无法执行；应改成"与本地 `git show f314268:<path> | md5sum` 一致"。

---

# 第七轮：v9 基线、共享初始化与确定性（2026-11-12 00:40）

## 11.1 v9 是第一个自洽基线

- `HEAD = 6c405ad`（v9 锁，lock-only），代码提交 `d1e0527`。
- 锁 `baseline_version=v9`、`git_sha=d1e0527`、15 文件，与 `git cat-file blob 6c405ad:<path>` **逐个一致，0 不匹配**。
- `python -m paper_b.hc.code_lock --check` → **CODE LOCK OK，退出码 0**，且是在 Windows CRLF 工作树上跑的（`d1e0527` 的换行归一化生效）。
- a40 已同步：`train_hc 069f8977`、`run_hc 8784311b`、`model_hc c55eb2b4`。

**教训（今晚重复三次）**：所有"锁过期"的警报里至少两次是假的，根因分别是 PowerShell 管道转码、Windows CRLF、以及取 `git show HEAD:` 时 HEAD 已前进。**规矩：取 blob 一律用显式 SHA，且该 SHA 从被核对的那个锁文件里读。**

## 11.2 v9 smoke 验收：通过

`HC-SMOKE-FD001-s101-20261112T0045Z`：`written_by=run_hc`、`worktree_dirty=false`、`baseline_version=v9`；15 个 `code_md5` 与 v9 锁 **0 不匹配**（manifest 拷回本地比对）；`main_metric=val_engine_mean_rmse`；`62.593074 ≤ 63.647172`（Jensen）；`preds.csv` 101 行。

## 11.3 共享初始化 2×2：两对逐位相同，σ = 0

| variant | run | val_engine_mean_rmse |
|---|---|---|
| A3a | `…T1640Z` | 11.316705661052403 |
| A3a | `…T0030Z` | 11.316705661052403 |
| A3a | `…R1-0035Z` | 11.316705661052403 |
| A3p | `…T1508Z`（v6） | 10.895096839289387 |
| A3p | `…T0005Z`（v8） | 10.895096839289387 |
| A3p | `…T0030Z`（v9） | 10.895096839289387 |
| A3p | `…R1-0035Z` | 10.895096839289387 |

- 同 seed 重复（含不同 init dump 文件、不同 run 目录）**逐位相同** → 训练确定性成立，同 seed 噪声底线 **σ = 0**。
- A3p 在 v6/v8/v9 三个版本上数值完全一致 → 这几版改动（init remap、CRLF、锁）没有触碰 A3p 的前向与训练路径。

## 11.4 Δ 方向翻转，证明初始化控制必要

- v9（共享初始化）：A3a `11.3167` vs A3p `10.8951` → **A3a 差 +0.4216（3.87%）**，frozen pre-norm 注入更好。
- v6（未共享初始化）：A3a `10.5455` vs A3p `10.8951` → A3a 好 3.2%。
- 结论：v6 的"A3a 更好"是 `pe_proj` 导致的 RNG 流错位造成的假象，不是注入点效应。若不做共享初始化控制，论文会把结论写反。

## 11.5 未闭合项

- `A3p-…T1640Z` 无 `close.json`（进程已退）→ unclosed。
- `A3p-…T0105Z`（测试负责人启动的重复，按裁定不计入）被 `pkill` 终止 → unclosed。
- 两个 init dump（`init_A3p_FD001_s101.pt` / `init_v9_A3p_s101.pt`）sha256 不同但训练结果逐位相同；推断差异只在 payload 元数据（`created_utc`/`code_md5`），**尚未逐张量断言 state_dict 相等**。写进论文前应补一条断言，不能只靠"结果相同"反推。

## 11.6 对 Gate A 的含义

- 代码基准：v9（待主控签发）。
- 主臂 A3p，敏感性臂 A3a；首条 `A3p-FD001-s101-W1` 必须复现 `10.895096839289387`，否则说明 Gate A 配置与 2×2 不一致，应停波排查而不是继续铺量。

---

# 第七轮：测试负责人独立实跑（2026-11-11 23:10）

第六轮对 A3p 的结论是静态链路推理。本轮补**可复跑的数值证据**，并更正一处说法。

## 11.1 A3p vs frozen：同权重实跑

在 a40 上构造 `DSPT(use_cycle=True)` 与 `DSPT_HC_Frozen`，`load_state_dict` 同一份权重，同输入比较 `encode`（脚本经 stdin 传入，未写任何项目文件）：

```
contract c<=250 max|diff| = 0.000e+00
435 via correct clipped path max|diff| = 0.000e+00
435 passed raw to DSPT_HC_Frozen max|diff| = 1.812e+00
```

前两行是 A3p 的真实契约：`cycle_source='clipped'` 下 c≤250，435 的引擎被映射为 250；与 frozen `min(c/250,1)` 路径是同一串 float32 除/乘，逐位一致。第三行是反证：绕过 `cycle_vector` 直接把 raw 435 传给 `DSPT_HC_Frozen`，它不截断，与 frozen 差 1.812——这正是 11.2 要补的类内守卫。第六轮静态链结论**成立**，现在有实跑背书。

## 11.2 新 P2-1：类内不强制 250 上界

`DSPT_HC_Frozen.encode`（`model_hc.py:61-63`）只 `clamp(min=0)`，不 clamp 250。它假设调用方已按 `cycle_source='clipped'` 传入 ≤250；A3p 的 config 确实固定 clipped，`cycle_intervention` 也经 `cycle_vector` 重映射，所以当前无活 bug。但把 raw 435 直接传给该类会静默算 435 而非 250（与 frozen 的 1.0 输入相差 1.81）。建议类内 `torch.clamp(cycle, 0.0, 250.0)` 或 `assert cycle.max() <= 250`：等价性应由类保证，而不是靠调用方纪律。

## 11.3 新 P2-2：A3a 与 A3p 同 seed 的骨干初始化流不同

第六轮说“A3a vs A3p 是单因子对照”。**架构/注入点层面成立，初始化层面不成立**：

- A3p：`DSPT.__init__` 在 `dspt.py:118-119` 创建 `pe_proj`，位置在 `temporal_pe`、`layers`、`sensor_attn`、`sensor_score`、`reg_head` **之前**；
- A3a：`DSPT(use_cycle=False)` 跳过 `pe_proj`，`FourierCycle.proj` 在整个骨干**之后**创建。

同 seed 下，A3p 的 `pe_proj` 消耗的 RNG 会让后续所有共享骨干权重与 A3a 不同。所以单 seed 的 A3a−A3p 差 = 注入点 + 初始化噪声。A2/A3a/A4/A5 之间没有这个问题（都先建同一个 `DSPT(use_cycle=False)`，编码器最后建），编码形式比较是干净的。建议：A3a vs A3p 对照跑 3 seed，或训练前把 A3p 的共享骨干权重显式复制给 A3a，并在 `CYCLE_DEPENDENCE_STUDY.md` 写明。

## 11.4 verify_loop 仍缺 A3p 断言

`verify_loop.py` 只覆盖 A1（无 cycle）的循环转录，不含 A3p/frozen 等价性。建议把 11.1 的检查（约 10 行）并入，让 Gate A 的控制臂有机器可验底座。

本轮我在 a40 重跑了 `python -m paper_b.hc.verify_loop`：`VERIFY_LOOP PASS`，46 个参数张量，`max |loss diff| = 0.000e+00`、`max |param diff| = 0.000e+00`——B4 这一项是独立复现，不是转述 tmh。

## 11.5 1500Z smoke 复核（v3 基线上唯一应登记的 smoke）

`HC-SMOKE-FD001-s101-20261111T1500Z`：registry launch/close 成对；launch 与 manifest 的 14 个 `code_md5` **逐项相同且全等于 HEAD**（含 `config_hc 0f865719`）；artifacts 8 项齐全；`val_engine_mean_rmse=62.5931 < val_pooled_rmse=63.6472`（Jensen）；params 115586；1 epoch。B2 端到端链路在 HEAD 上成立。

## 11.6 放行结论

- **A1 / A2 / B2 / B3 / B4：关闭**（HEAD f314268）。
- **A3p：契约内等价，放行**；附 11.2、11.3、11.4 三条 P2。
- **Gate A 发车条件**：只锁 HEAD `f314268` 的 14 个 blob。工作树 `config_hc bffe552f / train_hc 3fb05844 / run_hc d1c4d87e` 未提交、未上 a40、未 smoke，若要用必须先 commit 成 v4 → 重跑 smoke → 重定 §1。

---

# 12. 第一波（W1）审计 · 2026-11-12 01:20–02:15

（本节起 `##` 编号从 12 开始，避免与上方重复的 11.1–11.6 冲突。）

## 12.1 并发是否改变数值：**不改变（逐位相同）** —— 但并发**没有吞吐收益**

**问题**：`cudnn.benchmark=True` 下，同一份 init 在并发负载中可能选到不同算法，导致主表里混进「并发伪影」。G1 的阈值只有 1.0%，若并发扰动达到同量级，整张表都不可用。

**做法**：在 `A2-FD001-W1r3` 正在跑（2 路负载）时，用与 `A3p-FD001-W1` **完全同源**的配置起一格探针
`HC-C-A3p-FD001-s101-R3CONC2`（`--init-from init_v9_A3p_s101.pt`，v9 锁校验通过 `git_sha=d1e0527e8325`）。

| 项 | 串行（`A3p-FD001-W1`） | 2 路并发（`R3CONC2`） |
|---|---|---|
| `val_engine_mean_rmse` | `10.895096839289387` | **`10.895096839289387`** |
| `epochs_run` / `best_ep` | 36 / 24 | 36 / 24 |
| `elapsed_s` | 201 | **954.7** |

**结论一（科学侧）**：`got == ref` 逐位相等，`absdiff = 0.0`。**并发不改变数值**，主表在 2–3 路并发下跑不会引入并发伪影；`cudnn.benchmark` 的算法选择在同形状下稳定。

**结论二（工程侧，与直觉相反）**：并发**不是免费的**。同一格在 2 路负载下 `elapsed_s` 从 201 s 涨到 954.7 s（**4.75×**）。两格并发总耗时 954.7 s，而串行两格只需 ~402 s —— **2 路并发的吞吐是串行的 0.42 倍，越并越慢**。

所以「并发能把 8 小时压到 3 小时」这个假设是**错的**，不需要再由主控在串行/并发之间取舍：**串行是实测更优的选择**，调度器现状不用改。

限定：n=1 格，并发伙伴是 A2（不同变体、不同规模），4.75× 这个倍数不可外推到 3 路；但「并发不改变数值」这条对本表是充分的，且吞吐方向的结论只需要「并发没有明显收益」就能支撑。

## 12.2 A2 / A4 共享投影初始化：**已验证 60/60 逐位相同**

计划要求 A2/A4 用同一次 `Linear(1,64)` 抽样，否则两臂的差里混初始化噪声。两个 init 文件 md5 不同（`deeb0d9f` vs `0071b91b`）、字节数却同为 479266，**md5 不同不能直接判「没共享」**。

实测（`torch.equal` 逐张量）：`state_dict` 各 60 键，键集相同，**identical 60 / differing 0**；文件差异只来自 payload 元数据 `variant`（`'A2'` vs `'A4'`）与 `note`。**共享初始化成立。**

这与 §11 的 A3p 双 dump 是同一现象：**文件级 sha 不同 ≠ 张量不同**，溯源字段会让 sha 分叉。判定一律用逐张量比较，不用文件 sha。

## 12.3 `manifest.git_sha` 在标准路径下是字面量 `"WORKTREE"`，不是提交号 —— 命名陷阱

`run_hc.py:213`：

```python
git_sha_field = guard['git_sha'] if guard['mode'] == 'git' else 'WORKTREE'
```

锁模式（`mode == 'lock'`，即正常路径）下，`manifest.git_sha` 被**故意**写成字面量 `WORKTREE`，真实提交号写进 `code_lock_git_sha`。而 `run_hc.py:155` 其实已经算出了锁里的 `git_sha`，第 213 行又把它覆盖掉。

10 个 W1 格实测一致：

```
git_sha=WORKTREE   code_lock_git_sha=d1e0527e83   worktree_dirty=False   baseline_version=v9
```

**后果**：台账 `git_sha` 列对每个 v9 run 都显示 `WORKTREE`，看起来「无法归属到提交」。今晚两次误判（先判 `1505Z`/`1508Z` 是「脏树」，又用单个 blob 反推 v4）都踩在这个坑上：**证据本来是够的，是字段名把人引偏了**。

**建议（一行修）**：把 `run_hc.py:213` 改成 `git_sha_field = guard['git_sha']`（两种模式都返回真实提交号），`code_lock_git_sha` 保持不变用于锁溯源。这样 `manifest.git_sha` 永远是真提交号，`WORKTREE` 这个值可以彻底消失。**这属于可逆改动，但 `run_hc.py` 在我写入边界外，由 tmh 执行。**

## 12.4 A0 触顶 40 epoch 是否截断：**已收敛，不是截断**

`A0-FD001`（`epochs_run=40, best@37`）与 `A0-FD003`（`40, best@38`）打满 epoch 上限，按「margin < patience」规则会被标为「未收敛即截断」。**这个标记是假警报**——查 `history.json` 最后三 epoch：

| run | 最后三 epoch 的 `val_engine_mean_rmse` | best | 判定 |
|---|---|---|---|
| A0-FD001 | 30.1416 / 30.1350 / 30.1386 | 30.0986 | 曲线已平（±0.14%），**收敛** |
| A0-FD003 | 39.9834 / 39.9849 / 39.9896 | 39.9834 | 曲线已平（±0.02%），**收敛** |
| A0-FD002 | 26.4538 / 26.4723 / 26.4707 | 26.2874 | 正常早停（best@26, ep=38） |
| A0-FD004 | 37.2933 / 37.3926 / 37.2580 | 37.1218 | 正常早停（best@17, ep=29） |

**结论**：A0 四格都没有「还在下降就被砍断」的情况，G3 不存在「A0 被低估、差距被放大」的偏置。判据只看 `epochs_run == best_ep` 或 `margin < patience` 会误报，**必须同时看尾段斜率**。

顺带：`train_hc.py:313` 的 `DONE ... ep=%d` 打印的是 **`best_ep`**，不是 `epochs_run`。日志里 `ep=10` 曾让我以为只跑了 10 epoch（实际 22）。`metrics.json` 两个字段都在，数据无损，但日志措辞会误导——建议改成 `best_ep=%d ep_run=%d`。

## 12.5 早期 Gate A 读数（不作结论，仅登记方向）

已 `ok` 且过完整性核对的主表格（`code_lock_git_sha=d1e0527`、`worktree_dirty=False`、v9、`preds` 行数与子集验证引擎数一致）：

| 变体 | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| A0（纯 cycle，无传感器） | 30.0986 | 26.2874 | 39.9834 | 37.1218 |
| A1（无 cycle） | 11.2806 | 11.8031 | 12.0306 | 13.2087 |
| A2（cycle + 线性投影） | **10.5925** | **10.2074** | 在跑 | — |
| A3p（Fourier，pre-norm，参考臂） | 10.8951 | — | — | — |

两条已可算的判据：

- **G3（A0 相对 A1 至少差 10%）**：四子集 A0/A1 = 2.67× / 2.23× / 3.32× / 2.81×，即 123%–232% 劣化，**远超 10% 阈值，四子集同向**。前提成立：这个 benchmark 不是只靠年龄就能解。
- **G1（A1→A2 至少改善 1%）**：FD001 −6.10%、FD002 −13.52%，**两子集同向且远超阈值**。

**必须提前提示的科学风险**（不是结论）：`A2-FD001 = 10.5925` **优于** `A3p-FD001 = 10.8951`。A2 与 A3p 同时差在「编码形式」和「注入位置」两个变量上，不能直接归因；但它提示一种可能：**收益来自「引入了 cycle 信息」，而不是「Fourier 编码」本身**。若 `A3a`（Fourier，post-norm）落在 A2 附近而非显著更好，则 G2（Δ(A2→A3a) ≥ 1.0%）会失败，论文主张会退化成「cycle 有用」——这既不新颖、也不是本文的机制贡献。

`A3a` 四格是本波最关键、目前**一格都没跑**的部分。建议调度器在 A2 收完后**优先排 A3a 四格**，而不是按字典序排 A4/A5：A3a 的结果决定这张表还值不值得跑完，早 3 小时知道比晚 8 小时知道有价值。

## 12.6 写入边界声明（我自己的动作）

为做 §12.1 的探针，我在 a40 上产生了两个 run 目录与一条 registry 记录：

- `experiments/hc/HC-C-A3p-FD001-s101-R3CONC`（第一次启动因 `sklearn` 缺 `GLIBCXX_3.4.29` 崩溃，`LD_LIBRARY_PATH` 未设；已删除该目录后换 id）
- `experiments/hc/HC-C-A3p-FD001-s101-R3CONC2`（成功，`ok`，`10.895096839289387`）
- `docs/paper_b_hc/registry/launch/HC-C-A3p-FD001-s101-R3CONC.json`（崩溃那次留下的孤儿 launch 记录，run_id 已被占用，无法复用）

三条都应登记为 `gate=R`、`determinism_only=true`、`in_gate_a_table=false`；`R3CONC` 的 `close` 若存在应为 `failed`，原因是环境（`LD_LIBRARY_PATH`），不是方法。我没有改 `paper_b/hc/**` 的任何源码，也没有覆盖任何既有 run 目录。

**环境教训（给所有在 a40 手工起格的人）**：`run_hc` 必须带 `LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib`，否则 `sklearn` 的 `_argkmin` 加载失败；调度器脚本里已经处理，手工起格容易漏。

## 12.7 格级锁与调度器：读源码 + 实跑确认

`experiments/hc/tmp/cell_lock.py` 的去重键只拼 `HC-C-{v}-{s}-s101-W1` 加后缀 `['', 'r2'..'r6']`，**不匹配 `…-20261112T0030Z` 这类时间戳 run_id**。所以台账里「`0030Z` 是 A3a-FD001 首个 ok」这条**不会**把 `A3a-FD001-W1` 判成已完成——实跑 `--status` 返回 `A3a FD001 free`。这条我原本担心会漏掉主表最关键的格，实测没问题。

调度器 `run_wave1_sched.sh:43-44` 遍历 `A0 A1 A2 A3a A4 A5 A3p × FD001..FD004`，**A3p 参考臂三格在列**，不会漏。单实例、前台串行、起格前过 `used + 12000 ≤ 44000` 显存门。

## 12.8 A3a / A3p 共用同一 init 文件：**已实证安全，我的担心是多余的**

`run_wave1_sched.sh:34-35` 把 `[A3a]` 与 `[A3p]` 都指向 `init_v9_A3p_s101.pt`。两者架构不同（`DSPT_HC` post-norm vs `DSPT_HC_Frozen` pre-norm），我一开始怀疑 remap 可能静默错位。

读 `train_hc.py:190-219` 后确认**不会**：remap 是**目标驱动**的（按 `net.state_dict()` 里实际有的键决定改名方向），只处理 `pe_proj` ↔ `cycle_enc.proj` 与 `pe_periods` ↔ `cycle_enc.periods` 两组，且 `load_state_dict(strict=False)` 之后**只要 missing 或 unexpected 非空就 raise**——不存在「静默丢张量、指标照跑」的路径。

更强的证据是**已经在 v9 上跑通了**：

| run | init 文件 | 加载行 | params | 结果 |
|---|---|---|---|---|
| `A3a-0030Z`（post-norm） | `init_v9_A3p_s101.pt` | `loaded=61 missing=0 unexpected=0 (remapped)` | 115586 | `11.316705661052403` |
| `A3p-W1`（pre-norm） | **同一文件** | `loaded=60 missing=0 unexpected=0 (remapped)` | 115586 | `10.895096839289387` |

61 与 60 的差是 A3a 多一个 `cycle_enc.periods` buffer（`train_hc.py:210` 的 copy 而非 pop），不是参数差异——两者 `params` 同为 115586。**A3a/A3p 的共享初始化控制是成立的**：它消除了「初始权重」这一变量。

**但 G2′（A3a→A3p）不是单变量对比（主控指正，采纳）。** 两者同时差两件事：

| | A3a | A3p |
|---|---|---|
| 循环模块 | **可训练**（`DSPT_HC`） | **冻结**（`DSPT_HC_Frozen`） |
| 注入位置 | post-norm | pre-norm |

所以 `Δ(A3a→A3p) = +3.73%`（FD001，`11.3167 → 10.8951`）**只能报数，不能归因**——不能称「注入点效应」。要单独测注入点，需要第四个臂：**可训练 + pre-norm**。

## 12.9 我自己的一处误判：用中途 `best` 当结果

`A2-FD003` 跑到 ep020 时 `best=12.3068@16`，我据此对主控说「FD003 相对 A1 反号 +2.30%，G1 可能破」。**这个判断是错的**——`best_ep` 是「到目前为止最好」，不是「最终」。它继续跑到 ep040：

```
ep020  best=12.3068@16
ep030  best=11.0391@29
ep040  best=10.6517@35   ← 最终
```

最终 `A2-FD003 = 10.6517347274`，`A1-FD003 = 12.0305702011`，**Δ = −11.46%**，与 FD001/FD002 同向。

**规则（写死，后续审计照此执行）**：凡是 `close.json` 未落盘的格子，`best_*` 字段一律不作结论；判据只读 `close.json status=ok` 之后的 `metrics.json`。中途 `best` 只能用来判断「是否在收敛」。

## 12.10 `A2-FD003` 触顶截断：−11.46% 是保守估计

`epochs_run=40`、`best_ep=35`，margin 5 < patience 12，**早停未触发即打满上限**。所以 10.65 是**上界**，真实误差可能更低，`−11.46%` 是**保守估计**。

措辞校准（tmh 指出，采纳）：**不能写「到上限时仍在刷新 best」**——`best` 在 ep35 之后再未刷新，最后几个 epoch 是走平（ep36–40：10.8499 / 10.8619 / 10.7731）。准确说法是「**到上限时早停尚未触发（patience 仅消耗 5/12），故不能排除还会下降**」。结论方向不变。

这个偏置方向**对本文假设有利**，论文里必须写明，否则审稿人会问「为什么这格没收敛」。

**同类触顶共 4 格，不止这一格**：

| run | ep | best | margin | 性质 |
|---|---|---|---|---|
| A0-FD001 | 40 | 37 | 3 | 曲线已平，真收敛 |
| A0-FD003 | 40 | 38 | 2 | 曲线已平，真收敛 |
| **A2-FD001** | **40** | **34** | **6** | 早停未触发，**G1 的 FD001 锚点** |
| A2-FD003 | 40 | 35 | 5 | 早停未触发 |

**因此 FD001 的 −6.10% 与 FD003 的 −11.46% 同为保守值，论文中须同等标注。** A0 两格是参考线，不影响 treatment 判定。

对照：A1 四格全部干净早停（margin 恰好 12），`A2-FD002` 与 `A3p-FD001` 也是。

## 12.11 G1 三个子集已同向（截至 12 格 ok）

| 子集 | A1 | A2 | Δ(A1→A2) |
|---|---|---|---|
| FD001 | 11.2806 | 10.5925 | **−6.10%** |
| FD002 | 11.8031 | 10.2074 | **−13.52%** |
| FD003 | 12.0306 | **10.6517** | **−11.46%** |
| FD004 | 13.2087 | 在跑 | — |

三个子集全部远超 G1 的 1.0% 阈值且同向。**但仍须四子集分开报**：G1 的判据里含「无子集劣化超过 2%」这一条，均值会掩盖单子集异常——这一点在我误判 FD003 那次已经演示过一次代价。

## 12.12 五个 init 文件的骨干逐位相同：**已验证**，但 G2 有一个残留风险

主控提出「G1/G2 的每一对必须同 init 文件」。**方向对，判据写错了**——A1/A2/A5/A3p 的循环模块形状不同（A1 无、A2 `Linear(1,64)`、A5 `Linear(9,64)`、A3p `Linear(8,64)`），一个文件装不下四套形状，物理上做不到「同文件」。

**正确的判据是「共享骨干张量逐位相同」，实测全过**：

```
键数：A1=58  A2=60  A4=60  A5=61  A3p_v9=60
五者共有的键：58
逐张量 torch.equal：mismatches = 0
```

多出的键恰好是各自的循环模块：`cycle_enc.knots` 仅 A5；`cycle_enc.proj.*` 仅 A2/A4/A5；`pe_proj.*` 仅 A3p（载入 A3a 时 remap 成 `cycle_enc.proj`）。**58 个骨干键在全部五个文件里逐位相同、零不匹配**，所以 G1（A1→A2）、G2（A2→A3a）、G2′（A3a→A3p）三对的骨干是同一份，不混骨干初始化噪声。

**两两矩阵（10 对全做，排除各自特有的循环模块键）**：

```
file A                     file B                      keys  mism
wave1_init_A1_s101.pt      wave1_init_A2_s101.pt         57     0
wave1_init_A1_s101.pt      wave1_init_A4_s101.pt         57     0
wave1_init_A1_s101.pt      wave1_init_A5_s101.pt         57     0
wave1_init_A1_s101.pt      init_v9_A3p_s101.pt           57     0
wave1_init_A2_s101.pt      wave1_init_A4_s101.pt         57     0
wave1_init_A2_s101.pt      wave1_init_A5_s101.pt         57     0
wave1_init_A2_s101.pt      init_v9_A3p_s101.pt           57     0
wave1_init_A4_s101.pt      wave1_init_A5_s101.pt         57     0
wave1_init_A4_s101.pt      init_v9_A3p_s101.pt           57     0
wave1_init_A5_s101.pt      init_v9_A3p_s101.pt           57     0
```

57 个骨干键包含主控点名担心的全部键：`temporal_pe`、`sensor_in.*`、`conv3/5/7.*`、`ms_proj.*`、`time_norm.*`、`layers.*`。**一格没漏。**

另核：每个 W1 run 的 `config.json:init_from` 与预期一致（A1 的 11 个 run 全用 `wave1_init_A1`、A2 的 6 个全用 `wave1_init_A2`、A3p-FD001 用 `init_v9_A3p`、A0 四格无 init），**没有一格用错文件**。

**注意**：不存在 `wave1_init_A3a_s101.pt`。A3a 与 A3p 共用 `init_v9_A3p_s101.pt`（`run_wave1_sched.sh:34-35`），靠 `train_hc.py:203-212` 的 remap 完成 `pe_proj → cycle_enc.proj`。

### 残留风险：G2 的投影抽样噪声未测

- A2 的 `cycle_enc.proj.weight` 形状 `(64,1)`，来自 `wave1_init_A2_s101.pt`；
- A3a 的 `cycle_enc.proj.weight` 形状 `(64,8)`，来自 `init_v9_A3p_s101.pt` 的 `pe_proj`（remap）。

形状不同 → 无法共享 → **Δ(A2→A3a) 必然含一份独立投影抽样噪声，而目前没有任何数据点能测出它的量级**。

量级对照：G2 阈值 1.0%，按 A2 的 ~10.6 算约 **0.11 RMSE**；而今晚已实测「初始化能移动 0.77」（v6 A3a 无 init `10.5455` vs v9 A3a 共享 init `11.3167`）——那是骨干+投影同时更换的**上界**，不是投影单独的量级，但它说明可移动量与待检测量不必然是同一量级。

**建议的先导实验（1 格，~30 min）**：用另一次投影抽样跑 `A2-FD001`，记 `A2b`，用 `|A2b − A2|` 直接估计投影抽样噪声。

- ≪ 0.11 → G2 的 1% 阈值有意义；
- ≈ 0.1–0.3 → G2 单次 Δ 不可解释，须多 seed / 多抽样取均值，判据要改。

顺序上必须先有噪声底线再读效应量——这正是 σ=0 那件事给出的教训。

## 12.13 已退役调度器的设计缺陷：保护条件在失败路径上不可达

今晚四次 OOM 与 r2/r3 双份的根因，**不是「并发参数没调好」，是保护条件的条件写错了**。缺陷在**已退役**的 `experiments/hc/tmp/run_wave1.sh`（4996 B，3 slots），不在当前 v3（`run_wave1_sched.sh`，2852 B，串行 + 显存门）里——这一点必须写准，否则会被误读成「现役调度器有同样的洞」。

两个缺陷叠加：

**① brake 的触发条件在失败路径上不可达。** 该脚本第 15 行：

```bash
BRAKE_AT=900   # seconds; a single run slower than this drops us to 2 slots
```

意图是「单跑太慢就降到 2 slot 减压」。但 OOM 的失败 run 在 **0–24 秒**内返回，永远够不到 900 秒——**越失败越不降档**。这是典型的「保护条件只在正常路径可达、在最需要它的失败路径上不可达」。

**② 幂等判据只看文件、不看活进程。** `run_cell` 只查 `close.json`/`launch.json`，不查「这个格现在有没有活进程」。旧调度器的 bash 子 shell 被杀后，`run_hc` 被 reparent 到 pid 1 继续跑（实测 ppid=1），文件层面看起来「这格没跑完」→ 推 `r2` → `r2` 目录已存在 → 推 `r3` → 同格两份进程。

两者合起来：失败不降档 → 继续往同一张卡上加进程 → 更多 OOM → 更多重复。

**当前 v3 是结构性修复，不是参数调整**：

- **串行**，没有 slot 概念，`BRAKE_AT` 这类条件不存在了；
- `cell_lock.py` 判据是「同格任一 `run_id` 有活进程（`pgrep -af paper_b.hc.run_hc` 解析 `--run-id`）→ `busy`」，直接看进程表，不看文件；
- 起格前实查 `nvidia-smi memory.used`，`used + 12000 > 44000` 就 `WAIT`。

**一条残留的不一致**：`cell_lock.py` 的 `SUFFIXES = ['', 'r2'..'r6']` 最多试到 `r6`，而准入正则是 `^HC-C-{v}-{s}-s101-W1(r\d+)?$`（任意 N）。若某格真需要 `r7`，`cell_lock` 会返回 `exhausted` 而正则仍会接受——两套规则各写一遍迟早再打架（今晚已在 `gate` / `in_gate_a_table` 上犯过一次）。建议抽成共享常量，调度器与准入同读一份。

## 12.14 单点最小值口径的稳健性：G1 成立，但 FD002 的效应大幅缩水

### 早停核验（回应主控对 `A1-FD002-W1r2` 只有 20 epoch 的疑问）

**正常早停，不是被 kill**：`epochs_run=20`、`best_ep=8`、`8 + patience(12) = 20` 精确触发；`history.json` 长度 20、最小值在第 8 轮、与 `close.metrics.val_engine_mean_rmse` 逐位相同（`11.803124216970…`）。

**但「其余全是 `epochs = best_ep + 12`」不成立——12 个 ok 格里有 4 格打满 40 上限**：

| run | ep | best | margin |
|---|---|---|---|
| A0-FD001 | 40 | 37 | 3 |
| A0-FD003 | 40 | 38 | 2 |
| **A2-FD001** | **40** | **34** | **6** |
| A2-FD003 | 40 | 35 | 5 |

其中 `A2-FD001` 此前被漏报（我只标了 `A2-FD003`）。这两格的曲线在 40 上限处仍在波动，性质是「触顶截断」，与 `A0-FD001/FD003` 的「已收敛但打满上限」不同，note 不能合并写。

### 平台噪声与单点最小值的偏差

对每格取验证曲线**后 10 个 epoch** 的标准差与均值：

| 变体 | 平台噪声 std | 平台均值 − 最优 |
|---|---|---|
| A0 | 0.060 | 0.13 |
| A3p | 0.208 | 0.73 |
| A1 | 0.738 | 1.40 |
| A2 | 0.849 | 1.06 |

**A1/A2 的验证曲线抖动（0.74 / 0.85 RMSE）与 G1 要检测的效应量（约 0.7–1.6 RMSE）同量级。** 报告的指标是「40 个 epoch 里的单点最小值」，因此每个数都带一份「从噪声曲线里挑最小点」的选择偏差，且各格偏差不同。

最极端的是 `A2-FD002`：`best = 10.2074`，但后 10 epoch 均值 **12.5499**、std **1.756**——该最小值取自一条中心在 12.55 的噪声曲线。同格 `A1-FD002`：`best = 11.8031`、平台均值 13.0108。

### 两种口径下的 G1

| 子集 | A1 平台均值 | A2 平台均值 | 平台口径 Δ | 预注册口径 Δ |
|---|---|---|---|---|
| FD001 | 12.3938 | 10.8608 | **−12.4%** | −6.10% |
| FD002 | 13.0108 | 12.5499 | **−3.5%** | −13.52% |
| FD003 | 13.2936 | 11.2174 | **−15.6%** | −11.46% |

**G1 的符号与量级在平台口径下仍然成立**（三子集全负，均值约 −10.5%），所以结论不倒。但 **FD002 从 −13.52% 缩到 −3.5%，且三子集的排序完全改变**——单点口径下 FD002 最好看，平台口径下它最差。**原口径的子集排序不可信。**

### 处置建议

1. **主指标保持预注册的「best epoch」，不得事后更换**——换指标即 p-hacking。
2. **必须同时报平台口径**（后 10 epoch 均值，或 best±2 邻域均值），并说明 G1 在两种口径下是否同号、同量级。
3. 论文不得以「−13.5%」这类单点效应量作为卖点，应报双口径或区间。

判定器 `tmp/audit_hc/gate_a_judge.py` 增加平台均值列，**不改 G1 判定逻辑**，只多报一列。

## 12.15 撤回：4.75× 并发惩罚的归因站不住

### 我说过什么

我在 §12.1 写过：2 路并发下 `A3p-FD001` 的 `elapsed_s` 从 201 s 涨到 954.7 s（**4.75×**），吞吐只有串行的 0.42 倍，据此建议保持串行。主控据此把「3 路并发」改回串行。

### 为什么错了

燕子指出 `elapsed_s` 不在 `close.json` 而在 `metrics.json`（我引用时没说明出处）。顺着核同一配置的**全部六次**耗时：

| run | 并发度 | `elapsed_s` |
|---|---|---|
| `A3p-FD001-1508Z` | 串行 | 348 |
| `A3p-FD001-0005Z` | 串行 | 200 |
| `A3p-FD001-0030Z` | 串行 | 200.8 |
| **`A3p-FD001-R1-0035Z`** | **串行** | **811.0** |
| `A3p-FD001-W1` | 串行 | 201.3 |
| `A3p-FD001-R3CONC2` | **2 路并发** | 954.7 |

**`R1-0035Z` 是纯串行，也用了 811 秒**——与 `0030Z` 的 200.8 秒相差 4.0 倍，配置与结果均逐位相同。所以 a40 的墙钟时间波动主要来自**外部负载**（baselines worker、他人的 bpic 任务、常驻 math_modeling），我的探针 954.7 秒**落在串行的观测区间附近，不能归因于并发**。

### 现在能说的

- **成立**：并发不改变数值（`R3CONC2` 与五次串行逐位相同 = `10.895096839289387`）。
- **不成立**：并发带来 4.75× 惩罚。
- **正确表述**：**并发有没有吞吐收益，没有测出来。**

保持串行仍然可以是对的，但理由必须换成：① 未测到收益（证据不足，非证据反对）；② OOM 风险（今晚四次 OOM 均源于并发）；③ 混条件（已完成 12 格为串行）。

### 教训

我把**一次观测**（探针 vs 一个伙伴）当成了因果，没有先查同配置的历史分布——而 `R1-0035Z` 的 811 秒就在同一批数据里，查一次就能发现。这正是我当晚反复提醒别人「别拿单点下结论」的那件事。**任何「A 导致 B」的墙钟断言，先拉同配置的全部历史耗时，再谈归因。**

## 12.16 投影抽样噪声实测：|Δ| = 0.0301 RMSE（G2 只能报方向）

### 设计

G2 比较 A2（`cycle_enc.proj` 为 `Linear(1,64)`）与 A3a（同位置为 `Linear(8,64)`，由 `pe_proj` remap 而来）。**两者骨干完全共享（57 键逐位相同，见 §12.12），但循环投影形状不同、必然独立抽样**——这份噪声此前无人测过，而 G2 的阈值只有 1.0% ≈ 0.106 RMSE。

探针 `HC-C-A2-FD001-s101-NOISE1`：同一 `hc_config('A2','FD001',101)` + `build_hc` + 骨干从 `init_v9_A3p_s101.pt` 覆盖，**只把 RNG seed 换成 202**，让 `cycle_enc.proj` 重新抽一次。

init 验证：`wave1_init_A2_noise1_s101.pt`（sha256 `436d7449f60cb64f`）骨干 58 键与 `wave1_init_A2` **0 不匹配**；`proj.weight`/`proj.bias` 均不同（max|diff| 1.7734 / 1.8419）。

### 结果

| run | `val_engine_mean_rmse` |
|---|---|
| `A2`（原抽样） | `10.5924603632` |
| `A2b`（NOISE1，新抽样） | `10.5623371815` |
| **\|Δ\|** | **`0.030123` RMSE = 0.28% of A2** |

`NOISE1`：`status=ok`、`epochs_run=40`、`best_ep=31`、`elapsed_s=1004`。

### 判读（按主控预注册的三档表）

`0.03 ≤ 0.0301 ≤ 0.11` → **第二档**：

- G2 阈值 0.106 RMSE，而 `σ_Δ = 0.0378`（见 §12.18.1）→ **2.80σ**，阈值处单次检出概率约 50%；
- **但 σ 只有 1 个样本**，0.030 本身的不确定度极大（真值可能 0.015 也可能是 0.06）。

**结论：G2 的单次 Δ 只能报方向，不能报量级；要判 G2 必须多抽样取均值或配对比较。**

**方向性细节**：`A2b`（10.5623）**优于** `A2`（10.5925）0.03。若 A3a 也落在同量级，**单格无法区分「Fourier 略好」与「投影抽样运气好」**。

### 对 G2 的直接影响

1. 单格 `A3a-FD001` 只能报「落在哪一档」，不得写成「支持/不支持 G2」的定论；
2. G2 的判定（均值 ≥1.0% 且 ≥3/4 同号）需要四子集，且应在论文中说明投影抽样噪声量级；
3. 若最终 Δ 的均值与 0.106 同量级，需补多抽样（至少 3 次）才能给出区间。

## 12.17 触顶分类改用 OLS 斜率：真截断只有 2 格，不是 3 格

主控指出「用 std 判『是否波动』不够——低方差但单调下降的曲线 std 也小」，要求改用末 10 epoch 的 OLS 斜率。已算（只读 `history.json`）：

| run | ep | best | margin | **斜率（RMSE/epoch）** | std |
|---|---|---|---|---|---|
| A0-FD001 | 40 | 37 | 3 | **−0.0088** | 0.047 |
| A0-FD003 | 40 | 38 | 2 | **−0.0242** | 0.103 |
| A2-FD001 | 40 | 34 | 6 | **−0.0096** | 0.154 |
| A2-FD003 | 40 | 35 | 5 | **−0.1652** | 0.638 |
| A2-FD004 | 40 | 34 | 6 | **−0.1675** | 0.633 |
| A3a-FD001 | 37 | 25 | 12 | **+0.0160** | 0.266 |

**按斜率重划（|斜率| < 0.03 判收敛，> 0.15 判真截断）：**

- **真收敛**：A0-FD001（−0.009）、A0-FD003（−0.024）、**A2-FD001（−0.010）**
- **真截断**：**A2-FD003（−0.165）**、**A2-FD004（−0.168）**

**这修正了 §12.10 / §12.14 的说法**：先前写「A2 四格里三格触顶 → 三格都是上界」**不准确**。`A2-FD001` 的 margin 虽然是 6，但斜率 −0.0096 说明它已基本收敛（10 epoch 只降 0.096，占 10.59 的 0.9%）。**真正是上界的只有 FD003 与 FD004 两格**，10 epoch 各降约 1.65 RMSE（占 15%）。

因此 G1 的保守性偏置比我先前估计的**更小**，方向不变（仍偏保守）。

**同时否掉一个假设**：主控提出「延迟最优轮次的是可训练循环分支，不是 cycle 本身」（依据 A0/A2 的 `best_ep` 在 34–38，A1/A3p 在 24–25）。`A3a-FD001` 的 `best_ep = 25`、margin = 12（干净早停）**正好落在该假设预言会被打破的位置**——A3a 是「可训练 + Fourier」却和 A1/A3p 同档。加上 `A2-FD002` 的 `best_ep = 14` 比 A0 的 26 还早，该假设至少有 2 个反例。

**结论：`best_ep` 的早晚不能归因于「可训练 vs 冻结」或「有无 cycle」，应降级为观察项，不作机制断言。**

## 12.18 设计缺陷：Fourier 通道的输出尺度比线性通道小 158 倍，G2 的负号是尺度伪影

主控建议「两边各跑一次前向，比 `cycle_enc.proj(x)` 的输出 std」，以判断 `NOISE1` 的噪声能否迁移到 A3a。检查做完了，结论远超预期。

### 测量（FD001 训练集，n=16762）

cycle 标量：`mean=110.56  std=70.11  min=1  max=362`。

| 臂 | 模块 | 权重形状 | `w_std` | 输入 std | **输出 std** |
|---|---|---|---|---|---|
| A2 | `cycle_enc.proj` | (64,1) | 0.5356 | 70.107 | **69.665** |
| A3a | `cycle_enc.proj` | (64,8) | 0.1986 | 0.707 | **0.440** |
| A3p | `pe_proj` | (64,8) | 0.1986 | 0.707 | **0.440** |

**输出尺度差 158 倍。** 原因在编码器定义本身（`cycle_encoder.py`）：`ScalarCycle.forward` 喂的是原始 cycle（0–362），`FourierCycle.forward` 喂的是 `sin/cos`（有界 [−1,1]）。同一个 `nn.Linear` 初值下，输出尺度随输入尺度走。

### 训练没有把这个通道养起来

| 臂 | `w_std` init → trained | 输出 std init → trained |
|---|---|---|
| A2-FD001 | 0.5356 → 0.5437 | 69.665 → 70.748 |
| A3a-FD001 | 0.1986 → 0.1993 | 0.440 → 0.517 |
| A3p-FD001 | 0.1986 → 0.1706 | 0.440 → 0.391 |

三个臂的投影权重在训练中几乎未动——不是「训练会放大它」，而是该通道从初值到收敛一直是小尺度。

### 数值后果：A3a 退化成 A1

| 臂 | `val_engine_mean_rmse` (FD001) | 相对 A1 |
|---|---|---|
| A1（无 cycle） | 11.2805982338 | — |
| **A3a（Fourier, post-norm）** | **11.3167056611** | **+0.32%** |
| A3p（Fourier, pre-norm 冻结） | 10.8950968393 | −3.42% |
| A2（线性, post-norm） | 10.5924603632 | −6.10% |

**A3a 与 A1 差 0.32%——统计上与「没有 cycle」不可区分。**

### 结论

**G2 的负号（Δ(A2→A3a) = −6.84%）不能读成「Fourier 编码劣于线性编码」。** 它测的是「输出尺度不等（158×）+ 编码形式不同」两个变量的合成，**不能归因给编码形式**。

正确表述：**Fourier 通道在输出 std≈0.44 的尺度下基本惰性，A3a 退化为 A1。**

这同时解释了 `A3p` 为何比 `A3a` 好 3.87%：两者编码器完全相同、输出尺度同为 0.44，但 A3p 是 **pre-norm 注入**（归一化之前相加），小信号不被 LayerNorm 压掉；A3a 是 **post-norm**，0.44 的信号加进归一化后的表示，相对幅度更小。

### 处置建议

1. **G2 降级**：不得据此断言「Fourier 不如线性」；只能报「在未做尺度对齐的设计下，Fourier 通道未表现出增益」。
2. **要判编码形式，必须先对齐输出尺度**（给 Fourier 的 proj 加匹配因子，或对编码输出做归一化），那是**新臂**，不是重跑。
3. **`A3a-FD001 = 11.3167` 在台账里应标「cycle 通道惰性」**，不是「Fourier 更差」。
4. 标 **PENDING-L2**：是否补尺度对齐的 `A3a'` 臂，属主张取舍。
5. ~~**可事前预测**：A3a 剩余三格应贴近 A1 而非 A2。~~ **该预测已被证伪，见 §12.18.2。**

### 12.18.2 收窄：通道惰性是**子集相关**的，不是统一机制

上面 §12.18 曾写「Fourier 通道在这个初始化尺度下基本惰性」，并据此预测 A3a 剩余三格会贴近 A1。**该预测被 `A3a-FD002` 证伪：**

```
A3a-FD002   RUNNING @ ep015   val_eng=11.4324   best=10.6051@10
A1-FD002    11.80312421699082
A2-FD002    10.207390787535077
```

A3a-FD002 的 best 比 A1 **好 10.1%**，只比 A2 差 3.90%——**FD002 上 Fourier 通道显然被训练激活了**，与 FD001 的「退化成 A1（+0.32%）」形成对照。

**修正后的表述**：

- **FD001**：通道惰性（输出 std 0.44，A3a = 11.3167 ≈ A1 = 11.2806，差 0.32%），G2 的 −6.84% 是尺度混杂的产物——这一条成立；
- **FD002**：通道被训练激活，A3a 显著优于 A1——「惰性」**不能推广**；
- **所以结论是「通道是否被激活是子集相关的」**，不是统一机制断言。

可能的解释（**待验证，不是结论**）：同一份 init、同样 0.44 尺度，但 FD002 有 6 个工况、cycle 分布不同、训练动态不同，把这条小尺度通道养了起来；FD001 在 37 个 epoch 内没养起来。

**这条同时提醒**：§12.18 的尺度测量是在**初始化**时做的，它能说明「起点小」，不能说明「训练中不会被用起来」。**初始尺度小 ≠ 通道永久无效。**

## 12.19 `A2 vs A3a` 不是单变量比较：`cycle_source` 同时从 `raw` 变成 `clipped`

主控要求「在写进论文计划之前，必须由代码级确认 A4、A5 各自改的是哪一项，不要凭键名推」。按此执行——直接跑 `hc_config` 做字段级 diff：

```
A2 vs A4   只有 cycle_mode: 'raw' -> 'log'               ← 干净
A2 vs A5   只有 cycle_mode: 'raw' -> 'spline'            ← 干净
A2 vs A3a  cycle_mode:   'raw' -> 'fourier'
           cycle_source: 'raw' -> 'clipped'              ← 第二个变量
A2 vs A1   cycle_mode: 'raw' -> 'none'
           injection:  'post_norm' -> 'none'
```

**`A3a` 把 `cycle_source` 从 `raw` 换成了 `clipped`**（label：`sensor + Fourier cycle (clipped at 250)`）。因此：

- **`A2 vs A4`、`A2 vs A5` 是真·单变量**，可以归因；
- **`A2 vs A3a` 混着「编码形式（raw 标量 vs Fourier 特征）」与「cycle 是否在 250 处截断」两个变量**，`−6.84%` **不能归因给 Fourier 编码**。

`A3p` 的 `cycle_source` 同为 `clipped`，所以**所有 Fourier 臂都用截断 cycle，线性/样条臂用原始 cycle**——系统性差异，不是个别配置。

### 影响

1. **「`A2 vs A3a` 是唯一干净的编码形式比较」这个前提被推翻**——它今晚已被主控、tmh 与本报告各引用过一次。
2. 编码形式上真正干净的比较只剩 **`A2 vs A4`（log 变换）** 与 **`A2 vs A5`（样条基）**。
3. 要判 Fourier，需补一个 `cycle_source='raw'` 的 Fourier 臂，或补一个 `clipped` 的线性臂。
4. 标 **PENDING-L2**：是否补这一臂属主张取舍。

**教训**：这次是「按代码确认、不凭键名推」直接抓出来的。`A2/A3a` 的差异在 manifest 的键集合上**看不出来**（两者都有 `cycle_enc.proj`，只是形状不同），只有 diff `hc_config` 的字段才暴露。

### 12.18.1 附带更正：我那个「3.5σ」算错了（主控指正）

我写过「G2 阈值 0.106 约为噪声 0.030 的 3.5 倍」。**错在用 `E|Δ|` 当 σ。** 正确换算：

```
|Δ| = 0.03012 是一次差值的绝对值，不是 σ
半正态：E|Δ| = 0.798·σ_Δ；探针是两次独立投影抽样之差 → σ_Δ = √2·σ_p
E|Δ| = 0.798·√2·σ_p = 1.128·σ_p
σ_p = 0.03012 / 1.128 = 0.0267          （单次运行的投影噪声）
σ_Δ(G2) = √2 × 0.0267 = 0.0378
0.106 / 0.0378 = 2.80σ                   ← 不是 3.5σ
```

**并且四子集均值压不掉这一项**：A3a 四格共用同一份 init 文件，投影抽样是同一个，该分量在子集间**完全相关**，`σ/√4` 不成立。四子集平均只能压子集特有的训练噪声。所以有效 σ 是 0.0378，不是 0.019。

**但这套换算对 A3a 不适用**——σ_p = 0.0267 是 **A2 通道**的敏感度，而 §12.18 已证明 A3a 通道的输出尺度小 158 倍且惰性。**2.8σ 属于 A2，不属于 G2。**

**处置更正**：原计划「在 A2 上补抽样」或「在 A3a 上重抽投影」**都无效**——前者测的仍是 A2 的通道，后者测的是惰性通道的噪声（接近 0）。真正需要的是**尺度对齐的新臂**（给 Fourier proj 加匹配因子或对编码输出归一化，把输出 std 从 0.44 抬到 A2 量级后重跑）。标 PENDING-L2。
