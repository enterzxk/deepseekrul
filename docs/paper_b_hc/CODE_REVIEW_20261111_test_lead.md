# 代码逻辑审计报告 · paper_b/hc（测试负责人 · 第一轮）

- 状态：**已完成（针对下列固定哈希的版本）**
- 最后更新：2026-11-11 22:10 (+08)
- 审计对象：`paper_b/hc/` 全量 8 个文件 + `paper_b/training/` 三处改动（tmh 22:04–22:06 落地）
- 审计方式：只读代码审查 + 本地只读探针（`tmp/audit_hc/test_hc_smoke.py`）+ `git diff` 对照 HEAD + 数据契约复算
- 未修改任何 `paper_b/hc/**`、`experiments/hc/**` 文件；本文件是唯一写入

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
