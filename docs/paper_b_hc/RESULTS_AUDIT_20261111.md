# DSPT-HC 结果审计 · 2026-11-11

**2026-11-11 22:35 B5 更正：9 倍说法作废，改为 Jensen 关系 + 3.94 平方误差质量比**（见 §2c 风险 1，原文保留删除线）

> 审计人：实验结果管理员
> 范围：冻结基线的只读代码审计（`paper_b/configs/protocol.json`、`paper_b/training/data.py`、
> `paper_b/training/train.py`、`paper_b/models/dspt.py`），外加台账可用性自检与代码 md5 基准核验。
> **代码基线以本地仓库为准**（五份文件 md5 已与主控给定值逐一核对一致，见 §2.0）；
> **a40 产物路径以新盘 `<HC_ROOT>` = `/media/sangfor/vdb/user/zxk/deepseek-rul` 为准**，
> 旧目录 `/home/yndx/deepseek-rul` 仅为只读参考，不得再写入。
> 方式：只读，未修改任何被审文件；所有结论附 `文件:行号`，关键结论用本地数据独立复算。
> 当前 run 数：**26**（24 格 legacy 回填 + 2 个 SMOKE；`<HC_ROOT>/experiments/hc/` 仍为空，见 §8）。

## 结论总览

| 项 | 结论 |
|---|---|
| 台账可用性（runs.csv / registry 可写 / 索引分区） | ✅ 通过（已修 BOM 与远端镜像滞后） |
| **代码 md5 基准（本地 / 新盘 / 旧盘三处对照）** | ✅ 通过（本地与新盘 5/5 一致；旧盘相对 v2 已 5/5 全部过期，不再作为基准） |
| **旧目录是否被违规写入** | ✅ 通过（迁移后 0 个新文件） |
| a) 80/20 按发动机 ID 划分是否引擎不相交 | ✅ 通过（四子集交集实测均为 0） |
| b) val_cuts 是否只依赖训练数据 | ✅ 通过（不读官方 test RUL 分布） |
| c) 模型选择指标是否发动机级、有无窗口挑最优 | ⚠️ 有风险（**低**）：无窗口挑最优；v2 已改为发动机级选点；残留 select-on-val 偏差与 `val_engine_rmse` 命名陷阱（原"9 倍加权"结论已由 B5 更正作废） |
| d) Cycle 编码路径、有无未来信息 | ✅ 通过（无未来信息），但有饱和与年龄捷径两点须记录 |
| 额外发现 | 已修 3 项（F1/R1、F2/R4、F6/R2）；仍开放 F3、F5、F7、F8、F9、F10、F11 |

**总体判定：可以开始跑 Gate A。** 阻塞项 0；F1/F2/F6 已由 R1/R2/R4 修掉并复核，
P0 的 R8/R9/R10 在首个 run 前确认即可。

---

## 1. 台账可用性自检

| 检查项 | 结果 | 证据 |
|---|---|---|
| `runs.csv` 列数 | ✅ **32 列**，与规范 §6 完全一致（顺序一致） | `registry/runs.csv:1` 与 `RESULTS_REGISTRY.md:116-124` 逐列比对 |
| `registry/launch/` 可写 | ✅ 实测创建并删除探针文件成功 | 探针 `.write_probe_ignore` 写入后立即删除 |
| `registry/close/` 可写 | ✅ 同上 | 同上 |
| `RESULTS_INDEX.md` Gate 分区 | ✅ A / B / C / D 四区齐全，另有「异常与撤回」 | `RESULTS_INDEX.md:8,14,20,26,32` |
| 重复/冲突行 | ✅ 无（仅表头） | 文件共 1 行 |
| **`runs.csv` 编码** | ⚠️→✅ **原文件带 UTF-8 BOM**，首列名会被解析成 `\ufeffrun_id`，`csv.DictReader` 会读成脏列名。**已重写为无 BOM 的 UTF-8**，32 列不变 | 修复前首字节 `EF BB BF 72 75 6E`；修复后 `72 75 6E 5F 69 64`（`run_id`） |
| **远端镜像同步** | ⚠️→✅ 远端 `<HC_ROOT>/docs/paper_b_hc/` 原为滞后副本（`RESULTS_INDEX.md` 648 B 旧版、无审计文档）。**已同步**，本地与远端三份文件 md5 逐一相同 | 见 §1.1 |
| **旧目录违规写入** | ✅ 迁移（2026-11-11 22:05）之后 `/home/yndx/deepseek-rul` 下**新增文件数 = 0** | `find /home/yndx/deepseek-rul -newermt "2026-11-11 22:00" -type f` 返回 0 |

**已修的两项**：`runs.csv` 的 BOM（会导致台账首列无法被脚本读取）、远端镜像滞后。B/C/D 三区的表格表头也已补齐为与 A 区一致，避免后续追加时列错位。

### 1.1 本地 ↔ 远端同步约定（本管理员执行）

台账三份文件（`registry/runs.csv`、`RESULTS_INDEX.md`、`RESULTS_AUDIT_*.md`）**以本地仓库为准**，
每次修改后同步到 `<HC_ROOT>/docs/paper_b_hc/` 同名路径，同步后 md5 必须一致。

同步后核对值（2026-11-11 22:58，回填 24 格后）：

| 文件 | md5 |
|---|---|
| `registry/runs.csv` | `065011f565918736d10ddf964f8a6dec` |
| `RESULTS_INDEX.md` | `3395dbf73462af0dfb672e873d27003e` |

> 审计文档自身的 md5 不在此自列（每次修订都会变，自引用必然失效）。
> 核对方式：本地与远端 `diff` 为空即视为同步。

反向约定：`registry/launch/` 与 `registry/close/` 由 tmh 在 **a40 上**写，因此这两个目录
**以远端 `<HC_ROOT>` 为准**，我登记时读远端，不读本地空副本。

---

## 2. 四问审计

### 0) 前置核验：代码 md5 基准（v2，R1/R2/R4 修复后）

**基准 = 本地仓库，版本 v2（2026-11-11 22:0x 起）。** 三处实际 md5：

| 文件 | 基准 md5（v2） | 本地 | 新盘 `<HC_ROOT>` | 旧盘 `/home/yndx` | 变更 |
|---|---|---|---|---|---|
| `paper_b/models/dspt.py` | `fb7011023b284a5f571cba4d4abb6927` | ✅ 同 | ✅ 同 | ❌ `341c2e148804974673fe4ac487cd455e` | 未动 |
| `paper_b/training/train.py` | `07e7379dc2a8ae93c0f828efa8694b6c` | ✅ 同 | ✅ 同 | ❌ `cc158ca2bc3cbbb699dfbc960feb15db` | R2 双口径 + R4 docstring |
| `paper_b/training/data.py` | `c2ee079f6e60a8b0d0eb608676ff871a` | ✅ 同 | ✅ 同 | ❌ `94413991597eb8909c581251d05538e1`（v1 旧基准） | R1 `cuts` 必填 |
| `paper_b/configs/protocol.json` | `c55720ffe09deff3a9b59c218a180114` | ✅ 同 | ✅ 同 | ❌ `54e74dd0624d6d6cc841a846e777e2cd`（epochs 36 / patience 6） | 未动 |
| `paper_b/training/metrics.py` | `87e8245b49f4d23f61c6dd019958aa24` | ✅ 同 | ✅ 同 | ❌ `87c6637c1ae705cdfa5f3bad9883bddf`（v1 旧基准） | R2 新增两函数 |

**结论：本地与新盘 5/5 一致，即为冻结基线。旧盘相对 v2 已是 5/5 全部过期**
（此前"3 旧 + 2 新"的说法基于 v1 基准；R1/R2 之后 `data.py`、`metrics.py` 的旧盘副本也一并作废）。
结论不变：**必须逐文件比 md5，基准永远取本地仓库，旧盘不参与任何比对。**

**v1 → v2 的基准变更（旧值作废，勿再引用）**：

| 文件 | v1（作废） | v2 |
|---|---|---|
| `data.py` | `94413991597eb8909c581251d05538e1` | `c2ee079f6e60a8b0d0eb608676ff871a` |
| `metrics.py` | `87c6637c1ae705cdfa5f3bad9883bddf` | `87e8245b49f4d23f61c6dd019958aa24` |
| `train.py` | `4853414442df0a286dce4c1ae1fbf3fc` | `07e7379dc2a8ae93c0f828efa8694b6c` |

> **与任务书的一致性**：`TASK_BOOK.md:67-75` 的冻结 md5 表已于 2026-11-11 22:20 同步为同一组 v2 值，
> 与本表一致。若两者再次出现分歧，**以任务书当前版本为准**，并记入下一次审计。

### 0.1 R1 / R2 / R4 修复验证（行为不变证据，入档）

**R1（`cuts` 必填）—— 行为不变，已核**：

- 实现：`data.py:281` 签名改为 `cuts=None`；`data.py:290-292` 省略即
  `raise ValueError('cuts must be explicit; ...')`；docstring `data.py:285-288` 写明理由。
- 主控复验：改前/改后四子集 **32/32 字段逐元素相等**，`max|diff| = 0`，
  两个 npz 的 sha256 同为 `38032744387b71be50410aac2a4c23d52454d6f72cf504ad9bab5b153a66f11f`；
  省略 `cuts` 确实 raise。**采纳，作为"修复未改变数值"的证据。**
- R6 四个划分 hash 复验一致（`f81d7f2496c494ca` / `cacf4ea1d7cf96b6` /
  `f81d7f2496c494ca` / `eb1b0f00047d393e`，见 §2a）。

**R2（发动机级双口径）—— 已核，实现正确，但有一个字段命名陷阱（见 F9）**：

- `metrics.py:27-42` `per_engine_rmse()`：先按引擎算 RMSE 再等权平均，符合任务书
  "以发动机为统计单位"；`metrics.py:45-54` `per_engine_rmse_map()` 为逐引擎诊断。
- `train.py:103-112` `val_metrics()` 同时返回 pooled 与 engine_mean；**选点用 `engine_mean`**
  （`train.py:153-155`），`train.py:147-149` 两个值都进 history。
- `train.py:161-169`：`val_engine_mean_rmse` = 发动机级（主指标）；
  `val_engine_rmse` **保留为 pooled 的 legacy 别名**（`train.py:164-166` 有注释说明）。

**R4（docstring 修正）—— 已核**：`train.py:7-9` 现在写的是
"cuts 0.4/0.55/0.7/0.85/0.95, from paper_b/configs/protocol.json"，与代码一致。

### 0.2 对本审计的影响：行号已重定位，结论不变

代码变更后 `data.py`（314→323 行）与 `train.py`（172→187 行）行号发生位移。
**已逐条重新定位并复核**，本文档所有行号均为 **v2 基准**下的行号：

| 引用点 | v1 行号 | **v2 行号** |
|---|---|---|
| `split_units` | 89-96 | **89-96**（未变） |
| `make_val_windows` | 199-230 | **199-230**（未变） |
| 截断点与标签计算 | 215-223 | **215-223**（未变） |
| `fit_hi` 只喂训练引擎 | 289-290 | **298-299** |
| `prepare_test` | 304-314 | **313-323** |
| `cuts` 默认值（F1） | 281 | **281 + 290-292** |
| `val_rmse()` | 101-103 | **103-112** |
| best epoch 选点 | 143-149 | **153-157** |
| 结果落盘与指标 | 150-153 | **158-169** |
| `DEFAULTS` 的 cuts | 30 | **32** |

四问 a)–d) 的判定**全部不变**（通过 / 通过 / 有风险 / 通过），因为被审逻辑本身未被修改，
只有 `cuts` 的传入方式与指标聚合方式变了 —— 后者恰好**修掉了 c) 的部分风险**（见 §2c 更新）。

**新增审计纪律（并入 §5 清单第 11 条）**：核对 `code_md5` 时，基准永远取**本地仓库**对应文件；
发现 a40 上运行文件的 md5 与本地不一致 → `audit_verdict=fail`，不区分"新盘还是旧盘"。
旧盘任何文件都不再作为比对来源。

### a) 80/20 按发动机 ID 划分是否真的引擎不相交？

**结论：✅ 通过。验证/测试不会泄漏训练引擎。**

证据：

1. `data.py:89-96` `split_units()`：对引擎 ID 排序 → `rng = default_rng(split_seed + 20261020)` →
   置换 → `val = sh[fold*n_val : (fold+1)*n_val]` → `train = units \ val`。**train 与 val 由构造互斥**。
2. `data.py:261-263`：`tr = raw[raw.unit.isin(train_units)]`、`va = raw[raw.unit.isin(val_units)]`，
   两者来自同一 train 文件的不同引擎子集。
3. `data.py:264-271`：`OpScaler` 与 `pop`（工况基线）**只在 tr 上拟合**，val 只做 transform。
4. `data.py:298-299`：`fit_hi()` 传入 `s['train_units']`，HI 也只在训练引擎上拟合。
5. `data.py:313-323` `prepare_test()`：测试引擎来自独立的 `test_FD00x.txt`，仅复用训练折的
   `scaler` / `pop` 做 transform，**不重新拟合任何统计量**。
6. `data.py:283`：`prepare_train` 的 docstring 明确 "Never opens test files"，实现中确实无 `load_test` 调用。

独立复算（本地 `data/CMAPSS`，复刻 `split_seed=7, fold=0, val_frac=0.2`）：

| 子集 | 引擎总数 | train | val | 交集 | val_engine_ids_hash |
|---|---|---|---|---|---|
| FD001 | 100 | 80 | 20 | **0** | `f81d7f2496c494ca` |
| FD002 | 260 | 208 | 52 | **0** | `cacf4ea1d7cf96b6` |
| FD003 | 100 | 80 | 20 | **0** | `f81d7f2496c494ca` |
| FD004 | 249 | 199 | 50 | **0** | `eb1b0f00047d393e` |

> hash 定义（本审计约定，建议写入 launch schema）：`md5(",".join(str(u) for u in sorted(val_ids)))`
> 取前 16 位。以上四值可作为 launch 文件 `val_engine_ids_hash` 的**对照基准**，不一致即为划分漂移。

**须记录的一点（非缺陷）**：FD001 与 FD003 的 val 集合完全相同（`f81d7f...`），因为两者都是
100 台、ID 均为 1..100，同一 RNG 产生同一置换。这不是泄漏，但**四个子集的划分并不相互独立**，
写论文时不能把四子集当作四次独立随机重复。

### b) 验证截断 `val_cuts` 是否只用了训练数据、有没有用到官方测试 RUL 的经验分布？

**结论：✅ 通过。截断点完全由验证引擎自身的轨迹长度决定，与官方 test RUL 无关。**

证据：

1. `protocol.json:5-11`：`val_cuts = [0.4, 0.55, 0.7, 0.85, 0.95]` 是硬编码常量。
2. `data.py:215-223` `make_val_windows()`：`c = int(min(n, max(1, round(f * n))))`，`n` 是该**验证引擎
   在训练文件中**的轨迹长度；标签 `y = min(max(n - c, 0), rul_cap)`。
3. `data.py:205-206` docstring 明确："The cut rule is a protocol constant; it never reads official
   test RUL values."
4. `data.py:38-44` `load_test()`（唯一读取 `RUL_FD00x.txt` 的函数）只被 `prepare_test()` 调用，
   训练/验证路径不触及。
5. 全项目无任何 `percentile(RUL_*)` 之类以官方 test RUL 分布反推截断点的代码路径。

**风险提示（低，但必须写进论文）**：验证集是 **run-to-failure 完整轨迹**，官方 test 是**截断轨迹**，
两者任务分布不同。因此 val 数字与 test 数字**不可比、不可混表**——这与 Paper A 已知的
"val/test 差 ~1.2" 是同一现象。`RESULTS_REGISTRY.md:136` 已要求 close 的 metrics 在 FREEZE 前
不得出现 `test_*`，此处无冲突。

**登记/抽查口径补充（主控实测，已采纳）**：`va['cut']` 存的是**绝对周期数**，不是截断比例
（`data.py:226` 的 `cts.append(int(c))`，`c` 就是截断所在的循环号）。所以 FD001 的 100 个
val 窗里会出现 **74 个不同的 `cut` 值**（20 引擎 × 5 点，各引擎寿命不同）。正确断言是：

- 每台引擎恰好 **5** 个验证点（本次复算：四子集中"不足 5 点"的引擎数均为 **0**）；
- `len(val)` = 引擎数 × 5 → FD001 **100** / FD002 **260** / FD003 **100** / FD004 **250**；
- **禁止**用 `unique(cut) == 5` 判断，那会把正常情况误判为失败。

### c) 模型选择指标是不是发动机级聚合？有没有按窗口挑最优导致乐观偏差？

**结论：⚠️ 有风险（中低）。无窗口挑最优，但聚合口径与 select-on-val 偏差需整改。**

干净的部分（先说清楚）：

1. **没有按窗口挑最优**：验证窗口集合在训练开始前由 `make_val_windows()` 一次性固定
   （`data.py:199-230`），训练循环中只按 epoch 选点，不存在"从多个窗口中挑最好的那个"。
2. 每台验证引擎贡献**相同的 5 个截断点**（实测 FD001/FD002/FD003/FD004 的 val 引擎中，
   截断点不足 5 个的引擎数均为 **0**），因此不存在"某台引擎因点多而被过度加权"的问题。
3. 早停用的是 `val_metrics()`（`train.py:103-112`），**从未接触 test**；`--force` 之外
   无任何测试集参与选择的路径（见 F3）。v2 起选点用其中的 `engine_mean`（`train.py:153-155`）。

风险 1：**池化 RMSE ≠ 发动机级平均** —— **v2 已修复（R2），残留两点见下**

- v1 时 `train.py:101-103` 与 `150-151` 用 `metrics.py:4-6` 的**全局池化**平方误差均值，
  把所有 (引擎 × 截断点) 混在一起算，不是"先算每台引擎 RMSE 再平均"。
- ~~由于截断点由引擎寿命 `n` 决定，标签 `(1-f)·n` 随 `n` **线性放大**，长寿命引擎权重更高。
  实测：FD004 的 50 台验证引擎寿命 **139–435**，cut=0.7 处标签跨 **42–125**，
  同等相对误差下长引擎对池化 RMSE 的贡献可达短引擎约 **9 倍**。~~
  **【2026-11-11 22:35 B5 更正 —— 上述"9 倍"作废】** 冻结 val 下**每台引擎恰好 5 个窗口**
  （`make_val_windows` 的 `round(f*n)` + 去重；本审计复算四子集 `pts/eng` 全为 `[5]`），因此
  `pooled² = (1/(E·5))·Σ_e Σ_j e² = (1/E)·Σ_e MSE_e` —— **每台引擎权重相同，都是 1/E**，
  不存在"长引擎行数多所以权重大"。
  池化与发动机级平均的真实关系是 **先平方后平均 vs 先平均后平方（Jensen 不等式）**：
  `pooled = sqrt(mean_e MSE_e) ≥ engine_mean = mean_e sqrt(MSE_e)`，**池化恒 ≥ 发动机级平均**；
  两者不是加权关系。等权，但少数高误差引擎对池化值影响更大，因为**平方放大了尾部**。
  平方误差质量比（每台引擎 Σy² 的 max/min，冻结 cap=125）：本审计独立复算
  FD001 **3.26** / FD002 **3.17** / FD003 **2.54** / FD004 **3.93**
  （主控给 3.28 / 3.17 / 2.55 / 3.94，差 ≤0.02，属去重与取整细节，量级一致）。
  所谓 9.28 只在**不封顶**的 `(1-f)·n` 标签下才算得出，冻结 cap 下不成立。
- **结论仍然成立**：两种口径的排序**可能不同**（Jensen 差异在高误差引擎上被放大），
  所以论文与索引必须固定一个口径并注明，不得混排。
- **v2 起选点改用发动机级**：`train.py:103-112` 同时返回 pooled 与 engine_mean，
  `train.py:153-155` 用 `engine_mean` 选 best epoch，`train.py:161-169` 同时落盘两个口径。
  这一条风险**已消除**。
- **残留 1（F9）**：`metrics.json` 里的 `val_engine_rmse` 字段是 **pooled 的 legacy 别名**，
  名字却像"发动机级"。台账登记时若照字面取它，会把池化值当发动机级值。
- **残留 2**：历史 v1 数字（若有）是 pooled 口径，与新 run 的 engine_mean **不可直接比较**，
  必须标明口径，不得混排。

风险 2：**select-on-val 的乐观偏差**

- `train.py:153-157` 用验证集选 best epoch，`train.py:158-169` 又用同一验证集报告。
  40 轮、patience 12 下，报告的 val 数字是**"按 val 挑出来的最优轮"**，存在系统性乐观偏差。
  幅度有限，但**必须在论文中标注为 select-on-val，不能当作无偏估计**。此风险 R2 未涉及，仍然成立。

风险 3：**截断点上限饱和**

- 因 `rul_cap=125`，早截断点大量落在 cap 上：实测 val 点中标签恰为 125 的占比
  FD001 7.0% / FD002 8.5% / FD003 11.0% / **FD004 20.4%**。
- 这些点几乎不含区分信息（预测 ~125 即近乎无误差），会稀释聚合指标的判别力，
  尤其 FD004 有两成验证点属此类。

整改要求见 §4 的 R1–R3。

### d) Cycle 编码路径：cycle 是输入特征还是只进位置项？有没有用未来信息？

**结论：✅ 通过（无未来信息）。但饱和与年龄捷径两点必须记录在案。**

路径追踪：

1. `data.py:186`（训练）：`cycle = np.minimum(cyc / CYCLE_SCALE, 1.0)`，`CYCLE_SCALE=250`（`data.py:22`）。
2. `data.py:225`（验证）：`cycle = min(cyc[t] / 250, 1.0)`，`t = c - 1`，即**当前截断点**的 cycle。
3. `data.py:244`（测试）：`cycle = min(cyc[t] / 250, 1.0)`，`t = len(seq) - 1`，即**最后观测点**的 cycle。
4. `dspt.py:239-243`：`c_abs = cycle * 250` → 用 `pe_periods = (25, 50, 125, 250)`（`dspt.py:74`）
   做 Fourier 编码 → `pe_proj` 投影为 `d_model` → `.unsqueeze(1)` 后**逐时间步相同地**加到
   `pooled` 表示上。
5. `dspt.py:97-98`：`sensor_in` 的输入是 `n_feat = 3` 通道（raw / raw−baseline / causal EMA，
   见 `dspt.py:167`），**cycle 不在输入特征里**。

因此：**cycle 只作为全局年龄/位置偏置进入表示，不是输入特征；取的是当前已观测 cycle，无未来信息。**
`make_train_windows` 的滑窗也严格因果（`data.py:174-176`：左侧补零后 `sliding_window_view`，
窗口 `i` 结束于 `seq[i]`，不含 `t` 之后任何帧）；`_baseline_ref` 只用 `cycles <= 30` 且
`np.arange(len(seq)) <= t` 的帧（`data.py:148`）。

必须记录的三个实现事实：

1. **饱和**：`data.py:186` 的 `min(..., 1.0)` 使 cycle ≥ 250 全部映射到同一 Fourier 相位。
   FD004 存在寿命 435 周期的引擎，其 250 之后完全失去年龄分辨力。**这是 P2 的直接证据来源，
   也是基线本身的已知上限**——研究 Cycle 表达时，必须把"不截断 / 合理归一化不饱和"作为
   对照臂之一（任务书 §6 A3 内部对比已要求）。
2. **幅度被后续归一化部分抵消**：`dspt.py:243` 把 pe 加到 `pooled` 上，`dspt.py:244` 立刻
   `return self.norm(pooled)`（LayerNorm）。常数偏移会被 LayerNorm 部分消解，因此"cycle 影响
   有多大"不能只看 pe 的范数，必须靠**干预实验**测。
3. **年龄捷径**：训练集是 run-to-failure，`cycle` 与 `RUL` 单调对应，cycle 天然是 RUL 的强代理。
   这与任务书 P2 的判断一致。Gate A 必须用"固定传感器证据、只改 cycle"的干预来分离
   年龄先验与退化证据，不能靠相关性。

---

## 3. 额外发现

| 编号 | 级别 | 发现 | 证据 |
|---|---|---|---|
| ~~F1~~ | ~~高~~ **已修（R1）** | `prepare_train` 的 `cuts` 默认值曾与协议不一致（`(0.5,0.75,0.9)` vs `[0.4,0.55,0.7,0.85,0.95]`），省略即静默用错验证集。**现为 `cuts=None`，省略直接 raise**，行为不变已由主控 32/32 字段复验。 | `data.py:281, 290-292` |
| ~~F2~~ | ~~中~~ **已修（R4）** | `train.py` docstring 曾写 "cuts 0.5/0.75/0.9" 与实际不符；**现已改为引用 `protocol.json` 的 0.4/0.55/0.7/0.85/0.95**。 | `train.py:7-9` |
| F3 | 中 | `evaluate_test.py` 存在 `--force` 开关可绕过 FREEZE 清单。虽标注 diagnostic，但审计上无法从产物区分"是否用过 --force"。 | `evaluate_test.py:46,57-58` |
| F4 | 低（信息） | `paper_b/results/FREEZE.json` **不存在** → 当前任何 run 都无法合法跑 test，符合"test 未解封"状态。 | 文件不存在 |
| F5 | 低（信息，已更新） | `paper_b/hc/` **已创建**（`model_hc / train_hc / run_hc / data_hc / config_hc / cycle_encoder / snapshot_val / compare_val_snapshot / __init__` 共 9 个文件）；`experiments/hc/` 本地与 a40 新盘**仍为空** → run 数仍为 0。 | `paper_b/hc/` 目录列表 |
| ~~F6~~ | ~~低~~ **已修（R2）** | `metrics.py` 已新增 `per_engine_rmse()` / `per_engine_rmse_map()`，发动机级聚合有统一实现，不再是各脚本各写各的。 | `metrics.py:27-42, 45-54` |
| F7 | **中** | **旧盘相对 v2 基准已 5/5 全部过期**（v1 时期曾出现"3 旧 + 2 新"的混合状态，随 R1/R2 落地后 `data.py`、`metrics.py` 的旧盘副本也一并失效）。凭"旧盘=旧代码"或"旧盘=某份代码"的直觉去比对都会判错。 | §2.0 表 |
| F8 | 中 | 远端 `docs/paper_b_hc/` 副本**不会自动同步**，本次已滞后（`RESULTS_INDEX.md` 停在 648 B 旧版）。若无固定同步动作，tmh 会读到过期索引。 | §1.1 |
| **F9** | **中** | **字段命名陷阱**：`metrics.json` 的 `val_engine_rmse` 实为 **pooled** 的 legacy 别名（名字却像发动机级），`val_engine_mean_rmse` 才是发动机级。台账/报告若照字面取 `val_engine_rmse`，会把池化值当发动机级值。 | `train.py:164-169` |
| **F10** | **中** | `paper_b/experiments/probe_selection_metric.py:45-46` 调用 `prepare_train` **省略 `cuts`**，产出的 `paper_b/results/selection_metric_probe.csv` 属 **off-protocol**。R1 之后该脚本会直接 raise。任务书 §5.2 已标"不得作为证据引用"。 | `probe_selection_metric.py:45-46` |
| ~~F11~~ | ~~中~~ **已关闭（2026-11-11 23:25，牢莫核验）** | 曾记录"B5 错误表述已扩散进代码 docstring"。**当前工作树与 `0280cb8` 里两处都已是 Jensen 表述**（`metrics.py:30` "Three facts fix the relationship…"、`train.py:110` "sqrt(mean MSE_e) >= mean sqrt(MSE_e) by Jensen"），全文 `more rows` / `9x` 命中 **0**。本行引用的 `metrics.py:30-31`、`train.py:107-109` 是**证据引用，指向已修复版本之前的代码，不是开放项**。 | 旧：`metrics.py:30-31`、`train.py:107-109`（修复前） |

---

## 4. 对 tmh 的整改要求

> 我只记录，不改 tmh 的文件。以下由 tmh 在 `paper_b/` 侧执行；完成后在对应交付文档里注明，
> 我在下一次审计复核。

| 编号 | 要求 | 优先级 | 验收方式 |
|---|---|---|---|
| ~~R1~~ | ✅ **已完成**（v2 `c2ee079f…`）：`cuts` 改为必填，省略即 raise | ~~P0~~ | 已复核：`data.py:281, 290-292`；主控 32/32 字段、`max\|diff\|=0`、npz sha256 一致 |
| ~~R2~~ | ✅ **已完成**（v2 `87e8245b…` / `07e7379d…`）：新增发动机级聚合，选点改用 engine_mean，双口径同时落盘 | ~~P1~~ | 已复核：`metrics.py:27-54`、`train.py:103-112, 153-155, 161-169` |
| ~~R4~~ | ✅ **已完成**：docstring 已与 protocol 对齐 | ~~P2~~ | `train.py:7-9` |
| R3 | 论文与所有报告把 val 数字标注为 **select-on-val**（按验证集挑最优轮），不得表述为无偏估计 | P1 | 检查 `METHOD_SPECIFICATION.md` 与 Gate 报告措辞 |
| R5 | 正式 run 禁用 `--force`；若诊断必须用，在 close 文件 `notes` 里显式写明 | P2 | 我在抽查时核对 `notes` |
| R6 | 每个 launch 的 `val_engine_ids_hash` 使用 §2a 的 hash 定义，并与基准值比对；不一致视为划分漂移，立即 fail | P1 | launch 文件中的 hash 与 §2a 表格一致 |
| R7 | 报告四子集时注明 FD001 与 FD003 的划分相同（非独立重复） | P2 | Gate D 报告措辞 |
| **R8** | launch 的 `code_md5` 一律用**新盘 `<HC_ROOT>` 上的实际文件**计算，并与本地基准比对；`ckpt/log/preds/manifest` 四列写 `/media/sangfor/vdb/user/zxk/` 开头的绝对路径 | **P0（首个 run 前）** | 我逐文件比 md5，路径前缀正确 |
| **R9** | 任何新产物、新脚本、临时文件**不得写入** `/home/yndx/deepseek-rul`（旧盘只读参考） | **P0** | 我按 `find -newermt` 复查，出现即按违规记 |
| **R10** | 台账与报告一律以 **`val_engine_mean_rmse`** 作为 `val_rmse`；`val_engine_rmse` 只在需要 pooled 对照时使用，且必须标注"pooled"。建议 tmh 在后续版本把它改名或删掉别名，避免再被误读 | **P0（首个 run 前）** | 我核对 `runs.csv` 的 `val_rmse` 等于 `metrics.json` 的 `val_engine_mean_rmse` |
| **R11** | `results/selection_metric_probe.csv` 及其派生数字**不得出现在任何报告、索引或 run 的引用中**；若要保留，必须标注 off-protocol 且不与正式数字同表 | P1 | 我在抽查报告与 `RESULTS_INDEX.md` 时检索该文件名 |
| ~~R12~~ | ✅ **closed (false positive)** | 误报：当前 `0280cb8` 与工作树里 `metrics.py:30` 已是 "Three facts fix the relationship…"、`train.py:110` 已是 Jensen 表述，全文 `more rows`/`9x` 命中 0。**本行原引用的 `metrics.py:30-31`、`train.py:107-109` 是行号指向已修复版本的证据引用，不是开放项。** tmh 无需再改。 | ~~P1~~ | 已核 23:25 |

---

## 5. 登记节奏（本管理员决定，不另行请示）

| 动作 | 频率 | 说明 |
|---|---|---|
| 扫 `<HC_ROOT>/docs/paper_b_hc/registry/launch/` + `close/` | **tmh 每产出一个文件即触发**（主路径） | tmh 在 a40 上落盘后 @我，我立即登记并回执 |
| 兜底全量扫描 | **每 30 分钟**（实验运行期间） | 防止 tmh 忘记通知导致台账滞后；无新文件则不动 |
| 抽查 | 每次登记时抽 **3 条**（不足 3 条则全查） | 按 §8 清单 10 项，重点：产物存在、条数对得上、可复算、无泄漏 |
| 全量审计 | **每个 Gate 收口时** | Gate A/B/C/D 各一次，产出 `RESULTS_AUDIT_<日期>.md` |
| 日审计 | 每日 21:00 一次 | 汇总当日新增 run、异常、撤回；无异常则一行"当日无异常" |
| 冻结前审计 | `FREEZE.json` 出现时**强制全量** | 冻结是唯一一次解封前的最后关口，必须 100% 覆盖 |

**兜底扫描的执行方式说明**：定时扫描需要建计划任务，按纪律需用户显式授权，我不会自行创建。
在授权之前，主路径（tmh 通知即登记）+ 我每次被唤起时主动扫一遍，已能保证台账不滞后；
一旦发现滞后超过 2 小时，我会在审计文档里记一条 `registry_lag`。

**升级规则**（对齐 `RESULTS_REGISTRY.md:159-167`）：发现无 launch 文件的 run、数字对不上产物、
同 run_id 重跑、疑似泄漏/未来信息 → 立即在 `runs.csv` 标 `fail`，写审计文档，**并立刻上报主控**；
其中疑似泄漏/未来信息按规范触发**该 Gate 暂停推进**。

### 5.1 审计清单第 11–13 条（本轮新增）

11. **代码基准正确**：`code_md5` 与**本地仓库**对应文件一致（不是旧盘，也不是任何 a40 目录）；
    路径类字段（`ckpt_path / log_path / preds_path / manifest_path`）一律以
    `/media/sangfor/vdb/user/zxk/` 开头的绝对路径填写；发现产物或新文件出现在
    `/home/yndx/deepseek-rul` 下 → 按违规记。

12. **验证窗数断言正确**：`va['cut']` 是**绝对周期数**，不是截断比例。断言用
    "**每台引擎恰好 5 个点** 且 `len(val) = 引擎数 × 5`"（FD001 100 / FD002 260 /
    FD003 100 / FD004 250）；**禁止**用 `unique(cut) == 5` 判断（FD001 实测 100 个窗里有 74 个
    不同 cut 值，属正常）。

13. **证据来源合法**：任何被引用的数字必须能定位到 `runs.csv` 某行的产物。
    `selection_metric_probe.csv` 等 off-protocol 产物**不得作为证据**（F10）；
    `val_rmse` 取 `val_engine_mean_rmse`，不是 `val_engine_rmse`（F9 / R10）。

---

## 6. 附录：本审计的复算方法

复算脚本未写入项目（属临时核查），方法如下，任何人可重跑：

```python
# 引擎划分（对应 data.py:89-96）
rng = np.random.default_rng(7 + 20261020)
sh  = rng.permutation(sorted(units))
n_val = max(1, int(round(len(units) * 0.2)))
val   = sorted(sh[0:n_val])
train = [u for u in units if u not in set(val)]

# val_engine_ids_hash
hashlib.md5(",".join(map(str, val)).encode()).hexdigest()[:16]

# 截断点与标签（对应 data.py:215-223）
c = int(min(n, max(1, round(f * n))))
y = min(max(n - c, 0), 125)
```

---

## 7. 本轮状态

- 已登记 run：**26**（24 格 legacy 回填 + 2 个 SMOKE；见 §8）
- 新增审计：**§8（22:50）** 24 格 seed 202/303 因子消融回填审计 —— 6/10 清单项可验通过，
  查出 3 处口径/统计缺陷（池化 vs 发动机级、FD004 显著性依赖口径、跨 seed t-CI 自由度用错）
- 已审计文件：4 个（只读，未修改）+ 5 份代码 md5 三处对照（本地 / 新盘 / 旧盘）
- 已修台账问题：`runs.csv` 去掉 UTF-8 BOM（32 列不变）；远端镜像同步（三份文件 md5 一致）
- 已更正审计结论：**B5（22:35）** —— 池化口径"9 倍加权"说法作废，改为 Jensen 关系 +
  平方误差质量比 3.26/3.17/2.54/3.93；原文保留删除线，未删改
- 旧目录违规写入：**0**
- 阻塞项：**0**
- 待 tmh 处理：R1 / R8 / R9（P0）、R2 / R3 / R6（P1）、R4 / R5 / R7（P2）
- 下一步：等 tmh 在 `<HC_ROOT>` 上的首个 launch 文件；到达后按 §5 节奏登记 + 抽查

---

## 8. §B · 24 格 seed 202/303 因子消融回填审计（2026-11-11 22:50）

**对象**：`base` / `cand` / `full` × FD001–FD004 × seed 202/303 = **24 runs**，
产物在本地 `paper_b/results/raw/<tag>/<subset>/s<seed>/`。
已全部登记进 `registry/runs.csv`（`registry_note` 标 `LEGACY-BACKFILL`）。

### 8.1 通过项（清单 7/10 项可验）

| # | 检查 | 结果 | 证据 |
|---|---|---|---|
| 1 | 产物存在 | ⚠️ 部分 | `config.json`/`metrics.json`/`test_metrics.json`/`val_pred.csv`/`test_pred.csv`/`model.pt` 各 24/24；**无 `train.log`、无 `manifest.json`** |
| 2 | 条数对得上 | ✅ | val 100/260/100/250 = 引擎数×5（FD001 20、FD002 52、FD003 20、FD004 50）；test 100/259/100/248 = 官方 test 引擎数 |
| 3 | 能复算 | ✅ | 用 `val_pred.csv` / `test_pred.csv` 重算池化 RMSE/MAE/bias，与 `metrics.json` / `test_metrics.json` **差 < 1e-9（24/24）**；s101 另 12 格同法核过，36/36 全对 |
| 4 | 无泄漏 | ❓ 无法核验 | 产物不含 train 引擎列表，只能核"val 引擎集合跨 cell/seed 完全一致"（✅ 四子集各只有一个 hash） |
| 5 | 无未来信息 | ✅ | `config.cuts=[0.4,0.55,0.7,0.85,0.95]`，按验证引擎自身轨迹长度取点，不读官方 test RUL |
| 6 | 无测试集调参 | ✅ | `FREEZE_FACTORIAL_S23.json` 写于 **13:26:36Z**，早于 24 次训练结束（13:29–14:14Z）；`test_metrics.json` 全部在 **14:15:02–14:15:13Z** 落盘 |
| 7 | 预算一致 | ✅ | 24 行 `epochs=40 / patience=12 / batch=512 / span=70 / 五档 cut` 完全一致 |
| 8 | 代码没漂移 | ❓ 无法核验 | 无 launch 声明；`code_md5` 只能取提交 `122e0c8` 的 blob：`dspt.py=fb701102…`、`train.py=4853414…`、`data.py=9441399…`、`metrics.py=87c6637…` |
| 9 | 种子真实 | ✅ | `config.seed`、`metrics.json`、`preds.csv` 三处一致 |
| 10 | 重复检查 | ✅ | 同 (stage,variant,subset,seed,span) 无重复 |

### 8.2 缺陷 1 · `metrics.json.rmse` 是池化值，不是发动机级均值

- **证据**：24/24 行的 `rmse` 与 `preds.csv` 池化复算**逐位相同**（差 0.00e+00）；
  发动机级（`mean_e RMSE_e`）系统性更低 0.6–5.0。例：`base FD004 s202` 池化 15.9865，
  engine-mean 11.7032；`cand FD004 s202` 15.9865 → 12.1335。
- **原因**：这 24 次训练跑在 R2 **之前**的 `metrics.py`（`87c6637c…`），当时没有
  `per_engine_rmse`，只有 `rmse` 与一个同值的 `val_engine_rmse` 别名 —— 正是 F9/R10 记的那个命名陷阱。
- **影响**：`paper_b/reports/SEED_ROBUSTNESS.md` 标题写 "Validation RMSE (engine level)"，
  **表里的数其实是池化值**。但**结论不翻**：两口径下逐 seed 赢家序列都是 s101→cand、s202→full、s303→cand。
- **口径敏感点（测试集）**：池化排序 cand 11.2091 < full 11.6121 < base 11.7798；
  engine-mean 排序 cand 7.5193 < base 7.8230 < full 7.9971 —— **base 与 full 互换**。
  描述性指标换口径就会改排序，任何引用都必须写明口径。

### 8.3 缺陷 2 · FD004 的"显著"依赖口径

`cand − base` 在 FD004 上三个 seed 同号（−1.044 / −0.912 / −0.695，均值 −0.883），但：

| 口径 | 跨 seed t 95% CI（df=2） | 排除 0 |
|---|---|---|
| 池化 | [−1.322, −0.445] | **是** |
| 发动机级 | [−1.029, +0.073] | **否** |

其余三子集两口径下都符号不稳、CI 含 0。所以"cand 优势仅在 FD004 成立"这句话，
**池化口径下成立，发动机级口径下只剩"符号稳定 3/3"，不能说显著**。
`RESULTS_INDEX.md` 的判定 R-2026-11-11-01 已按此写。

### 8.4 缺陷 3 · 跨 seed t 区间的自由度用错

`SEED_ROBUSTNESS.md` 的跨 seed 区间用的是 `t = 3.182`（df=3）；`n_seeds = 3` 应为
**df = n−1 = 2，t = 4.303**，即报告里所有跨 seed CI **偏窄约 26%**。
复核：修正后 FD004 池化 CI 仍排除 0（[−1.322, −0.445]），**结论不翻**；
但报告里任何要引用的跨 seed 区间必须重算。

### 8.5 合规缺口（已逐行标注，未阻塞）

1. **无 launch/close 文件** → 严格按规范 §9 应记 `unregistered` 且不登记。本轮按主控指令回填，
   每行 `registry_note` 标 `LEGACY-BACKFILL` 并写明缺口，`audit_verdict` 记 `pass`（数字可复算，协议未走）。
2. **产物不在 a40**：`<HC_ROOT>/experiments/hc/` 下只有 `tmp`，无任何 run 目录；
   这 24 格的产物**只在本地仓库**。规范 §2 要求产物在 a40，本地机器是单点。
   建议 tmh 把这 24（+12 个 s101）目录 rsync 到 `<HC_ROOT>/experiments/hc/legacy/`，或主控确认本地仓库为权威副本。
3. **preds 旧 schema**：`unit,cut,y_true,y_pred`，缺规范要求的 `engine_id/cycle/split`。
4. **run_id 词表缺 base/cand/full**：规范 §3 的 variant 只有 A0–A5 / B0–B3 / C0–C3 / V0C/V0N/V0H。
   本轮回填用**产物 tag** 作 variant（`F_single_proj_mean_hi0` 等），`stage=B`、`gate=B`，
   请主控批准该映射或另定编号。
5. **host 字段**：产物里无记录，按 s23 会话上下文推断为 `a40`，已在 `registry_note` 说明。
6. **run_id 时间戳**取 `metrics.json` 的 mtime（UTC），`launched_utc = mtime − elapsed_s` 反推 —— 不是真实启动时刻。

### 8.6 台账并发写入（需要主控确认写权限归属）

- 22:26 我读 `runs.csv`：**仅表头**（299 B）。
- 22:48 我写入前再读：**已有 2 行**（`HC-C-SMOKE-FD001-s101-…T1431Z` failed、`…T1436Z` ok），
  `audited_by` 字段是乱码 `ʵ��������Ա`（GBK 字节被当 Latin-1 读），不是本管理员本轮所写。
- 我按**追加模式**写入 24 行，未覆盖任何他人行；写入后 26 行、32 列、无 BOM、LF、run_id 唯一。
- **待确认**：这 2 行是谁写的、`runs.csv` 是否只有管理员一个写入者。
  两个写入者并发追加会破坏"唯一台账"的假设。乱码字段建议由写入方修正（我不改他人行内容）。

---

## 9. 基线改为 git blob + 首批 3 条 SMOKE 登记（2026-11-11 23:05）

> 本节由实验结果管理员追加。§8 由测试负责人写入，两者独立，互不覆盖。

### 9.1 哈希基准：从工作树快照改为提交 blob

工作树在 10 分钟内变了三次（`ef14ee21`/`a799d43f` → `13ec2fbb`/`39202682`），
任何"工作树 md5"都必然过期。**基准改为提交 `f314268` 的 blob**：

```bash
git show f314268:paper_b/training/data.py    | md5sum
```

| 文件 | blob@f314268 |
|---|---|
| `paper_b/training/data.py` | `c2ee079f6e60a8b0d0eb608676ff871a` |
| `paper_b/training/metrics.py` | `39202682865d14dc6be5d332ac7a1c16` |
| `paper_b/training/train.py` | `13ec2fbb75ed09a389e4e979a82e11e6` |
| `paper_b/models/dspt.py` | `fb7011023b284a5f571cba4d4abb6927` |
| `paper_b/configs/protocol.json` | `c55720ffe09deff3a9b59c218a180114` |
| `paper_b/hc/config_hc.py` | **`0f86571997d7e2685c479a9bf835f6a3`** |
| `paper_b/hc/data_hc.py` | `3ad7c111cf0268409bb40d79f7271cee` |
| `paper_b/hc/model_hc.py` | `adcbceb901ae59d2bd15bbfc4067fc3f` |
| `paper_b/hc/cycle_encoder.py` | `f0ddc9115ba49627d4804fa6d5752e6d` |
| `paper_b/hc/train_hc.py` | `64b90518c42cff6e36074f3967d51caa` |
| `paper_b/hc/run_hc.py` | `959a490fce77b419bbb96cb822517742` |
| `paper_b/hc/verify_loop.py` | `bf3022ba60a804bdbd7b2ead0a38577c` |

上面 12 个值均由我在本地 `git cat-file blob f314268:<path>` 逐文件重算，非转录。

### 9.2 三条 SMOKE 的登记结果

| run_id | status | 审计 | 依据 |
|---|---|---|---|
| `HC-C-SMOKE-…1431Z` | `failed` | `suspect` | 无 `registry/launch`、无 `close`（§9 的 unregistered），仅 run 目录内有 launch 副本；首个优化器步前崩溃，0 epoch，无 ckpt/preds/metrics |
| `HC-C-SMOKE-…1436Z` | `smoke` | `suspect` | 提交前快照代码；8 个产物齐全；复算通过 |
| `HC-SMOKE-…1447Z` | `smoke` | **`fail`** | **`config_hc.py` 与提交不一致**，见 9.3 |

**1447Z 的复算复核（全部通过）**：`preds.csv` 100 行 = 20 引擎 × 5 截断点；
重算 pooled `63.6471717416`、engine-mean `62.5930740798`，与 `metrics.json` **差 0.000e+00**；
`pooled ≥ engine_mean` 成立（Jensen）；`params=115586`、`epochs_run=1`、`best_ep=1`。

### 9.3 关键发现：1447Z 跑在脏工作树上，不能作为放行依据

`HC-SMOKE-…1447Z` 是主控指定的放行依据候选，但它不满足"`code_md5` 与提交逐字一致"：

- 14 个文件里 **13 个与 `f314268` 完全一致**
- 唯一不一致：`paper_b/hc/config_hc.py`
  - run 声明（launch/manifest/config 三处一致）：`64bf6d5543f7fae6a8cd05cf254b2bcd`
  - 提交 blob：**`0f86571997d7e2685c479a9bf835f6a3`**
- 说明该 run 启动时工作树里 `config_hc.py` 是未提交改动 → 按主控规则应 **fail**

**加剧问题的一点**：两个 smoke 的 `manifest.json` 都**没有 `worktree_dirty` 字段**
（只有 `"git": "WORKTREE (no git on this host)"`）。也就是说主控设计的"脏树自动标记"机制
**尚未实现**，这次是靠逐文件比 blob 才发现的。建议 tmh 在 manifest 里补
`worktree_dirty` 与 `git_sha`，否则每次都要人工比对。

> 结论：**1447Z 不作为放行依据**；需在干净 `f314268` 工作树上重跑一次 SMOKE。

### 9.4 放行检查清单（五条核，已可执行）

1. `registry/launch/<run_id>.json` 与 `registry/close/<run_id>.json` **成对存在**
2. `code_md5` 逐文件等于 `git show f314268:<path> | md5sum`（12 个文件全比，不只比 5 个）
3. `val_engine_mean_rmse ≤ val_pooled_rmse`（Jensen 必要条件）
4. `preds.csv` 行数 = 验证窗口数（= 引擎数 × 5；FD001 100 / FD002 260 / FD003 100 / FD004 250）
5. `close.main_metric` 为 `val_engine_mean_rmse`，且与 `metrics.json` 同名字段差 < 1e-6

### 9.5 台账并发写入（呼应 §8.6，需主控裁定）

实测时间线：22:26 仅表头 → 22:48 已出现 2 条 SMOKE 行 → 我按追加模式写入 →
22:49:05 提交 `f314268` 把 24 条回填行与我的写入一并提交。

结论：`runs.csv` 目前**至少有两个写入方**。本管理员按"只追加、不删行、不覆盖他人行"执行，
但**唯一台账的假设已被打破**。请主控裁定：`runs.csv` 是否只允许本管理员写；
若允许他人回填，需规定格式与 `audited_by` 的署名规则（当前 24 行的 `audited_by` 署名本管理员，
但并非本人所写，且存在 GBK 乱码 `实锟斤拷员`）。

---

## 10. 更正（2026-11-11 23:30）：F11/R12 关闭 · `2ae9b5a` 作废 · A3a/A3p 判定改判

### 10.1 F11 / R12：closed (false positive)

F11 与 R12 **关闭**。当时引用的 `metrics.py:30-31`、`train.py:107-109` 是记录**旧代码**时的行号，
文件改过后行号已移位。当前 `0280cb8` 与工作树中：

- `metrics.py:30` = "Three facts fix the relationship to the pooled ``rmse``"
- `metrics.py:37` 附近 = Jensen 那句；`train.py:110` 同款
- 全文 `more rows` / `~9x` 命中 **0**

**结论：误报，tmh 不需要再改这两处 docstring。** 已在 §3 的 F11 行与 §4 的 R12 行标注 closed。

### 10.2 `2ae9b5a` 作废，v4 基准为 `0280cb8`（但 v4 本身仍非定稿）

- `2ae9b5a` 是**游离 commit**：`git merge-base --is-ancestor` 返回非祖先，`git branch --contains` 为空。
- 台账与审计中凡引用 `2ae9b5a` / `2482baa` / `8066f3f` 之处，**一律改为 `0280cb8`**。
- 主控基于 `2ae9b5a` 算的"v4 15-blob 表"**作废**，等 v5 定稿重发。
- 当前提交链：`f314268` → `f83e27e` → `a203f4b` → `0280cb8`（HEAD）。
- **`0280cb8` 也不是可复核定稿**：`--dump-init` 只在工作树（`run_hc.py:192-208`），HEAD 里没有；
  工作树仍脏（`model_hc` / `run_hc` / `train_hc` 改动 + `run_v0.py` 未跟踪）。等 v5。

**因此 §9.1 的哈希表只对"training 三件 + dspt + protocol"继续有效**（这 5 个在 v4 里未变）；
**hc 目录的 7 个值以 v5 定稿为准**，本文件暂不钉 v4 的 hc 值，避免又一次过期。

### 10.3 A3a-1505Z / A3p-1508Z：维持 `fail`（前两次判定均撤回）

**第一次判 `fail`（脏树）→ 改判 `suspect`（比对一个 blob）→ 现在撤回，维持 `fail`。**
两次撤回的原因都是同一个：用一个文件推整体。

**逐 blob 归属核查（本仓库 221 个提交全量搜索）**：

| 文件 | run 声明 | 出现在哪个提交 | 结论 |
|---|---|---|---|
| `paper_b/hc/config_hc.py` | `bffe552f…` | `0280cb8`、`4c7f238` | 有提交 |
| `paper_b/hc/model_hc.py` | `adcbceb9…` | **仅 `f314268`** | 有提交但属旧版 |
| `paper_b/hc/train_hc.py` | `3fb05844…` | **任何提交都没有** | **无法复现** |
| `paper_b/hc/run_hc.py` | `d1c4d87e…` | 任何提交都没有 | **无法复现** |
| `paper_b/hc/verify_loop.py` | `bf3022ba…` | `f314268` | 有提交但属旧版 |

**判定：`fail`。理由 = "code blob 不属于任何单一提交，无法从版本库复现"。**
它既不是 v4 也不是 v5，而是 `f314268` 与当时未提交工作树的**混合体**——
`model_hc` 取自旧提交、`train_hc`/`run_hc` 取自从未提交的改动。

**附带发现（基准表勘误）**：主控与测试负责人给出的"`0280cb8` 的 model_hc=`c55eb2b4…`、
train_hc=`58c1847b…`"与我的仓库不符。我实测：

| 提交 | model_hc | train_hc |
|---|---|---|
| `0280cb8`（v4） | `26e2d7c3ad963a0dcc2eeda437902f1d` | `c8f3cfe5cf12387019bbf280e4dd7975` |
| `4c7f238`（v5，当前 HEAD） | **`c55eb2b430e2d356a809a0ccbbb3f3f0`** | **`58c1847b6362cdd2a4432549c8ddc2db`** |

即 `c55eb2b4…` / `58c1847b…` 是 **v5 的值**，被误标成了 v4。这不影响本条判定（两条 run 的
blob 都不在 v5 里），但**基准表发下来之前必须先把这处对齐**，否则又会一次十分钟过期。

**第二重不可用（仍然成立）**：A3a 与 A3p 各自独立随机初始化，`pe_proj` 的 RNG 流错位
导致主干权重整体不同，差值混入初始化方差。

**处置**：不进 Gate 表、不作证据、不参与任何结论。唯一允许用途：给
`CYCLE_DEPENDENCE_STUDY.md` 留一句"初版实现曾给出 Δ=−0.350，但该对照含初始化方差，
重跑后以 v5 结果为准"。**不得产生"接近 v4、马上可用"的升级预期。**

### 10.3.1 越权写入记录（按任务书 §11 新增规则）

| 时间 | 被改文件 | 越权方 | 如何处理 |
|---|---|---|---|
| 2026-11-11 22:50 前后 | `RESULTS_AUDIT_20261111.md` | 测试负责人（追加 §8 全部内容） | 内容保留并标注归属（§8 开头已注"由测试负责人写入"）；本管理员版本为准 |
| 2026-11-11 22:49 提交 `f314268` | `registry/runs.csv` | 主控（批量回填 24 行 Gate B） | 行保留；`audited_by` 乱码已记；本管理员抽 5 条复算通过 |
| 2026-11-11 23:2x | `RESULTS_AUDIT_20261111.md` | 并发写导致一次 `ReplaceFileW EIO` | 已改为"先写自己的快照再覆盖"，不再静默合并 |

### 10.4 教训

本轮同一份审计文档在 5 分钟内被三个写入方（我 / 测试负责人 / 主控）改过，
出现一次 `ReplaceFileW EIO` 写冲突。**审计文档与 `runs.csv` 都需要单一写入方**，
否则"唯一真相源"不成立。此项已并入 §8.6 / §9.5 的待裁定事项。

---

## 11. 台账 schema 升级（2026-11-12 00:20）：两轴语义 + 口径列 + legacy 分区

### 11.1 两轴语义（主控裁定）

`status` = **进程结果**（`ok` / `failed` / `smoke`）；`audit_verdict` = **证据可用性**
（`pass` / `fail` / `suspect` / `n/a`）。**`ok + suspect` 是合法组合**——进程跑完但结果不可用。

合法组合只有五种：`ok+pass`、`ok+fail`、`ok+suspect`、`failed+n/a`、`smoke+n/a`。
**可用结果的唯一查询式：`status == 'ok' and audit_verdict == 'pass'`。**

据此修正 5 处非法组合（均改为 `n/a`）：

| run_id | 原 | 现 | 原因 |
|---|---|---|---|
| `HC-C-SMOKE-…1431Z` | failed / suspect | failed / **n/a** | 崩了，无可评估结果 |
| `HC-C-SMOKE-…1436Z` | smoke / suspect | smoke / **n/a** | smoke 本就不作证据 |
| `HC-SMOKE-…1447Z` | smoke / fail | smoke / **n/a** | 同上（fail 理由另记于 registry_note） |
| `HC-SMOKE-…1500Z` | smoke / **pass** | smoke / **n/a** | **smoke 标 pass 最危险，会被当证据** |
| `HC-C-A3p-…1505Z` | failed / suspect | failed / **n/a** | CUDA 崩、0 epoch、无 manifest |

### 11.2 新增两列（32 → 34 列）

| 列 | 取值 | 说明 |
|---|---|---|
| `val_metric_kind` | `engine_mean` / `pooled` / 空 | 同一列名 `val_rmse` 曾混装两种量 |
| `legacy_metric` | `true` / `false` | 是否属 R2 改口径之前的旧口径 |

回填结果：`pooled` 24 行（legacy）、`engine_mean` 5 行（hc 新 run）、空 2 行（无指标的失败 run）。
**任何按 `val_rmse` 的聚合必须按 `val_metric_kind` 分组，跨组求均值一律禁止。**

### 11.3 越权写入处理（主控裁定，追加记录）

| 时间 | 文件 | 越权方 | 处理 |
|---|---|---|---|
| 2026-11-11 22:49（提交 `f314268`） | `registry/runs.csv` | 未知执行者 | 数据保留（抽 5 条复算通过）；**`audited_by` 由 `实验结果管理员`（GBK 乱码 `实锟斤拷员`）改为 `(unattributed; 抽检 5/24 by 实验结果管理员)`** |

裁定原文要点：`runs.csv` / `RESULTS_INDEX.md` / `RESULTS_AUDIT_*` **只允许本管理员写**；
他人只发替换文本，由本人落笔；越权修改视为无效。

### 11.4 一致性检查器

`tools/check_runs.py`（本管理员工作区，只读、可复跑）：

```bash
python tools/check_runs.py            # 退出码 0 = 无 flag
```

检查项：BOM、表头 34 列与顺序、`run_id` 重复、status/verdict 合法组合、
`val_rmse` 与 `val_metric_kind` 一致、legacy 行必须 `pooled`、非 legacy 行路径必须在
`/media/sangfor/vdb/user/zxk/` 下、FREEZE 前不得有 `test_*`。
legacy 行豁免路径与 freeze 检查（产物在本地、历史 test 属既有事实）。
**每次登记完必须跑一遍，退出码非 0 不得收工。**

当前状态：**31 行 / 34 列 / 0 flag**。

### 11.5 v9 起的三件套 + 锁校验（缺一即 fail）

新 run 的 `manifest.json` 必须带 `worktree_dirty`、`git_sha`、`baseline_version`。
版本演进：`f314268`(v3) → `0280cb8`(v4) → `4c7f238`(v5，判不合格) → v6 链 `4eb03f9`/`6dc8348`/`20b9f3b` → v7 锁（目标 `20b9f3b`）→ HEAD `441c92f`。**v6 的 15-blob 表已无意义，等 v8**（tmh 在 `441c92f` 上生成 v8 锁、只提交 `code_lock.json`、`--check` 全绿）。

**新增硬校验（登记 Gate A 的 run 时必做）**：`code_md5` 必须与 **该 run 声明的 `baseline_version` 对应的锁**逐字一致，且该锁的 `git_sha` 必须是当时 HEAD 的**祖先**。不一致即 `fail`，不需要人工判断。

**示例（`HC-C-SMOKE-…1548Z`，2026-11-11 15:58Z）**：`baseline_version=v7`，锁目标 `20b9f3b`；实测声明值 **15/15** 命中该锁的 blob，且 `20b9f3b` 是 HEAD `441c92f` 的祖先；与 HEAD 相比有 4 个不同（`train_hc`/`run_hc`/`verify_loop`/`code_lock`）。故它是 **v7 的流水线证明**，不是 Gate A 的代码基准。

另：`code_lock.json` 是 `python -m paper_b.hc.code_lock --write` 的产物，由 tmh 写，
**本管理员不手工编辑**。

**路径更正**：`code_lock.json` 在 `docs/paper_b_hc/registry/code_lock.json`，不在仓库根。此前记的"a40 上不存在"是我查错了路径，以该路径为准。

**版本号勘误（2026-11-12 00:10）**：版本号以 **v9** 为准——代码 `d1e0527`、锁提交 `6c405ad`；
此前 v6 / v7 / v8 的表述**作废**，不追溯改历史。

**口径约定**：版本号只在对账时用，**权威标识永远是 commit SHA**；版本号只在定稿时贴一次，
中间提交不贴。此后版本信息一律写成「vN = 代码 SHA / 锁 SHA」。

---

## 12. 版本链快照（2026-11-12 00:35）：v9 已出现，v8 也已过期

**这节只记录实测 SHA 与 blob，不做版本号推断。权威标识永远是 commit SHA。**

| 版本 | 代码 commit | 锁 commit | 状态 |
|---|---|---|---|
| v3 | `f314268` | — | 早期基线（`HC-SMOKE-…1500Z` 跑在这棵树上） |
| v4 | `0280cb8` | — | 被取代 |
| v5 | `4c7f238` | — | 不合格（dump payload 缺字段、`--dump-init` 写完不退出） |
| v7 | `20b9f3b` | — | 被取代（`HC-C-SMOKE-…1548Z` 跑在这把锁上） |
| v8 | `aef910c` | `441c92f` | 被 v9 取代 |
| **v9** | **`d1e0527`** | **`6c405ad`**（当前 HEAD） | **待主控确认锁有效性** |

**v8（`441c92f`）实测 blob**（我本地 `git cat-file` 重算，非转录）：

| 文件 | blob@441c92f |
|---|---|
| `paper_b/hc/config_hc.py` | `bffe552fefa49c0e9cff38a31c6cd8ff` |
| `paper_b/hc/model_hc.py` | `c55eb2b430e2d356a809a0ccbbb3f3f0` |
| `paper_b/hc/train_hc.py` | `bc24115673b8d57bbc47694b5991321b` |
| `paper_b/hc/run_hc.py` | `8784311bebdc6891b59058f278e8d1f8` |
| `paper_b/hc/verify_loop.py` | `2c15fcb4293e226102e368c717ef3f3a` |
| `paper_b/hc/code_lock.py` | `cd492a1cdc6a812120eceea6f69f7a41` |

> 测试负责人报的 `train_hc=069f8977` 与实测不符（v8 实际是 `bc241156`，与主控给的 a40 侧值一致）。

**`HC-SMOKE-…1500Z` 的定性**：它跑在 **`f314268`（v3）** 的干净树上，14/14 blob 命中，
所以"干净"没错；但**基线已过期**，它不是当前锁的放行 smoke，不能给 Gate A 开门。
它的 `audit_verdict` 是 `n/a`（smoke 不作证据），这一点在 §11.1 已改，无需再改。

**A3a/A3p 的判定理由**（沿用，不挂 `config_hc`）：`model_hc=adcbceb9` 仅存在于 `f314268`；
`train_hc=3fb05844`、`run_hc=d1c4d87e` 在 221 个提交里都找不到 → **不属于任何单一提交，无法从
版本库复现**。`config_hc=bffe552f` 确实命中 v8/v9，但那一个文件对上不改变整体结论。

### 12.1 v9 复核（2026-11-12 00:40）

测试负责人复核 `6c405ad`：锁里 `baseline_version=v9`、`git_sha=d1e0527`、
15 个 blob 与 HEAD 逐个一致、`code_lock --check` 返回 **CODE LOCK OK（退出码 0）**。
我在本地抽查确认（`git cat-file blob 6c405ad:<path>`）：

| 文件 | blob@6c405ad (v9) | 与 v8(`441c92f`) |
|---|---|---|
| `paper_b/hc/config_hc.py` | `bffe552fefa49c0e9cff38a31c6cd8ff` | 同 |
| `paper_b/hc/model_hc.py` | `c55eb2b430e2d356a809a0ccbbb3f3f0` | 同 |
| `paper_b/hc/train_hc.py` | `069f8977bee8cf6aa01953c817d01601` | **不同**（v8 是 `bc241156`） |
| `paper_b/hc/run_hc.py` | `8784311bebdc6891b59058f278e8d1f8` | 同 |

> **注意**：测试负责人先前报的 `train_hc=069f8977` 其实是 **v9** 的值，被误标成 v8。
> 这已是同一类误差的第二次——用 `git show HEAD:` 取值时 HEAD 前进了。
> **取 blob 一律用显式 SHA**（`git cat-file blob <sha>:<path>`），不要用 `HEAD:`。

**状态**：v9 已由主控签发（2026-11-12 00:05）。签发前 `HC-SMOKE-…1500Z`（v3 树）与 `HC-C-SMOKE-…1548Z`（v7 锁）
都不能作为 Gate A 的放行 smoke；需要一次 **v9 树上的干净 smoke**。

### 12.2 v9 的 15 个 blob 实测表（`6c405ad`，2026-11-12 00:45）

**这是我本地 `git cat-file blob 6c405ad:<path>` 逐个重算的结果，不是转录。**
测试负责人复核：`baseline_version=v9`、`git_sha=d1e0527`、15/15 与 HEAD 一致、
`code_lock --check` 返回 CODE LOCK OK（退出码 0）、工作树干净。
**状态：主控已签发（2026-11-12 00:05），此表即 Gate A 的代码基准。**

| 文件 | blob@6c405ad |
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

> 用法：登记 Gate A 的 run 时，取 run 的 `code_md5`，与本表逐行比对；
> 全中 → 代码合格；任一不符 → `fail`。**不要用 `HEAD:` 取值，用显式 SHA。**

### 12.3 版本号复用规则（2026-11-12 01:05）

**一个版本号只能对应一种代码状态；复用即该版本下所有 run 无效。**

实测：v7 出现过两把锁——`70c8779` 的锁 `git_sha=20b9f3b`；`dc77575` 的锁是 from-index、
`git_sha=None`、`code_md5` 不同。v6 同理。因此 **v6、v7 全部标 `retired_version=true`、
`audit_verdict=n/a`（含原先判 pass 的）**，索引里两组标「版本号复用，作废」。

新增列：`retired_version ∈ {true,false}`（台账 36 列）。

### 12.4 A3a 崩溃根因（已确认，非环境问题）

- `HC-C-A3a-…1630Z`(v8)、`…0005Z`(v8)：`RuntimeError: init_from mismatch: missing=['pe_periods']`
- `HC-C-A3a-…0012Z`(v6)：`RuntimeError: init_from mismatch: missing=['cycle_enc.periods']`

A3a 的 `FourierCycle` 带 `periods` buffer，A3p 的 state_dict 里没有对应键，严格模式拒绝载入。
**这正是 v9 `d1e0527` 修掉的 bug**（"copy pe_periods, do not pop"）。所以这三条在旧版本上
必然失败，与 A3p 能跑满 36 epoch 不矛盾。**A3a 在 v9 上重跑是当前唯一待办。**


### 12.5 列变更记录（不悄悄加）

| 时点 | 变更 | 列数 |
|---|---|---|
| 2026-11-12 00:20 | 加 al_metric_kind、legacy_metric（主控授权） | 32 → 34 |
| 2026-11-12 00:45 | 加 aseline_version（主控采纳测试负责人建议） | 34 → 35 |
| 2026-11-12 01:05 | 加 
| 2026-11-12 02:40 | 加 `determinism_only`（重复实验标记） | 36 → 37 |
| 2026-11-12 02:55 | 加 `in_gate_a_table`（**Gate A 表准入**，按预登记 24 格 `-W1` 清单，默认 false） | 37 → 38 |
etired_version（版本号复用规则） | 35 → 36 |

aseline_version 回填口径：有字段直接取；无字段按 code_lock_git_sha 反推
（3f38cfa→pre-v5、20b9f3b→v6、dc77575→v7、ef910c→v8、d1e0527→v9）；
两者都没有填 unknown。当前分布：v9 1 / v8 3 / v7 1 / v6 2 / pre-v5 25 / unknown 7。

RESULTS_INDEX.md 已按 aseline_version 分组：当前基线 v9 一组，历史版本分组，v6/v7 标「版本号复用，作废」。

### 12.6 越权写入（第二次）+ 非规范枚举值（2026-11-12 01:35）

我在登记 A3a `…0030Z` 时发现 `runs.csv` 已被他人追加 7 行，并使用了**不在规范 §6 枚举内**的取值：

| 非规范值 | 出现次数 | 规范应为何 |
|---|---|---|
| `status=unclosed` | 6 | `running`（§6 有 `running`） |
| `audit_verdict=duplicate` | 1 | `suspect`（§6 只有 pass/fail/suspect/空） |

**处理**（按 §11 越权规则，以本管理员版本为准并留痕）：
- 6 条 `unclosed` → `running`，verdict 置空，note 改写；
- 1 条 `duplicate` → `suspect`，note 保留重复说明；
- 另将 2 条 `baseline_version=v5` 的行补标 `retired_version=true`（v5 已被主控判不合格）。

**这是第二次越权写入**（第一次是 22:49 提交 `f314268` 时的 24 行回填）。
两次都说明：规则已定，但执行层面没有强制。**建议主控定一个可执行的处理办法**（例如每次提交前
由我跑一遍 `tools/check_runs.py`，非 0 即拒绝提交）。

### 12.7 共享初始化证据（与主控判定依据有出入，请复裁）

主控判定 A3a `…0030Z` 用 `suspect`，依据是「两条 launch 均为 `init_from=None`、close 无 `init_sha256`」。
我实测：

| 位置 | 字段 | 值 |
|---|---|---|
| `launch.json` | `init_from` | 缺失 |
| `close.json` | `init_sha256` | 缺失 |
| **`manifest.json`** | `init_sha256` | `336074a6e173d8df70bba5f5053508990318260c793a4c4515ad88033456c3c8` |
| **`config.json`** | `init_from` / `init_from_sha256` | `experiments/hc/tmp/init_v9_A3p_s101.pt` / 同值 |
| 磁盘文件 | 实测 sha256 | 同值（482146 B） |

**两处记录与磁盘文件三方一致 → 共享初始化可验证。** 我仍按主控指令保留 `suspect`，
但建议复裁为 `pass`；若确认，Δ=+0.4216 可作 Gate A 对照结论。

### 12.8 复裁结果 + 强制门（2026-11-12 01:50）

- `A3a-…0030Z` → `ok` / **`pass`**（共享初始化三方一致，主控已复裁）；`A3p-…0030Z` 保持 `ok`/`pass`，
  note 改为「配对成立：同代码 v9、同 seed、同 init `336074a6…`、唯一差异为注入点」。
- `A3a-…1640Z` → `status=superseded` / `audit_verdict=n/a`（重复记录，不用 `suspect`）。
- 枚举规范化：`unclosed` → `running`（进程仍在）或 `failed`（进程没了且无 close）；
  `duplicate` → `superseded`（status 轴），不进 `audit_verdict`。已由主控写入 `RESULTS_REGISTRY.md`。
- **强制门**：自 2026-11-12 起，`runs.csv` 提交前必须通过 `tools/check_runs.py`，
  **退出码非 0 拒绝提交**。

### 12.9 第三次越权写入（2026-11-12 01:4x）

我在登记 `A3a-…0030Z` 后，其 `registry_note` 被他人改写（我的 `worktree_dirty=false, 15 code_md5…`
被替换为 `init_shared=yes, init_sha256=n/a…`），`status`/`verdict` 未变，数据无丢失。

这是今晚第三次（22:49 回填 24 行、01:1x 追加 7 行含非规范枚举、01:4x 改写单行 note）。
**已上报主控：连续三次说明有进程或另一个 Bot 在写这个文件，仅靠"以我为准覆盖"治不了根。**

### 12.10 新增 `in_gate_a_table`（37 → 38 列，2026-11-12 02:40）

**问题**：`gate` 记的是**实验类型**（A=treatment，R=重复/确定性），而"能否进 Gate A 表"是**准入**，
是两个轴。此前硬塞进 `gate` 导致同一对 run 来回改。

**做法**：新增 `in_gate_a_table`（布尔，默认 `false`）。**只有** `GATE_A_WAVE1_PLAN.md` §3 的
24 个 `HC-C-{variant}-{subset}-s101-W1` 填 `true`。**判定只看这一列**，与 `gate`/`status` 无关。

具体：`A3a/A3p…0030Z` 保持 `gate=A`、`in_gate_a_table=false`（不在 24 格清单）；
`A3a…1640Z` → `gate=R`/`superseded`/`retired=false`/`in_gate_a_table=false`；
`R1`/`R2` → `gate=R`/`in_gate_a_table=false`。

**并发控制**：写入前比对 sha256，变了即 abort；每轮登记后 `git commit` 把 registry 纳入版本控制
（此前 `launch/`、`close/` 全是 untracked，四次越权写入没有一次进过 git）。

**冻结**：本轮改完、`check_runs.py` 0 flag、提交后停手。

### 12.11 Gate 表准入按「格 + -W1rN」（2026-11-12 02:5x 生效，主控裁定）

**依据**：W1 的 OOM 重试落在 -W1r2 / -W1r3 后缀上，按「精确 un_id」准入会把真实成功的格错误排除。
当时的唯一 ok 证据正是两条重试：HC-C-A1-FD001-s101-W1r2（11.2806）、HC-C-A1-FD003-s101-W1r2（12.0306）。

**规则（已同步写入 RESULTS_REGISTRY.md §3.1，4a6f703）**

1. 格 = (baseline_version, variant, subset, seed)；un_id 必须匹配前缀
   HC-C-{variant}-{subset}-s101-W1，**允许 -W1r2 / -W1r3 重试后缀**（上限 r3）。
2. 同格取**第一个 ok**，排序用 close.json 的 closed_utc（不用字符串序）；
   ailed / borted / unclosed 的重试**不占位**；多个 ok 取最早，其余标 superseded。
3.  030Z / 1640Z 形如 …-s101-20261112T0030Z，**不匹配 -W1 前缀 → 永不是 Gate A 证据**，
   in_gate_a_table=false，只进 CYCLE_DEPENDENCE_STUDY.md preliminary 小节。
4. **aseline_version 是承重件**：省掉它，A3a-FD001 会按「最早 ok」选中 pre-versioning 的
   …T1505Z（10.5455，udit_verdict=fail、已 retired）而非 v9 的  030Z（11.3167），差 0.77 RMSE。

**格数纠正**：主表 24 = {A0, A1, A2, A3a, A4, A5} × 4（**A3a 替换**原 A3p 四格，不是额外加）；
A3p 参考臂 4 格（FD001 已完成）；**合计 28 格，待跑 27 格**。

**A3p-FD001-W1 的 note 口径**：与  030Z 逐位相同（同 init、同 seed），**非独立证据，同格只计一次**。

**按本条实测的当前准入：6 / 24** —— A0×4（30.0986 / 26.2874 / 39.9834 / 37.1218）+
A1-FD001-W1r2（11.2806）+ A1-FD003-W1r2（12.0306）。A3p-FD001-W1（10.8951）属 frozen 表，不计入主表。

**遗留阻塞（已报主控，待裁）**：调度器 un_wave1.sh 的 cell_done() 仍按 (variant, subset, seed)
匹配**任何** v9 ok run，不要求 -W1 前缀。实测它会把 A3a-FD001 判成已完成（因为 A3a-FD001-…T1640Z 是 v9 ok），
而前缀规则又不准入该 run → **该格永久缺失，主表停在 23/24**，且它正是 G2 主臂的第一格。
需二选一：收窄 cell_done() 到 -W1 前缀（推荐），或恢复  030Z 别名。

**发车冻结**：wave1_init_{A1,A2,A3a,A4,A5} 五份 init 的骨干键两两比对通过后，才恢复发车。

