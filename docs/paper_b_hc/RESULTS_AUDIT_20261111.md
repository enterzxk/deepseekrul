# DSPT-HC 结果审计 · 2026-11-11

**2026-11-11 22:35 B5 更正：9 倍说法作废，改为 Jensen 关系 + 3.94 平方误差质量比**（见 §2c 风险 1，原文保留删除线）

> 审计人：实验结果管理员
> 范围：冻结基线的只读代码审计（`paper_b/configs/protocol.json`、`paper_b/training/data.py`、
> `paper_b/training/train.py`、`paper_b/models/dspt.py`），外加台账可用性自检与代码 md5 基准核验。
> **代码基线以本地仓库为准**（五份文件 md5 已与主控给定值逐一核对一致，见 §2.0）；
> **a40 产物路径以新盘 `<HC_ROOT>` = `/media/sangfor/vdb/user/zxk/deepseek-rul` 为准**，
> 旧目录 `/home/yndx/deepseek-rul` 仅为只读参考，不得再写入。
> 方式：只读，未修改任何被审文件；所有结论附 `文件:行号`，关键结论用本地数据独立复算。
> 当前 run 数：**0**（`<HC_ROOT>/experiments/hc/` 为空，`registry/launch`、`registry/close` 均为空）。

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

同步后核对值（2026-11-11 22:10）：

| 文件 | md5 |
|---|---|
| `registry/runs.csv` | `9713326e3cfdfed2282b52dc842b7d62` |
| `RESULTS_INDEX.md` | `b9941a0ca47e02a5bf22fb32426185f6` |

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
| **F11** | **中** | **B5 错误表述已扩散进代码 docstring**：`metrics.py:30-31` 写 "a long-lived engine contributes more rows and can dominate the score"，`train.py:107-109` 写 "over-weights long-lived engines (a FD004 engine contributes ~9x the rows of a short one)" —— 与冻结 val 下"每引擎恰好 5 窗、等权"的事实不符，需由 tmh 改为 Jensen 表述。 | `metrics.py:30-31`、`train.py:107-109` |

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
| **R12** | 把 `metrics.py:30-31` 与 `train.py:107-109` 的 docstring 改为 **Jensen 表述**（每引擎 5 窗等权；池化 = 逐引擎 MSE 的均方根 ≥ 发动机级平均 RMSE；差异来自平方放大尾部，不是行数加权） | **P1（首个 run 前）** | 我 grep 确认两处不再出现 "more rows" / "~9x" |

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

- 已登记 run：**0**
- 已审计文件：4 个（只读，未修改）+ 5 份代码 md5 三处对照（本地 / 新盘 / 旧盘）
- 已修台账问题：`runs.csv` 去掉 UTF-8 BOM（32 列不变）；远端镜像同步（三份文件 md5 一致）
- 已更正审计结论：**B5（22:35）** —— 池化口径"9 倍加权"说法作废，改为 Jensen 关系 +
  平方误差质量比 3.26/3.17/2.54/3.93；原文保留删除线，未删改
- 旧目录违规写入：**0**
- 阻塞项：**0**
- 待 tmh 处理：R1 / R8 / R9（P0）、R2 / R3 / R6（P1）、R4 / R5 / R7（P2）
- 下一步：等 tmh 在 `<HC_ROOT>` 上的首个 launch 文件；到达后按 §5 节奏登记 + 抽查
