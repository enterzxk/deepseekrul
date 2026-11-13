# N-CMAPSS 子集台账索引

状态：四个子集全部封存
最后更新：2026-11-13 13:20 (+08)
维护人：实验结果管理员（单一写入者）

> 每子集一张台账（`runs_<subset>.csv`），同一套 64 列 schema（见 `SCHEMA.md`）。
> **封存后只许追加 `status=superseded` 的更正行，不许改已有行。**
> `protocol_blob` 是**逐行字段**，所以封存过的子集在结构上不可能被后来的锚改写。

## 证据资格（2026-11-13 变更，**先读这一节再引用任何数字**）

八张表（dev 480 行 + test 80 行）**原样保留，不改不覆盖**，但**退出以下主张的主证据资格**：

- 架构优劣（N1 / N2 / N3 之间谁更好）；
- 模块效应（N3 − N2 是零、正还是负）；
- 跨数据集迁移有效性。

降级为「**本实现下的描述性结果**」。原因：三处已核实的实现缺陷，且代码指纹与全部 80 行测试记录一致——

1. **传感器 token 列序错配**：特征实际是 `[14 个均值, 14 个标准差, 14 个斜率]`，DSPT 却按
   `reshape(14, 3)` 切成「传感器 token」，于是每个 token 是三个不同传感器的同一个统计量。
   影响 token 切分、`sensor_score` 池化权重与 SA；**MS 卷积沿飞行时间轴，不受影响**。
2. **50 个物理特征未标准化**：`build_matrix` 只对 age 做 `cycle/q95`，其余原值拼接，
   列量级跨度约 **8.4 个数量级**（最大 8208，最小 3e-5）。协议 §4.1 明写 z-score 折内拟合——
   这是**实现违反冻结协议**，不是调参选择。
3. **验证期 dropout 未关**：`_loss_on(..., train=False)` 在 epoch 循环内、`net` 仍是 `train()` 状态时调用，
   `net.eval()` 在循环之后。`epoch*` 的选轮曲线一直带 dropout 噪声。
   **N0 无 dropout 层，不受此影响。**

**N0 的时钟基线不受三条影响**：只吃 age（唯一被正确缩放的列），不做 tokenization，不参与选轮。

四条引用约束：

- 测试集**已经开封**，修复后重评**不能称为首次独立验证**，必须标「实现修正后的重复评价」，这句话要进论文正文。
- **修哪个缺陷不能按已开封的测试成绩来选**——修复方案只依据协议与 dev 冻结。
- 修 ③ 会改变随机数消耗，修复后 `fixed_passes:200` 的数字与旧批**不可逐位比较**，这是预期不是回归。
- `ρ = 0.983` 是**秩相关**，只能说「误差排序与寿命距离高度一致」，不能写成「解释了 98% 的误差」。

| subset | 台账文件 | 行数 | status | git_sha | protocol_blob | 文件 sha256 | 审计 |
|---|---|---:|---|---|---|---|---|
| DS02 | `runs_DS02.csv` | 120 | **sealed** | `49f6939` | `a3c40c634da7a3bb44ace788f74896b17e51c3fb` | `c9de7b187f4c0a064ac7d1dcc0ff4d6887597ca1efb2a81476ddd5bcfaa12bf8` | `partial`——**试点批无 checkpoint**，指标可从 `pred.csv` 复算、权重不可验 |
| DS05 | `runs_DS05.csv` | 120 | **sealed** | `b889bd2` | `a3c40c634da7a3bb44ace788f74896b17e51c3fb` | `7cf4e237237952385425d07b2b2a68518b0e63b813c3d5e191492daf47dc6b6a` | `pass`——480 行全量机械审计，0 失败 |
| DS06 | `runs_DS06.csv` | 120 | **sealed** | `b889bd2` | `a3c40c634da7a3bb44ace788f74896b17e51c3fb` | `9ae1f8759a92acb95360dbf2cfe7ca73a7c6c444cdfd70e8e64077a5bdeddbda` | `pass` |
| DS07 | `runs_DS07.csv` | 120 | **sealed** | `b889bd2` | `a3c40c634da7a3bb44ace788f74896b17e51c3fb` | `affae9e5693a2909a3d8a94be6be07e8e6db71924376c8ac325d25aaa98dc1fe` | `pass` |

## 测试批（`phase=test`，第 15 代锚）

测试行**不进**已封存的 dev 表——那四张表只许追加 `superseded` 行。测试批另开四张表，同一套 64 列 schema。

| subset | 台账文件 | 行数 | status | git_sha | protocol_blob | 文件 sha256 | 审计 |
|---|---|---:|---|---|---|---|---|
| DS02 | `runs_DS02_test.csv` | 20 | **sealed** | `ad53371` | `6ec1c014008df80156a13101fb600c9e3e00cfdc` | `2f1483ec642dcf0bdf786360dc798872eaa96cd4c3a30d516962af1fa40a5bc0` | `pass` |
| DS05 | `runs_DS05_test.csv` | 20 | **sealed** | `ad53371` | `6ec1c014008df80156a13101fb600c9e3e00cfdc` | `2c0829e4e3eb2f8b3c0d79552da710c0b583bedbed03b3a4a43ca7e1644707da` | `pass` |
| DS06 | `runs_DS06_test.csv` | 20 | **sealed** | `ad53371` | `6ec1c014008df80156a13101fb600c9e3e00cfdc` | `20133e0db6997e4dd90eaf84093d66e41a2eaa677c9cab2dc2a2333d069f01dd` | `pass` |
| DS07 | `runs_DS07_test.csv` | 20 | **sealed** | `ad53371` | `6ec1c014008df80156a13101fb600c9e3e00cfdc` | `142686ce9eb17213fa806a8cf1478d733317c9310a235682ee9529b978c25237` | `pass` |

**封表前的全量机械审计（80 行逐行，0 失败）**：`ckpt_sha256` 与磁盘实算一致、`n_flights`/`n_units` 与 `pred.csv` 一致、机号等于清单的 `test_units`、`freeze_id`/`protocol_blob`/`data_sha256` 与清单一致、`eval_start_utc` 等于 opens 记录且晚于 `freeze_ts_utc`、`coincident` 全 `false`（40 对里没有一对数值相同）。

### 20 个聚合列按逐机等权从 `pred.csv` 复算

运行器的 `metrics()` 在测试行把**池化值写进了 `_unit` 和 `seg_*` 两类列**——它对整条误差向量算 RMSE，从不分机。dev 上恒等（每折只留出一台机，池化 == 逐机等权），所以 480/480 相等**不是检查通过，是一个不可能失败的检查**。

封表时这 **20 列**（8 个 `_unit` + 12 个 `seg_*`）全部从 `pred.csv` 按逐机等权重算，`_flightpooled` 两列保持运行器的池化值（已核一致）。`metrics.json` 里的池化原值不丢。每行 `registry_note` 都带 `AGG=per_unit_equal_weight` 与 `n_units`。

**`seg_*` 的口径此前没在任何地方声明过**——这次一并定死：12 个 `seg_*` 列 = 先按机算该段 RMSE，再对**该段非空的机**等权平均。

**一条必须跟着数字走的前提**：`seg_early_absolute_*` 只统计 RUL ≥ 75 的飞行，寿命不到 75 的机在该段没有行——
DS02 只有 unit 14 一台（11、15 的寿命是 58/66），DS05/DS06 各 3 台（unit 9 缺席），DS07 4 台全在。
**DS02 的 `seg_early_absolute_*` 是单机数字**，不要当成 3 台机的均值引用。

**但「逐机等权」并非在所有格子上都能被断言**，判据**按聚合族分**：RMSE 族（含 12 个 `seg_*`）与
NASA 族只在**恰好一台机贡献数据**时退化；线性族（MAE/Bias）在各机样本数相同时退化。
1600 格里 **40 格 `degenerate`**——全部是 DS02 的 `seg_early_absolute_capped` / `_raw` × 20 行
（该段只有 unit 14 够到 RUL ≥ 75）；`pass_coincident` 为 **0**。
逐格标定见 [TEST_AGGREGATION_CALIBER.md](TEST_AGGREGATION_CALIBER.md)，重算 `python scripts/nc_agg_caliber.py`。
**封存表字节不变**——数值本来就对，退化只说明该列在这些行上不携带聚合口径的信息。

## 事故记录：第二轮覆盖第一轮产物（2026-11-13）

**根因**（都在 launcher，两条）：

```bash
--out-root experiments/ncmapss     # 目录名与第一轮完全相同
                                   # 且没传 --ledger，写回默认路径
```

新台账名定了 `runs_<subset>_r2.csv`，但**产物目录名没跟着改**——等于只给台账换了名字，
`preds_path` / `ckpt_path` 全部指回第一轮的目录。

### 损坏面（按内容判定，不按 mtime 猜）

| 类别 | 范围 | 判定方式 |
|---|---|---|
| **封存行被污染** | **DS06 4 行**：`DS06_N1_LOEO-1_s101_iv`（25.79 → 12.88）、`_s101_f200`（15.27 → 13.02）、`_s202_iv`（23.60 → 9.06）、`_s202_f200`（16.98 → 11.16） | 拿封存行自己的 `preds_path` 复算，与封存值不符 |
| **权重被换、指标仍对** | DS06 另 2 个 N0 目录 | N0 对四处修复全部不敏感（验过三次逐位不变），所以**指标复算会通过，但 checkpoint 已是第二轮的** |
| **权重来源失效** | `experiments/ncmapss/DS02/` 下 120 个目录 | 见上一节撤回 |

**「指标对得上」不等于「文件没被换」**——DS06 的 N0 就是这一类。只靠指标复算做验收会漏掉它。

### 未受影响的部分（已核，不是推测）

**测试批 80 行完好**：逐行核 `ckpt_sha256` 与磁盘实算一致、`rmse_raw_unit` 从 `pred.csv` 逐机等权复算一致——
**80/80 通过，0 缺失**。第二轮跑的是 dev 格，test 目录名（`<subset>_<rung>_TEST_*`）与 dev 不重叠。

### DS02 路径映射（封存表不改）

DS02 封存行的 `preds_path` 写的是归档前的位置，例如
`experiments/ncmapss/N0_LOEO-2_ev20/pred.csv`；实际文件在 **`DS02_pilot/`** 下：

```
experiments/ncmapss/<run_id>/…  →  experiments/ncmapss/DS02_pilot/<run_id>/…
```

从 `DS02_pilot/` 重新复算：**120/120 通过，0 缺失**。所以 **DS02 的封存证据本身完好**，
缺的只是这条映射没写进文档。封存表**不改**——改了会动 sha，而 sha 已经写进引用。

### 恢复方案

- **DS06 那 4 行**：用第一轮代码（`ad53371`）在**隔离目录**重跑这 4 格，
  验收标准是**指标与 `ckpt_sha256` 双双对上封存值**。只看指标不够（理由见上表第二行）。
- **DS02 权重**：不追。回到 `partial` 原义。
- **封存表**：八张表一个字节不动（四份 dev sha256 已复核未变）。

### 已加的两道闸门（第二轮重启前）

1. `--out-root experiments/ncmapss_r2`，且显式传 `--ledger`；
2. **preflight：目标目录已存在就拒绝启动**——这次的损失全部来自「覆盖时没有任何东西阻止它」。

## 全量机械审计（封表前做的，不是抽样）

480 行逐行核三件事，**0 失败**：
1. `ckpt_sha256` 与磁盘文件实算 sha256 一致（480/480）；
2. `rmse_raw_unit` 从 `pred.csv` 的 `y_true_raw`/`y_pred` 复算，与台账差 < 1e-6（480/480）；
3. `n_flights` 与 `pred.csv` 行数一致（480/480）。

命令见 `SCHEMA.md` §8；审计脚本不进仓库，只留方法与结果。

## DS02 跨批次复现（重要）

DS02 被跑了两次：试点批 `49f6939`（无 checkpoint，封存为 `runs_DS02.csv`）与四子集批 `b889bd2`（有 checkpoint，产物在 `experiments/ncmapss/DS02/runs_dev.csv`，未登记）。

**两次的 480 个指标值（120 行 × 4 项）逐位相同，最大绝对差 0.0。** 这是跨代码提交、跨「有无 checkpoint」的独立复现，比任何单批内部检查都强——记在这里，供论文的复现性一节引用。

## DS02 权重的取得方式（`partial` 的边界）——**2026-11-13 撤回**

DS02 封存行的 `audit_verdict=partial`，原因是**试点批**（`git_sha=49f6939`）没有保存 checkpoint，
所以那 120 行本身的权重无法核验。**这不等于 DS02 没有权重可用**：

- 四子集批（`git_sha=b889bd2`）在 `experiments/ncmapss/DS02/` 下留了 **120 个 checkpoint**，
  两个评价点各一份（240 个文件，逐对 sha256 互不相同）；
- 这批产出的指标与封存表**逐位相同**：120 格点 × 24 个数值列，最大绝对差 **0.0**，
  连 `epochs_run` 都一致（经理独立复核过）；
- 那 120 个 checkpoint 在封表前的全量机械审计中**逐个核过 sha256**，且对应指标可从 `pred.csv` 复算。

**撤回（2026-11-13）**：上面这条交叉引用**不再成立**。第二轮重跑的 launcher 复用了同一个
`--out-root experiments/ncmapss` 且没传 `--ledger`，`experiments/ncmapss/DS02/` 下那 120 个目录
在 12:40 被**第二轮的产物覆盖**。所以现在那里装的是第二轮权重，**与封存指标对应的不是同一套模型**。

**DS02 因此回到原始的 `partial` 状态**：封存行本身没问题（指标可从 `pred.csv` 复算），
但**权重要么取试点批（从未保存），要么就是不可得**。追回需要单独开工作树、用 `49f6939` 重跑
120 格——为一条 bonus 不划算，除非论文明确需要。

封存行**一个字节没改**（sha256 仍为 `c9de7b18…`）。

## 行数模型

每子集：dev 60 次训练 × 2 评价点 = **120 行**；test 10 格 × 2 评价点 = **20 行**。
四子集合计 **480 dev + 80 test = 560 行**。

## 不对称说明

DS02 试点批无权重、四子集有权重（两个评价点各一份 `ckpt_ev<epoch>.pt`，240/240 对 sha256 互不相同）。
**不要为了形式整齐去重跑 DS02**——重跑本身又是一次新 run。

测试批的代码是 `ad53371`，dev 两批是 `49f6939` / `b889bd2`：测试路径在 `b889bd2` 时还不存在（§6.10 之后才定义），所以 md5 必然不同。
`data.py` / `features.py` / `evaluation.py` 三份逐字节未变，变的只有 `train.py`（加 `load_split(fitted=)` 与 `fit_rung(no_holdout=)`）和 `run_loeo.py`（加测试路径与闸门）。
