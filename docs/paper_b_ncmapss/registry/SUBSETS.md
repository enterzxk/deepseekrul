# N-CMAPSS 子集台账索引

状态：四个子集全部封存
最后更新：2026-11-13 10:58 (+08)
维护人：实验结果管理员（单一写入者）

> 每子集一张台账（`runs_<subset>.csv`），同一套 64 列 schema（见 `SCHEMA.md`）。
> **封存后只许追加 `status=superseded` 的更正行，不许改已有行。**
> `protocol_blob` 是**逐行字段**，所以封存过的子集在结构上不可能被后来的锚改写。

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

**但「逐机等权」并非在所有格子上都能被断言**：若各机在该段贡献的样本数相同，两种聚合在代数上恒等。
80 行 × 20 列 = 1600 格里 **240 格是 `degenerate`**——`seg_late_absolute_*` 四个子集**全部**（该段恒为每机 25 个飞行）、
`seg_mid_absolute_*` 的 DS07（四台机寿命都 ≥ 75，各 50 个）、`seg_early_absolute_*` 的 DS02（只有 unit 14）；
`pass_coincident` 为 **0**。逐格标定见 [TEST_AGGREGATION_CALIBER.md](TEST_AGGREGATION_CALIBER.md)，
重算 `python scripts/nc_agg_caliber.py`。
**封存表字节不变**——数值本来就对，退化只说明该列在这些行上不携带聚合口径的信息。

## 全量机械审计（封表前做的，不是抽样）

480 行逐行核三件事，**0 失败**：
1. `ckpt_sha256` 与磁盘文件实算 sha256 一致（480/480）；
2. `rmse_raw_unit` 从 `pred.csv` 的 `y_true_raw`/`y_pred` 复算，与台账差 < 1e-6（480/480）；
3. `n_flights` 与 `pred.csv` 行数一致（480/480）。

命令见 `SCHEMA.md` §8；审计脚本不进仓库，只留方法与结果。

## DS02 跨批次复现（重要）

DS02 被跑了两次：试点批 `49f6939`（无 checkpoint，封存为 `runs_DS02.csv`）与四子集批 `b889bd2`（有 checkpoint，产物在 `experiments/ncmapss/DS02/runs_dev.csv`，未登记）。

**两次的 480 个指标值（120 行 × 4 项）逐位相同，最大绝对差 0.0。** 这是跨代码提交、跨「有无 checkpoint」的独立复现，比任何单批内部检查都强——记在这里，供论文的复现性一节引用。

## DS02 权重的取得方式（`partial` 的边界）

DS02 封存行的 `audit_verdict=partial`，原因是**试点批**（`git_sha=49f6939`）没有保存 checkpoint，
所以那 120 行本身的权重无法核验。**这不等于 DS02 没有权重可用**：

- 四子集批（`git_sha=b889bd2`）在 `experiments/ncmapss/DS02/` 下留了 **120 个 checkpoint**，
  两个评价点各一份（240 个文件，逐对 sha256 互不相同）；
- 这批产出的指标与封存表**逐位相同**：120 格点 × 24 个数值列，最大绝对差 **0.0**，
  连 `epochs_run` 都一致（经理独立复核过）；
- 那 120 个 checkpoint 在封表前的全量机械审计中**逐个核过 sha256**，且对应指标可从 `pred.csv` 复算。

所以对外交付权重的口径是：**取四子集批的 `experiments/ncmapss/DS02/` 目录**，不是封存行里的路径。
封存行**不改**（已封数据不动），这条交叉引用补在这里。

## 行数模型

每子集：dev 60 次训练 × 2 评价点 = **120 行**；test 10 格 × 2 评价点 = **20 行**。
四子集合计 **480 dev + 80 test = 560 行**。

## 不对称说明

DS02 试点批无权重、四子集有权重（两个评价点各一份 `ckpt_ev<epoch>.pt`，240/240 对 sha256 互不相同）。
**不要为了形式整齐去重跑 DS02**——重跑本身又是一次新 run。

测试批的代码是 `ad53371`，dev 两批是 `49f6939` / `b889bd2`：测试路径在 `b889bd2` 时还不存在（§6.10 之后才定义），所以 md5 必然不同。
`data.py` / `features.py` / `evaluation.py` 三份逐字节未变，变的只有 `train.py`（加 `load_split(fitted=)` 与 `fit_rung(no_holdout=)`）和 `run_loeo.py`（加测试路径与闸门）。
