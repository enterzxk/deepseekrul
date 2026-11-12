# N-CMAPSS 子集台账索引

状态：四个子集全部封存
最后更新：2026-11-13 03:35 (+08)
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

每子集：dev 60 次训练 × 2 评价点 = **120 行**；test 路径尚未实现，本批不产生 test 行。
四子集合计 480 行。

## 不对称说明

DS02 试点批无权重、四子集有权重（两个评价点各一份 `ckpt_ev<epoch>.pt`，240/240 对 sha256 互不相同）。
**不要为了形式整齐去重跑 DS02**——重跑本身又是一次新 run。
