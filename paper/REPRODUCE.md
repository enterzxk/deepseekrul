# 复现说明

本文件是把 `rul-calibration` 分支上全部结论从零复算的操作入口。
它不解释结论（那些在 `CALIBRATION_DECISION.md` 与 `NCMAPSS_RESULTS.md` 里），
只说明每一步跑什么、产出落到哪、以及怎么核对产出没被改动。

**先读这一句**：所有数字都由脚本从 JSON 生成，没有一个是手工抄的。
如果你看到某个数字对不上，先怀疑你手上的 JSON 版本，再怀疑脚本。

---

## 0. 环境

| 角色 | 位置 | 说明 |
|---|---|---|
| 重算机 | `lab-a40:~/zxk/bearing-rul-pi` | 有 GPU、torch、numpy、scipy、h5py |
| 控制机 | 本机 Windows | 只有 numpy + h5py；用来做数据构建与诊断，以及汇总表格 |

A40 上使用的解释器是 `/home/yndx/miniconda3/envs/a40_env/bin/python`，
运行时需要把仓库根加进 `PYTHONPATH`：

```bash
cd ~/zxk/bearing-rul-pi
export PYTHONPATH=$PWD/src:$PWD/scripts
export OMP_NUM_THREADS=4
```

**A40 无法访问外网**（没有 DNS）。任何下载都必须在本机完成再 `scp` 过去。

---

## 1. C-MAPSS 部分

### 1.1 数据与骨干预测（已有，不需重跑）

- 原始数据：`data/`（`train_FD00x.txt` / `test_FD00x.txt` / `RUL_FD00x.txt`）
- dev 逐发动机交叉预测：`experiments/cal/dev_preds/{FD001..FD004}/A_s*.csv`
- 官方测试集预测：`results/dsh/e21/preds/{FD001..FD004}_{A,BE,PE}.csv`

这三组文件是整条校准线的输入。`experiments/cal/RUN_RECORD.md` §1 给出了
它们各自的生成来源、5 折 × 3 种子的折叠方式，以及**已知的未修正缺陷**
（dev 是 3 模型集成、test 是 15 模型集成）。

### 1.2 主对比（三骨干 × 四舰队，七个方法）

```bash
for arm in A BE PE; do
  python scripts/cal/v2_compare.py --arm $arm            # pooled
  python scripts/cal/v2_compare.py --arm $arm --loso     # 留一舰队
  python scripts/cal/v2_compare.py --arm $arm --per-subset
done
```

产出 `experiments/cal/v2_<arm>/compare.json`、`v2_loso_<arm>/loso.json`、
`v2_persub_<arm>/per_subset.json`。

### 1.3 不可消除性（等渗在 test 上拟合、收缩族、逐分箱 oracle）

```bash
python scripts/cal/diag_bias_tradeoff.py     # -> experiments/cal/bias_tradeoff.json
python scripts/cal/diag_band_shift.py        # -> experiments/cal/band_shift.json
python scripts/cal/diag_identifiability.py   # -> experiments/cal/identifiability.json
python scripts/cal/diag_cap_mass.py          # 顶格质量，复现 v1 的那个 8 倍错误
```

### 1.4 汇总表格

```bash
python scripts/cal/v2_tables.py              # -> experiments/cal/v2_tables.md
```

### 1.5 冻结校准器的端到端复算

```bash
python scripts/cal/freeze_calibrator.py      # 只在需要重新冻结时跑
python scripts/cal/verify_frozen.py          # 仅用 JSON + 提交的预测重建 pmf
```

`verify_frozen.py` 当前输出 A 1.05e-06 / BE 1.45e-06 / PE 2.45e-06，三个都是 PASS。
这是"`experiments/cal/calibrators/` 里的权重确实能复现 `eval_report.json`"的唯一凭证。

---

## 2. N-CMAPSS 部分

### 2.1 下载（本机，约 4.3 小时，可分 8 段续传）

```bash
bash scripts/ncmapss/fetch.sh /tmp/ncmapss
```

八段各自 `curl -C -` 续传，全部长度校验通过后才拼接；任一短段会直接拒绝，
因为短段会静默破坏拼接、一小时后才在 inflate 处报错。

### 2.2 解包（本机，约 10 分钟）

```bash
python scripts/ncmapss/extract_inner.py \
  --zip C:/Users/20562/AppData/Local/Temp/ncmapss/full.bin \
  --out C:/Users/20562/AppData/Local/Temp/ncmapss/extracted \
  --members N-CMAPSS_DS
```

外层 15.76 GB 的 `data_set.zip` 以 deflate 存储，内层成员头在整条流之后，
无法随机访问。脚本把外层 inflate 流包成一个 `read(n)` 读取器，直接喂给内层
local header walker，**零中间文件**。每个成员写出时校验 `usz` 与 CRC。

产出 10 个 `.h5`（DS01–DS07、DS08a、DS08c、DS08d）。
**DS08d 会通过校验但打不开**——zip 声明 2,885,034,848 字节，HDF5 超级块记录
2,885,034,880。CRC 是对的，所以这是上游缺陷；不要零填充修补。

### 2.3 周期化（本机，约 1 分钟）

```bash
python scripts/ncmapss/build_cycles.py
```

把每飞行周期的 6,000–13,000 个时间戳聚合成一条 18 维记录，再做 10 周期窗口
（mean/std/delta → 54 维）。产出 `experiments/ncmapss/data/*.npz`（共 1.4 MB）
与 `summary.json`。**`capped_share` 应当全为 0.000**——这是"N-CMAPSS 无删失"的直接核对。

### 2.4 骨干（A40，约 4 分钟）

```bash
python scripts/ncmapss/train_backbone.py
```

按 `paper/NCMAPSS_PROTOCOL.md` §3 原样：54-64-32-1、tanh、dropout 0.1、MSE、
固定 300 epoch、无早停、3 种子、dev 留一单元交叉。
产出 `experiments/ncmapss/preds/{preds.csv,quality.json}`。

### 2.5 校准对比（A40，约 9 分钟三个协议）

```bash
python scripts/ncmapss/cal_nc.py --protocol all --n-boot 2000
```

**关键实现细节**：`layer.set_grid(100)` 必须在 `import v2_compare` 之前执行，
因为 `v2_compare` 在导入时就把 `K`/`MAX_RUL` 绑定成了值。网格是 0..100 而不是 0..125，
因为 N-CMAPSS 的 RUL 以飞行周期计、上界 59–99。层的常数写成 $S/125$ 的比值形式，
`S=125` 时与 C-MAPSS 形式逐系数相同——这一条已用 A 臂重跑逐字段验证过。

### 2.6 判据诊断（本机，只需 numpy）

```bash
python scripts/ncmapss/diag_nc.py            # -> experiments/ncmapss/diag_nc.json
python scripts/ncmapss/tables_nc.py          # -> experiments/ncmapss/tables.md
```

`diag_nc.py` 算的是"任何单调映射能不能修好分箱偏差"，它**不加载任何校准层**，
只用 `preds.csv` 里的 $(\hat y, y)$。这是全篇最容易被独立复核的一步：
拿一张 CSV 就能重算 §6 的核心结论。

---

## 3. 怎么核对"结论没被改过"

| 想核对的东西 | 看哪里 |
|---|---|
| 协议是否在实验之前冻结 | `git log --follow paper/NCMAPSS_PROTOCOL.md`，最早提交 `cab94d9` 早于任何 N-CMAPSS 数字 |
| 表格是否手抄 | 每个 `.md` 顶部注明生成脚本；`tables.md` 由 `tables_nc.py` 直接从 JSON 渲染 |
| 校准器是否真能复现 | `scripts/cal/verify_frozen.py`，误差量级 1e-6 |
| 重构是否改变了历史结果 | `git show 4a409e8 -- scripts/cal/layer.py`，以及该提交说明里的逐字段比对结论 |
| 哪些数字不可信 | `CALIBRATION_DECISION.md` §6 与 `NCMAPSS_RESULTS.md` §10，两份披露清单 |

---

## 4. 已知会在复算时出现的差异

1. **wall-clock 计时不同。** `compare.json` 的 `meta.fit_seconds` 每次都不一样，其余字段应当逐位相同。
2. **A40 与其它机器的浮点累加顺序。** 若换机器，`crps` 一类指标末位可能差 1e-12 量级；
   本项目所有比对都在同一台 A40 上完成。
3. **`torch` 版本。** 层的训练用了 `torch.optim.Adam` 与 `torch.special.ndtr`，
   不同版本的默认 `eps` 未变，但若结果有系统性偏移，先查版本。
