# Task C 四类条件误差 · 前置可行性核查（2026-11-14）

> **本文不含 Task C 的任何统计结果。** 它只回答一件事：**这四类分析在现有产物上做不做得成、以及哪一列不能按字面用。**
> 统计量、配对差、区间由 work1 产出后落 `UNRESOLVED_ERRORS.md`；本文是它的前置条件单。
> 本地实测：`paper_b/results/raw/<tag>/<subset>/<seed>/val_pred.csv` 存在，`A8_*` 与 `F_*` 均可读，**Task C 不需要等 3090**。

## 0. 一句话结论

**四类里只有第 ③ 类可以直接从 `val_pred.csv` 算；第 ① 类必须先把 `cut` 换成相对位置（否则每个箱只有 1 台发动机）；第 ②④ 类需要重建工况簇与传感器 z-score——它们不在 `val_pred.csv` 里，也没有落盘，但可由固定种子确定性重建。**

## 1. `val_pred.csv` 实际只有四列

```
unit, cut, y_true, y_pred          （实测 FD001/s101：100 行）
```

**没有 `cycle` 列，没有工况簇标签，没有任何传感器特征。** 所以：

| 第几类 | 需要的字段 | 在 `val_pred.csv` 里？ | 结论 |
|---|---|---|---|
| ① 同年龄下跨发动机误差 | 年龄（相对位置）、`unit`、误差 | `unit` 有；**`cut` 是绝对周期，不是共享年龄** | **可做，但必须先转换（见 §2）** |
| ② 工况与切换后误差 | 工况簇标签、切换位置 | **没有** | **需重建工况簇（见 §3）** |
| ③ 封顶区 vs 退化区 | `y_true` 是否 `=125` | **有** | **可直接算** |
| ④ 控制年龄+工况后传感器增量信息 | 工况簇 + 传感器特征 | **没有** | **需重建工况簇 + 原始传感器（见 §3）** |

## 2. `cut` 不是共享年龄——按字面分箱会让结论失效

`cut` 是**绝对周期号**（`FD001/s101` 实测范围 `54–269`），因为五个截断点是**每台发动机自身寿命的 `40/55/70/85/95%`**，而各机寿命不同。

**实测后果（`FD001/s101`，20 台 × 5 点 = 100 行）**：

```
按原始 cut 分箱：74 个箱，每箱中位数 1 行   ← 箱内只有 1 台发动机，"跨发动机离散度"无定义
按相对位置分箱：5 个箱，每箱 20 台          ← 这才是设计
```

**验证**：把每台机的 `cut` 除以它自己的寿命（`max(cut)/0.95`），比值全部落在 `0.40 / 0.55 / 0.70 / 0.85 / 0.95` 五档附近（实测离散 `0.398–0.403`、`0.5466–0.5527`、`0.6977–0.7027`、`0.8474–0.8537`、`0.95` 全等），**所以相对位置是干净的五档，绝对 `cut` 不是。**

**所以第 ① 类必须用「截断点序号 `0–4`」或 `cut/寿命` 作年龄轴。** 用原始 `cut` 会得到「每个年龄上只有一台机」的伪结果——**而那个伪结果看起来会像「同年龄下跨机差异极大」，方向正好支持一个错误结论。**

## 3. 工况簇与传感器 z-score：没有落盘，但确定性可重建

**不在 `val_pred.csv` 里**，而且 `paper_b/training/data.py` 里 `OpScaler` 只在内存中 fit、**没有任何 `joblib`/`pickle`/`np.save` 落盘**（实测 grep 无命中）。

**但它是确定性的**：

```
data.py:50   OpScaler(n_clusters=6, random_state=7)
data.py:61   KMeans(n_clusters=6, random_state=7, n_init=20).fit(ops)   ← 只在训练机上调
data.py:264  scaler = OpScaler(n_cond, split_seed).fit(tr)              ← split_seed=7
```

**所以第 ②④ 类的前提是「用同一 `split_seed=7` 重跑 `prepare_train` 拿回 `scaler`」，而不是从产物里 join。**
两条必须遵守的边界：

1. **必须在能复现同一 `train_FDxxx.txt` 与同一 `split_units(units, 7, 0, 0.2)` 的环境里重建**；换主机不影响（纯 numpy/sklearn），**但换 `split_seed` 或换数据文件会让簇标签与当年不一致**，那时算出来的「工况误差」不是这些 run 的工况。
2. **`n_init=20` 且 `random_state=7`**，所以重建结果逐次相同；**若重建出的 `means/scales` 与当年不同，说明环境或数据版本变了，必须停下查，不能接着算。**

**第 ② 类的「切换后前 N 个窗口」还需要原始序列的顺序**：`val_pred.csv` 只给截断点，不给窗口索引。切换位置要从 `scaler.labels(va)` 的标签序列里取（相邻标签变化处），**所以第 ② 类比第 ①③ 类多一层依赖**。

## 4. 第 ③ 类的口径已核，可直接用

```
y_true max = 125.0，100 行里 7 行落在 125（FD001/s101）
```

**封顶只作用在标签上，`y_pred` 未见裁剪**——与冻结主口径「标签封顶、预测不裁」一致。**所以第 ③ 类不需要再裁一次**；若有人把 `y_pred` 也裁到 125，那是敏感性版本，**不能与主口径同表**。

## 5. 给 work1 的三条硬要求（不满足则统计量无效）

1. **第 ① 类的年龄轴用截断点序号或 `cut/寿命`，禁止用原始 `cut` 分箱。** 分箱后每箱必须是 `n=20`（FD001）或对应的发动机总数，**若某箱不足该数，说明轴用错了。**
2. **第 ②④ 类必须先记录重建 `OpScaler` 的 `split_seed`、`n_init`、以及重建出的 `means/scales` 摘要**，与本文 §3 的三处代码行对上；对不上就停。
3. **每类都要报「n（发动机数 × 截断点）、口径、配对差、区间」四项**，其中配对差必须是**同一台发动机内**的差，不是两个子集均值相减。

## 6. 复核方式

```powershell
cd C:\Users\20562\Desktop\deepseek-rul
# 1. 四列与行数
python -c "import csv,io;r=list(csv.DictReader(io.open('paper_b/results/raw/A8_base_L70/FD001/s101/val_pred.csv',encoding='utf-8-sig')));print(list(r[0]),len(r))"
# 2. cut 不是共享年龄
python -c "import csv,io,collections;r=list(csv.DictReader(io.open('paper_b/results/raw/A8_base_L70/FD001/s101/val_pred.csv',encoding='utf-8-sig')));c=collections.Counter(int(x['cut']) for x in r);print('bins',len(c),'median',sorted(c.values())[len(c)//2])"
# 3. OpScaler 未落盘
Select-String -Path paper_b/training/data.py -Pattern "joblib|pickle|np\.save|to_json"
```
