# 机制两数的现状与边界（同龄排序 / 传感器置换）

日期 2026-11-13 · 执行 tmh · 探针 `probes/mechanism_table.py` + `probes/d_mechanism_m3b.py`

## 1. 现在有哪些数

**只有三个臂有数，而且全部来自 a40 时代的 checkpoint。**

| 臂 | 同龄排序准确率 | 置换退化（绝对，周期） | 置换退化（相对） | 来源 |
|---|---|---|---|---|
| B0 | 0.7699 | **+24.6** | +195.4% | a40，FD001，40ep |
| B2 | 0.5595 | **+0.45** | +1.81% | a40，FD001，40ep |
| AGE | 0.00%（构造性） | 0 | 0% | a40，无参数 |

绝对量差 **55 倍**，相对量差 108 倍。主结论用绝对量。

## 2. 边界一：这三个数不能当 3090 的基线

机制探针是**模型对自身**的诊断，不是跨机性能比较——但**checkpoint 是 a40 训的**。3090 上的 B0 是另一份权重，它的同龄排序与置换退化**没有测过**。

所以：
- 引用 B0 的 `0.7699 / +24.6` 时必须写「a40 checkpoint」；
- **不能**把这三个数和 3090 的 M2/M3B/M3C/M4/M5/M6/M7 放进同一张表比较——那正是「跨机不混」要防的。

## 3. 边界二：10 个臂是 `not_run`，原因是 checkpoint 在 3090

| 状态 | 臂 |
|---|---|
| 有数 | B0 / B2 / AGE（均为 a40） |
| `not_run` | B1 / B3 / M2 / M3B / M3C / M4 / M5 / M6 / M7 / AAR |

`not_run` 在表里是**一行带原因的记录**，不是空白——「缺席」和「零」在表里长得一样、意思相反，这是 `mechanism_table.py` 强制写 `status/reason` 的原因。

## 4. 边界三：到达算子会压低置换退化

`B2/B3/M3B/M3C/M6/M7` 的预测经过 `first_passage`，损伤单调累积，**这个结构本身就会让置换退变小**。所以它们的 `degradation_pct` **不是「传感器贡献占比」**。

`B0/B1/M2/M4/M5/AAR` 走 `reg_head`，没有这层抑制，置换退化直接反映传感器依赖——**两类臂的百分比不可横向比较**，这也是主结论必须用绝对量的原因之一。

`AGE` 是构造性 0：它从不读传感器。

## 5. 复现与补齐

```
cd <repo> && PYTHONPATH=. python docs/paper_b_hc/probes/mechanism_table.py \
  --root experiments/hc/d80 --data-dir data/CMAPSS \
  --init-from experiments/hc/tmp/wave1_init_A1_s101.pt \
  --shared-dir experiments/hc/d80 --device cuda:0 \
  --out docs/paper_b_edc/sprint_24h/MECHANISM_TABLE.csv
```

输出 CSV 列：`arm / subset / seed / status / reason / val_rmse / rank_acc / pairs / perm_mean / degrad_abs / degrad_pct / run_dir / note`。

**跑完之前，阶段四「每个候选模块都有两个数」这条不成立**——现有 3/13，且这 3 个是另一台机器的权重。
