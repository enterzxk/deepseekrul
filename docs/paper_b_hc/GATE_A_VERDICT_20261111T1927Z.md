# Gate A 判定快照 · 2026-11-11T19:27:38Z（a40 时钟）

> 本文件是**判定器的输出落盘**，不是人工汇总。重跑命令：
>
> ```bash
> python tools/gate_a_judge.py            # 在 a40 上运行（默认读 /media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc）
> ```
>
> **判定器**：`tools/gate_a_judge.py`
> **sha256**：`58c074a847f06f4af9b6876f8e8f246d258091746c2550afae7c8ecf6b6975d0`
> **数据来源**：只读每格的 `close.json` / `metrics.json` / `preds.csv`，**不读 `registry/runs.csv`**。
> **取格规则**：`HC-C-{variant}-{subset}-s101-W1` + `['', r2..r6]` 中**第一个 `status=ok`**；时间戳式 run_id（如 `…-20261112T0030Z`）不参与。

## 快照时刻的进度

13 格 `ok` / 剩余 15 格。已收：A0×4、A1×4、A2×4、A3p-FD001。在跑：`A3a-FD001-W1`。

## 输出原文

```
cell         FD001       FD002       FD003       FD004
A0         30.0986     26.2874     39.9834     37.1218
A1         11.2806     11.8031     12.0306     13.2087
A2         10.5925     10.2074     10.6517     11.7896
A3a             --          --          --          --
A4              --          --          --          --
A5              --          --          --          --
A3p        10.8951          --          --          --

G1    mean=+10.46%    [PASS]  +6.10% +13.52% +11.46% +10.74%
      rule: mean d(A1->A2) >= +1.0% and no subset < -2.0%
G2    mean=n/a        [PENDING]  n/a n/a n/a n/a
      rule: mean d(A2->A3a) >= +1.0% and >=3/4 subsets same sign
G2p   mean=n/a        [PENDING]  n/a n/a n/a n/a
      rule: report only; no threshold; not in the main table
G3    mean=-175.73%   [PASS]  -166.82% -122.72% -232.35% -181.04%
      rule: d(A1->A0) <= -10.0% on every subset
G5   {'G1': {'positive': 4, 'consistent': True}}

ROBUSTNESS (same deltas on the mean of the last 10 epochs;
            reported for comparison, NOT used as the verdict):
  G1   mean=+12.45%    +12.37% +3.54% +15.62% +18.29%
        RANKING FLIPPED (n=4): best-epoch FD002>FD003>FD004>FD001  vs  plateau FD004>FD003>FD001>FD002
  G2   mean=n/a        n/a n/a n/a n/a
  G3   mean=-148.30%   -143.30% -103.35% -201.37% -145.19%
        ranking stable (n=4): FD002>FD001>FD004>FD003

INTEGRITY: all collected cells pass baseline/written_by/preds/Jensen
```

## 符号约定

`Δ(A→B) = (RMSE_A − RMSE_B) / RMSE_A`，**正号 = B 优于 A**。与 `GATE_A_WAVE1_PLAN.md` §9 一致。

## 本快照的限定（写论文时必须带）

1. **G1 的效应量排序不可用**：单点最小值口径与平台均值口径下，四子集排序完全翻转（`FD002` 单点最好、平台最差）。可报「四子集全部正向、均值 +10~12%」，**不可报**「某子集效应最大」。
2. **触顶格**：`A2-FD001`(40/34)、`A2-FD003`(40/35)、`A2-FD004`(40/34) 早停未触发即打满 40 上限，且曲线在 best 之后走平或振荡——**不是仍在改善**。这三格的数值应读作「40 epoch 预算下的结果」，是保守值。`A0-FD001`(40/37)、`A0-FD003`(40/38) 同理，但属参考线。
3. **G3 的表述**：`A0/A1 = 2.2–3.3×` 只能表述为「**CycleOnly 小模型无法匹配传感器模型**」，不能写成「年龄信息不足」——A0 参数量远少于 A1，差值里混着容量。
4. **G2 的投影抽样噪声**：`|Δ| = 0.030123 RMSE`（run `HC-C-A2-FD001-s101-NOISE1`），G2 阈值 0.106 约为其 3.5 倍，但 σ 只有 1 个样本——**单次 Δ 只能报方向，不能报量级**。

## 未决

- **G2**：等 `A3a` 四格（`A3a-FD001-W1` 在跑）。
- **G4**：`cycle_intervention` 字段尚未纳入判定器输出（待 A2/A3a/A4/A5 四格齐后补）。
