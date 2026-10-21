# ORDER_02 执行报告

- 基线：`f333f59` / 核心实验 `e46afd1`
- 本轮提交：见文末
- 服务器：`/home/yndx/zxk/bearing-rul-pi`（A40，单卡）
- 训练/评价协议：`split_seed=7` 固定 80/20 划分；模型种子 2027-2028；24 epochs；
  checkpoint 由全窗口验证选择，所有部署式评价使用**同一** checkpoint。

> 所有数字来自 `experiments/order02/**/metrics.json`、
> `pseudo_test_predictions.csv`、`protocol_c_predictions.csv`；
> 汇总表在 `results/order02/`。未完成的单元格已明确标注，未参与任何结论。

---

## 1. 确定性错误修复与测试

| Issue | 修复 | 证据 |
|---|---|---|
| 1 tfA 缺周期坐标 | `TransformerWithFeatures` 增加 `feature_set`（raw / raw_base / raw_ema / raw_cycle / full），周期作为第 3N+1 通道输入 | 本报告表 3 的 tfA 已含周期；纯 tf 对照仍为 raw |
| 2 extras schema 不一致 | 四个路径（train / cutoff / test_mode / abs）统一为 `baseline(N) + cycle + [ref_count, ref_std, deviation, trend]` = **19 维** | `tests/test_extras_schema.py`：train/cutoff0.5/0.7/0.9/test_mode 全部 19，通过 |
| 3 可靠性证据不足 | 证据改为 4 个**可在线观测**量：参考样本数、参考早期稳定性(std)、当前窗口相对参考的平均偏离、最近 5 步趋势 | 单元测试通过；`Arec` 已在 FD002 端到端跑通（smoke，2 epochs） |
| 4 FD002 A/Aid 预测缺失 | 原因是旧汇总在 A/Aid 运行**完成前**生成；已重新生成完整预测 | 表 2、`results/order02/*.csv` |
| 5 统计独立性 | Bootstrap 以**发动机**为重采样单位，同一发动机的 3 个 cutoff 与所有种子一起被抽中 | `scripts/audit_stats.py`（engine-clustered） |

**聚合口径澄清（ORDER_02 要求）**：ORDER_01 主表中的 `10.743` 是
「三个 cutoff 各自 RMSE 的算术平均」；而 `results/“pseudo_summary.csv”` 中的
`12.205` 是「把 3 个 cutoff × 种子全部预测合并后的 pooled RMSE」。
两者含义不同，本报告统一使用 **pooled**，并在表格中注明。

---

## 2. 主结果（pooled pseudo-test，2 seeds，24 epochs）

### FD001

| 方法 | 参数量 | 全窗口 val | Protocol B (0.5/0.7/0.9 pooled) | Protocol C (80/120/160 cycles) |
|---|---:|---:|---:|---:|
| tf（纯 Transformer） | 144,146 | 12.462 | 11.822 | 12.861 |
| **tfA（Transformer + 完整 A 特征）** | 87,122 | **10.939** | **10.212** | **10.529** |
| A（候选主干 + A 特征） | 41,858 | 10.862 | 11.963 | 11.523 |
| Aid（A + Sensor-ID） | 42,754 | 12.115 | 12.983 | 12.423 |
| S2（A + 打乱身份，参数量匹配） | 42,754 | 11.849 | 11.631 | 13.328 |

### FD002

| 方法 | 全窗口 val | Protocol B | Protocol C | n(pred) |
|---|---:|---:|---:|---:|
| tf | 13.920 | 15.268 | 14.848 | 312 |
| tfA | 13.807 | 14.177 | 14.911 | 312 |
| A | 12.522 | 13.420 | 13.265 | 312 |
| **Aid** | **11.722** | **12.205** | **13.228** | 156（仅 1 个种子） |
| S2 | 未完成 | 未完成 | 未完成 | - |

---

## 3. 三条可复核的结论

1. **FD001 上真正的赢家是 tfA，不是候选主干。**
   tfA 在两种部署协议上都最好（10.212 / 10.529），比纯 tf 好 1.6 / 2.3。
   这与 ORDER_01 相反，原因正是 Issue 1 的修复：把**周期坐标**补进 Transformer 后，
   特征的作用才被完整体现。**结论：FD001 的收益主要来自输入表征，不是主干结构。**

2. **Sensor-ID 在 FD001 是负迁移，在 FD002 才有价值。**
   FD001：Aid 12.983 vs 参数匹配的 S2（打乱身份）11.631 vs A 11.963 ——
   身份信息不但没帮忙，还比"同样多参数但身份错乱"更差，说明它引入了过拟合。
   FD002：Aid 12.205 vs A 13.420（1 个种子，需补第二种子）。

3. **Protocol C 的排序与 Protocol B 不完全一致。**
   FD001：tfA(10.53) < A(11.52) < Aid(12.42) < tf(12.86) < S2(13.33)；
   FD002：Aid(13.23) ≈ A(13.27) < tf(14.85) < tfA(14.91)。
   **没有任何方法在两种协议上都稳定第一**，这一点必须写进论文而不加修饰。

---

## 4. 误差机制（`results/order02/band_errors.csv`）

FD001 分寿命段（pooled）：

| 方法 | [0,25) | [25,50) | [50,100) | [100,126) |
|---|---:|---:|---:|---:|
| A | 6.09 | 8.09 | 14.80 | — |
| tfA | 见 CSV | 见 CSV | 见 CSV | 见 CSV |

**临近失效段 [0,25)** 仍是候选主干的弱项（A 6.09），而 tfA 在整体上更好；
但这不等于"tfA 更安全"——需要单独报告**高估（late）比例**：
FD001 tfA 的 late fraction 为 0.692，是全部方法中最高，severe-late(>10) 18 例；
A 为 0.533 / 28 例。**tfA 更准但更"敢报晚"，这是安全关键场景的实质风险**，
不能只用 RMSE 下结论。

---

## 5. 工况条件健康参考与可靠性（RACDR 原型）

已实现但**尚未验证**：
- 参考量仍为「个体前 5 周期均值」（全局），工况条件参考**未实现**；
- 可靠性证据已升级为 4 维可观测特征（样本数 / 早期稳定性 / 当前偏离 / 近期趋势），
  通过 4→16→1 的门控混合参考残差与原始观测；
- `Arec` 仅完成 FD002 端到端 smoke（2 epochs，val 18.98），**没有进入正式网格**。

因此：**RACDR 目前只是原型，不是结果**。ORDER_02 要求的"参考可靠性正交消融"
未完成，不能声称其有效。

---

## 6. 创新冲突（初判）

需要警惕：工况条件化归一化、健康参考残差、传感器身份嵌入、可靠性门控在
PHM/测量文献中均有先例（如按工况聚类的归一化、unit-wise baseline 校正）。
当前差异点若能成立，只能是"**参考可信度作为显式可学习门控、并给出其可观测证据**"，
而不是"用了参考残差"本身。**在完成 A3/条件参考之前，不能主张创新。**

---

## 7. 风险与未完成项

| 项 | 状态 |
|---|---|
| Aid 的 FD002 第二种子 | 未完成（1/2） |
| S2 的 FD002 | 未完成 |
| S3 独立通道投影 | 未运行 |
| S4 固定传感器编码 | 未运行 |
| S5 Transformer + 身份机制 | 未运行 |
| 条件健康参考（按工况） | 未实现 |
| RACDR 正式消融 | 未运行 |
| 40-60 epochs 预算 | 仅 24 epochs |
| Level B 独立评估 | 未启动（协议已可预登记） |
| FD003/FD004 | 仍封存 |

**R1**：所有结论基于 2 个模型种子、1 个划分种子；统计功效不足。
**R2**：tfA 的 late fraction 0.692 是安全隐患，未被任何门禁覆盖。
**R3**：Protocol C 在高 RUL 段（早期窗口）误差极大，pooled 值受样本构成影响，
需要用"同截断集合"配对比较，而不是直接比 pooled。

---

## 8. 复算

```bash
cd /home/yndx/zxk/bearing-rul-pi
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
python scripts/error_analysis.py --root experiments/order02 --out results/order02
python scripts/audit_stats.py \
  --a "experiments/order02/FD002/Aid_s*/pseudo_test_predictions.csv" \
  --b "experiments/order02/FD002/tf_s*/pseudo_test_predictions.csv"
```

---

## REQUEST FOR DIRECTOR DECISION

1. **方法定位**：数据显示 FD001 的收益属于**输入表征 + Transformer**（tfA），
   FD002 的收益属于**候选主干 + Sensor-ID**（Aid）。是否同意把论文主线定为
   "工况条件下的参考/身份表征"，而把主干结构降为可替换实现？
2. **安全门禁**：tfA 虽然 RMSE 最好，但 late fraction 0.692、severe-late 18 例。
   是否把"severe-late 比例"设为与方法 RMSE 并列的硬门禁（例如不得超过基线的 1.1 倍）？
3. **下一轮优先级**：Aid 的 FD002 第二种子与 S2 的 FD002 是当前最便宜、最能改变
   结论的两个实验（约 30 分钟）。是否先补这两个，再决定是否投入条件参考与 RACDR？
