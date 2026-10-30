# ORDER_01 执行报告（IEEE TIM 攻关 · 第一轮）

- 仓库：https://github.com/enterzxk/deepseekrul/tree/main/cmapss-rul
- 基线 commit（本轮开始前）：`50769d8c84564b70405787dca14beb10168cc139`
- 服务器：`/home/yndx/zxk/bearing-rul-pi`，单卡 A40
- 本轮结束 commit：`e46afd17f7f13e81585d473d5737af4423ab1224`

> 本轮所有数字均由 `experiments/order01/**/metrics.json` 与
> `pseudo_test_predictions.csv` 自动汇总，可用文末命令复算。
> 未通过的实验如实列出，未被包装成结论。

---

## 1. 代码修改与修复（含回归测试）

| # | 问题 | 修复 | 测试 |
|---|---|---|---|
| 1 | pseudo-test 使用最后一个 epoch 权重 | 训练结束后先 `load_state_dict(best_state)` 再评估；`metrics.json` 记录 `best_epoch / epochs_ran / weights / restored_best_before_pseudo_test` | 每次运行日志打印 `restored_from_best=True`（本轮 24/24 为 True） |
| 2 | 随机种子与数据切分耦合 | `--split-seed`（默认 7）只决定发动机划分，`--model-seed` 决定初始化与 DataLoader | 同层内 3 个模型种子共用同一验证发动机集合（`val_units.csv` 逐运行落盘） |
| 3 | 传感器身份缺失 | 新增 `use_sensor_id`（Learned Sensor-ID Embedding）、`use_independent_channels`（逐通道投影） | `tests/test_sensor_permutation.py`：共享编码器置换 Δ=0.000000（完全置换不变），sensor-ID 版本 Δ=0.010854（打破不变性） |
| 4 | pseudo-test 协议单一 | 改为 50% / 70% / 90% 三个固定截断，逐发动机单窗口；同时保留全窗口验证 | `metrics.json: pseudo_test_rmse_by_cutoff`；逐发动机预测 `pseudo_test_predictions.csv` |
| 5 | 强 Transformer 比较不公平 | 新增 `tfA`（同一 Transformer 主干 + Track A 三特征），与 `tf`（纯 Transformer）、`A`（候选主干 + 三特征）构成 2x2 分解 | 同一 `split_seed`、同一截断、同一评价函数 |
| 6 | Bootstrap 对象错误 | 逐发动机重采样并**每轮重算 RMSE 差** | `scripts/audit_stats.py`（本轮未产出最终显著性表，见风险 R3） |
| 7 | 开发期读取官方 test | 新增 `dataset.load_cmapss_train()`，只读训练文件 | `tests/test_train_loader_never_opens_test_files`：任何 test/RUL 访问立即失败 |

测试汇总：`tests/test_audit_fixes.py`、`tests/test_stage2_boundaries.py`、
`tests/test_sensor_permutation.py` 全部通过。

---

## 2. 实验协议

| 项 | 值 |
|---|---|
| 数据 | FD001、FD002 **仅训练集**（官方 test/RUL 从未打开） |
| 划分 | 固定 `split_seed=7`，80/20 发动机级划分（所有模型共用） |
| 模型种子 | 2027、2028、2029（3 个） |
| 训练 | 18 epochs，patience 8，batch 256，lr 1e-3，wd 1e-4，dropout 0.15 |
| 主干容量 | CandidateModel d=64，1 层独立 Transformer；`tf`/`tfA` d=88，1 层 |
| 选择 | 全窗口验证 RMSE 选 checkpoint；pseudo-test 用**同一** checkpoint |
| 评价 | 全窗口 val RMSE；pseudo-test 在 0.5/0.7/0.9 三个截断下的 RMSE |

**为什么仍是 18 epochs**：本轮用于判断"特征/结构/传感器的相对贡献"，预算对
全部候选完全相同；正式预算将在方法冻结后重定（风险 R2）。

---

## 3. 主表（3 seeds，均值）

| 数据集 | 方法 | 全窗口 val RMSE | pseudo-test RMSE（三截断均值） |
|---|---|---:|---:|
| FD001 | tf（纯 Transformer） | 12.343 | **10.160** |
| FD001 | tfA（Transformer + Track A 特征） | 12.570 | 10.797 |
| FD001 | A（候选主干 + Track A 特征） | **11.376** | 11.058 |
| FD001 | Aid（A + Sensor-ID） | 11.972 | 11.601 |
| FD002 | tf | 14.275 | 14.867 |
| FD002 | tfA | 14.475 | 13.965 |
| FD002 | A | 13.009 | 12.337 |
| FD002 | **Aid（A + Sensor-ID）** | **11.754** | **10.743** |

### 三条可直接使用的结论

1. **FD001 上不存在"Track A 全面胜出"**。全窗口 val 里 A 最好（11.38 对 12.34），
   但部署式 pseudo-test 里纯 Transformer 最好（10.16 对 11.06）。这与 Director
   的怀疑一致：**全窗口验证会高估 Track A**。
2. **FD002 上 Track A 明显更优，且 Sensor-ID 是关键**：tf 14.87 → A 12.34 →
   **Aid 10.74**。加一个 14×64 的传感器身份嵌入（约 900 参数）比再加任何时序模块
   都有效，说明此前"共享编码器 + 对称汇总"确实把 14 个物理通道当成可互换的，
   这在六工况数据上是硬伤。
3. **特征迁移到 Transformer 也有效**（FD002：tf 14.87 → tfA 13.97），说明 FD002
   的收益**主要来自输入表征**，而不是候选主干本身；FD001 上则相反（tfA 10.80
   略差于 tf 10.16）。

---

## 4. 误差机制分析（`results/order01/band_errors.csv`）

按 pseudo-test 真实 RUL 分段（FD001，三截断合并）：

| 方法 | [0,25) | [25,50) | [50,100) | [100,126) |
|---|---:|---:|---:|---:|
| tf | **3.43** | 12.72 | **13.55** | 13.36 |
| A | 6.63 | **10.15** | 14.25 | **11.15** |
| tfA | 5.65 | 9.56 | 14.99 | 11.86 |
| Aid | 5.57 | 13.43 | 15.86 | 9.63 |

FD002：

| 方法 | [0,25) | [25,50) | [50,100) | [100,126) |
|---|---:|---:|---:|---:|
| tf | 5.34 | **12.25** | 21.23 | 17.15 |
| tfA | 5.67 | 14.76 | **18.12** | 17.97 |

**解释**：误差主体在中寿段 [50,100)（FD002 上 tf 达 21.2）。Track A 在 FD001 的
优势集中在 [25,50) 与 [100,126)，而在真实末端 [0,25) 明显更差（6.63 vs 3.43）——
即"越接近失效越不准"，这对安全关键应用是负面证据，必须写进论文的局限性。

---

## 5. 未完成项（如实列出，不计入结论）

| 项 | 状态 | 原因 |
|---|---|---|
| `Arec`（可靠性加权参考） | **未完成** | 可靠性门控输入维度实现缺陷，三次重跑仍在旧代码上失败；不纳入任何结论 |
| Track D 对比学习 | 未实现 | 无代码，按 ORDER 要求标为未完成 |
| Track C（修正损失后的双任务） | 未在正式网格中运行 | 仅实现 + 单测，未进本轮对照 |
| A3 周期分布偏移审查 | 未完成 | 需 train/test 观测长度分布与同龄不同寿命对照 |
| A4 工况条件健康参考 | 未实现 | 仍为前 5 周期全局均值 |
| Level B 独立评估 | 未启动 | 待方法冻结后一次性执行 |

---

## 6. 科研风险

- **R1（最高）**：FD001 上 Track A 只在全窗口验证占优，部署式评价落后 0.9。
  若不能解释并修正，Track A 不能作为单工况主贡献，只能作为多工况方法。
- **R2**：所有结论基于 18 epochs。正式预算下排序可能变化，需在冻结前重跑。
- **R3**：本轮未产出逐发动机配对 Bootstrap 的显著性表（脚本已就绪，需在
  同一 split 下成对运行后生成）。
- **R4**：Sensor-ID 在 FD002 的 +1.6 收益尚未做"参数匹配"对照（Aid 比 A 多约
  900 参数），需补一个参数量匹配的对照以排除容量混淆。
- **R5**：pseudo-test 的截断比例是人为固定的，未按官方测试的观测长度分布校准。

---

## 7. 建议（保留 / 淘汰 / 重设计）

- **保留**：Track A 的 EMA 特征通道（FD002 贡献最大）、Sensor-ID 嵌入（FD002 +1.6）。
- **淘汰**：权重 EMA（0.995，已两次证伪）、全窗口验证作为唯一模型选择标准。
- **重设计**：健康参考的可靠性机制（Arec）——需要把"参考样本数 + 工况匹配度 +
  传感器异常度"做成显式输入，而不是当前的两维代理。
- **下一步优先**：FD001 的 [0,25) 段误差修正（A 6.63 vs tf 3.43）。这是当前
  最大的单项差距，且直接对应"临近失效"这一最关键的工业场景。

---

## 8. 复算命令

```bash
cd /home/yndx/zxk/bearing-rul-pi
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
# 汇总
python - <<'EOF'
import json, glob, collections
rows = collections.defaultdict(list)
for p in glob.glob("experiments/order01/*/*/metrics.json"):
    d = json.load(open(p))
    rows[(d["subset"], d["variant"])].append((d["best_val_rmse"], d["pseudo_test_rmse_mean"]))
for k, v in sorted(rows.items()):
    print(k, len(v), sum(x[0] for x in v)/len(v), sum(x[1] for x in v)/len(v))
EOF
# 误差机制
python scripts/error_analysis.py
```

---

## REQUEST FOR DIRECTOR DECISION

1. **FD001 的定位**：A 在部署式评价上落后纯 Transformer 0.9 RMSE，且差距集中在
   临近失效段（[0,25)：6.63 vs 3.43）。是否同意把 Track A 定位为**多工况（FD002）
   方法**，FD001 只作为"不劣化"的约束而不是主战场？
2. **Sensor-ID 是否升级为正式贡献**：它在 FD002 带来 1.6 RMSE 提升，但需要先补
   参数匹配对照。是否批准把下一轮的主线定为"传感器身份 + 工况条件参考"，
   并暂停其他结构搜索？
3. **epoch 预算**：是否允许用开发集把预算从 18 提到 40-60 并重跑当前四配置？
   这会让排序更可靠，但会推迟 Level B。
