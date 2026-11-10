# 00 · 交接状态（上下文耗尽时的恢复点）

> 本文件由 DSH 在会话上下文接近上限时写就。**新会话从这里恢复，不要重做已完成部分。**

## 1. 当前状态矩阵（6 台开发轴承 × 4 模型）

| 模型 | 完成 | 缺失 |
|---|---|---|
| T0（CNN-GRU，= `bearing_tim3/C-Fixed`） | **17/17** | — |
| T1（快照 Token，20 tokens） | 4/6 | Lear_Bearing2_1 之外的两台待核 |
| T2（Patch Token，4 tokens） | 5/6 | 1 台 |
| T3（时频 Token，200 tokens） | 6/6（补跑后） | — |

**先跑一次清单核对再补跑**，不要凭本表：
```bash
ssh a40 "cd ~/deepseek-rul && ls experiments/bearing_tf/{T1,T2,T3}/*.json | sed 's#.*/##' | sort | uniq -c"
```

## 2. 三个待修问题（按用户第二阶段）

### 问题一：T3 频带范围 —— **已核实，代码正确，仅文档错误（已修）**

- `rfft` 对 320 点实数序列 → 161 个单边频点，覆盖 **0–1.6 kHz**（fs=3.2 kHz 的 Nyquist）；
- 5 个频带等分 1.6 kHz，无负频率、无空频带；
- **不含原始 8–12.8 kHz**；`06_MAIN_RESULTS.md` 的"0–3.2 kHz"已更正。
- 无需重训 T3。

### 问题二：T2 极端高误差 —— **未修**

现状：T2 在 5 台里全部垫底（ES RMSE 681–1456）。当前实现是
`nn.Linear(5*2*320, 64)` 直接展平原始波形——**没有 CNN 的局部归纳偏置**。

**待实现 `T2-CNN`**：每快照先用与 T1 相同的 `SnapshotEncoder` 编码成 64 维，
再把连续 5 个快照拼成 patch（`Linear(5*64, 64)`）→ 4 tokens → 同一 Transformer。
这能区分"Patch 压缩本身有问题"与"原始波形展平投影有问题"。

### 问题三：ES 与留出评价混淆 —— **需重生成主表**

- ES 轴承只用于 checkpoint 选择；`best_es_rmse` **不是**留出轴承表现。
- 留出轴承表现必须读 `experiments/bearing_tf/<model>/<bearing>.csv` 的 `err` 列。
- **`06_MAIN_RESULTS.md` 的表用的是 ES RMSE（代理指标），必须重生成。**

## 3. 下一步执行顺序（照做即可）

1. **补满 Stage 1**：6 台 × {T0,T1,T2,T2-CNN,T3}，缺哪补哪（约 10 次）。
2. **实现 T2-CNN**（约 30 行改动：在 `run_bearing_tf.py` 加 `arch="T2C"`）。
3. **重生成主表**：从 `<bearing>.csv` 的 `err` 算逐轴承 RMSE，宏平均；ES RMSE 只作训练日志。
4. **裁决骨干**（情况 A/B/C/D），再决定 PDEM 的骨干。
5. **PDEM 实现**（B0–B7）：因果创新量 `e_t = norm(z_t − pred(z_{<t}))`、
   有限预算 K 的事件记忆、写入/替换/读取规则、未来表示辅助损失 + 无记忆对照 B7。
6. Codex 审查；17 台扩展；XJTU 外部验证（数据未解压）。

## 4. 恢复命令

```bash
# 远端（a40）
ssh a40 "cd ~/deepseek-rul && bash ~/status.sh"          # 通用状态
ls experiments/bearing_tf experiments/bearing_tim3        # 结果
# 补跑单次
python scripts/run_bearing_tf.py --arch T1 --holdout Lear_Bearing2_1 --out experiments/bearing_tf
```

脚本清单：
- `scripts/bearing_data.py` — 严格加载（分隔符自适应、逐快照校验、失败即抛）
- `scripts/run_bearing2.py` — CNN-GRU 四臂（A/C/C-Fixed/C-MLP）
- `scripts/run_bearing3.py` — E0–E3 特征增强
- `scripts/run_bearing_tf.py` — T1/T2/T3 Transformer
- `scripts/audit_bearing_stats.py` — 设备级配对统计
- `scripts/spectral_device_stats.py` — 频域设备级统计

## 5. 已知约束（写论文时必须遵守）

1. 17 台已参与多轮方法选择 → **FEMTO 结果只能作开发证据**。
2. 加种子 ≠ 加设备；独立单位是轴承。
3. T3 的 token 数是 T1 的 10 倍，参数量约 2 倍（321k vs 163k）→ 比较必须并列报告。
4. 8 倍盒式平均不是理想低通（1.6 kHz 处 −9.7 dB），原始 8–12.8 kHz 已被抹掉。
5. E0–E3 已证明：把频带能量拼进模型无增量收益（E3 vs E0 +74%，p=0.055）。
   **PDEM 若只是"更花哨的频带输入"，会重复这个失败。**

## 6. 未完成（诚实清单）

- T2-CNN、主表重生成、Stage 1 补齐
- PDEM 核心机制与 B0–B7 消融
- 最近邻精读（MTEDAST / WeiCLTF / Swin-KAN / PatchTST / 事件记忆）
- XJTU-SY 解压与独立协议（4.27 GB 已下载，未验证）
- Codex 独立审查
- `02_CLOSEST_PRIOR_ART.md`、`03_PDEM_MATHEMATICS.md`、`04_MODEL_IMPLEMENTATION.md`、
  `07_MECHANISM_ABLATION.md`、`08_EXTERNAL_VALIDATION.md`、`09_CODEX_REVIEW.md`、`10_DECISION_ZH.md`
