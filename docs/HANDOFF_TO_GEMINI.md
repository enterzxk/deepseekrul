# STG-MambaNet 项目交接文档（给 Gemini 的下一步审查与决策）

日期：2026-10-20  
工作区：C:\Users\20562\Desktop\deepseek-rul  
远端：lab-server:/home/yndx/zxk/bearing-rul-pi

## 0. 一句话结论

数据管线、模型/损失代码和 FD001 三 seed 消融 pilot 都已经存在，但**当前结果不支持“Full Model 全面最优”的论文叙事**；同时项目里有两套模型/损失实现、测试 RUL 是否截断到 125 的协议冲突、以及官方测试集已经被多轮观察。Gemini 下一步不应直接写论文或继续扫测试集，而应先做**协议冻结、实现统一、验证集开发、参数匹配的消融重做**。

---

## 1. 当前资产

### 1.1 主要文件

| 路径 | 作用 | 状态 |
|---|---|---|
| dataset.py | 根目录数据管线，FD001–FD004 | 已通过真实数据测试 |
| src/dataset.py | 与根目录同哈希的数据管线 | 生产代码，被 run_ablation.py 使用 |
| src/model.py | 生产版 STG-MambaNet，含 graph/independent、SSM/LSTM、cross-attention、diagnostics | 被 12 次 pilot 使用；20 项测试通过 |
| src/loss.py | 生产版 PhysicsGuidedLoss：MSE + 非对称晚报惩罚 | 被 pilot 使用；**没有单调性项** |
| model.py（根目录） | 更贴近任务书公式的新版：use_gcn/use_ssm、轨迹输出、monotonic loss | 仅做过 4 配置 sanity、真实批次前向/反向、3 epoch 冒烟；**未进入 pilot** |
| loss.py（根目录） | PhysicsGuidedRULLoss：MSE + 0.5*Asym + 0.1*Mono | 同上；**未进入 pilot** |
| sanity_check.py | 根目录新版模型/损失的 4 配置自检 | GPU/CPU 均通过 |
| scripts/run_ablation.py | 生产消融 runner：engine-level split、验证集选 checkpoint、测试集最后评估一次 | 已产生 12 次运行 |
| scripts/summarize_ablations.py | 多种子聚合、paired delta | 已使用 |
| results/ablations_FD001_seeds42_67_123/ | 三 seed 消融结果与 REPORT.md | 已有完整产物 |
| docs/ABLATION_PROTOCOL.md | 消融协议 | 已冻结，但需与最终协议对齐 |
| docs/STEP1_VALIDATION.md | 数据验证记录 | 内容基本有效，但需注意测试 RUL 协议 |
| docs/STEP2_VALIDATION.md | 模型/损失验证记录 | **部分哈希已过期**，测试数写 19，当前实际 20 |

### 1.2 两套实现的区别（最重要）

**A. 生产版 src/model.py + src/loss.py（pilot 用的就是这套）**

- 构造函数参数：num_sensors, window_size, d_model, spatial_mode=graph/independent, temporal_mode=ssm/lstm, kernels=(3,5,7), num_heads, ...
- 空间分支：AdaptiveSpatialBranch 或 IndependentSpatialBranch
- 时间分支：MultiScaleStateSpaceTemporalBlock 或 LSTMTemporalBranch
- 融合：CrossAttentionGatedFusion（三路 branch gate + channel gate）
- 输出：final state + mean state 拼接后回归
- 损失：PhysicsGuidedLoss(mse_weight, asymmetric_weight, late_weight=2, early_weight=1)
- 没有逐周期轨迹，没有 L_Mono
- 默认参数量：Full 144,264；w/o A-GCN 143,816；w/o SSM 126,981

**B. 根目录新版 model.py + loss.py（更贴任务书，但未验证）**

- 构造函数：STG_MambaNet(num_sensors=14, seq_len=30, d_model=64, use_gcn=True, use_ssm=True)
- AdaptiveSpatialGCN：A = softmax(ReLU(E1 E2^T))，get_adjacency_matrix()
- TemporalSSMBlock：DWConv k=3 + 膨胀 1/2 双尺度 + 纯 PyTorch selective diagonal SSM + SiLU gate + residual + LayerNorm
- 轨迹：forward(x, return_trajectory=True) 返回 (B,1) 与 (B,L)
- 损失：PhysicsGuidedRULLoss(use_physics, gamma=2.0, lambda_asym=0.5, lambda_mono=0.1)
- 单调项：monotonic_violation(trajectory, lengths)
- 默认参数量：146,627
- 已验证：4 配置前向/反向、邻接矩阵行和、E1 梯度、真实 FD001 批次、CPU/GPU
- 未验证：未接入 run_ablation.py；未参加三 seed pilot；正式精度未知

**结论：不要把两套实现的指标混在一起。当前所有 12 次 pilot 数字只属于 A，不属于 B。**

### 1.3 数据协议（已实现）

- 数据集：NASA C-MAPSS FD001–FD004
- 传感器：标准 14 个：2,3,4,7,8,9,11,12,13,14,15,17,20,21
- 归一化：Min-Max，统计量只由训练发动机拟合，验证/测试只 transform
- 训练标签：min(125, last_cycle - cycle)
- 训练窗口：长度 30，stride 1
- 测试输入：每台发动机最后一个 30 周期窗口；不足 30 左侧复制首帧
- 指标：RMSE、NASA asymmetric score（d = pred - true；d<0 用 exp(-d/13)-1，d>=0 用 exp(d/10)-1，求和）
- 测试真值：dataset.py 默认 clip_test_rul=False，即使用 RUL_FDxxx.txt 原值；run_ablation.py 的 build_test_windows 也没有截断
- 任务书原始要求：训练与测试真值都截断到 125
- **这是当前最大的协议冲突之一，必须由 Gemini 先裁决并写死。**

### 1.4 FD001 三 seed pilot（生产版 A，原始测试标签）

运行：4 变体 x seeds 42/67/123，共 12 次；engine-level 80/20 split；验证 RMSE 选 checkpoint；官方测试集最后评估一次。

| Variant | RMSE mean ± std | NASA mean ± std | MAE mean ± std | Params |
|---|---:|---:|---:|---:|
| w/o A-GCN | 15.048 ± 0.308 | 347.370 ± 41.614 | 11.423 ± 0.376 | 143,816 |
| w/o SSM-Block (LSTM) | 14.756 ± 0.304 | 383.466 ± 60.829 | 11.007 ± 0.056 | 126,981 |
| w/o Physics Loss | 15.308 ± 0.442 | 450.918 ± 101.568 | 11.248 ± 0.254 | 144,264 |
| Full Model | 14.878 ± 0.712 | 379.771 ± 81.421 | 11.107 ± 0.790 | 144,264 |

Full 的逐 seed test RMSE：15.422 / 15.140 / 14.072。

观察：

- Full 不是 RMSE 最佳：LSTM 对照平均低 0.122。
- Full 不是 NASA 最佳：w/o A-GCN 平均低 32.4。
- 物理损失是唯一方向稳定的模块：去掉后 3/3 seeds 的 NASA 都变差，平均 RMSE +0.430、NASA +71.1；但只有 3 seeds，不能称“大幅”。
- A-GCN 与 SSM 的收益方向不稳定，不能按预设想定写“两个模块都有效”。

### 1.5 仅做标签截断重算（不重训）

用已保存的 test_predictions.csv，把 y_true 截断到 125 后重新计算（这是协议检查，不是新模型结果）：

| Variant | RMSE mean ± std (clip125) | NASA mean ± std (clip125) |
|---|---:|---:|
| w/o A-GCN | 13.833 ± 0.477 | 312.941 ± 46.393 |
| w/o SSM-Block | 13.506 ± 0.461 | 347.653 ± 70.564 |
| w/o Physics Loss | 14.329 ± 0.417 | 423.103 ± 100.506 |
| Full Model | 13.655 ± 0.711 | 342.600 ± 80.749 |

Full 逐 seed clip125 RMSE：14.147 / 13.978 / 12.839；NASA：368.6 / 407.2 / 252.1。

关键含义：

- 仅截断标签就把 Full 平均 RMSE 从 14.878 降到 13.655，说明**协议选择直接影响结论，必须冻结**。
- 截断后 Full 仍不是平均最佳：LSTM 的 RMSE 更低（13.506），w/o A-GCN 的 NASA 更低（312.9）。
- 只有 seed 123 的 Full 在 clip125 下达标（12.839 / 252.1），seeds 42/67 明显更差，seed 方差是当前主要风险。
- 原任务书的 <=13.0 / <=280 门禁，按均值仍未通过。

### 1.6 测试与溯源

- 当前远端单元测试：20 项全部通过（Python 3.11, torch 2.5.1+cu121）。
- 文档写 19 项，已过期。
- docs/STEP2_VALIDATION.md 中 src/model.py 的哈希 4f8ab... 与当前 D0592... 不一致；src/loss.py 哈希仍匹配。
- 当前关键哈希（前 12 位）：
  - dataset.py / src/dataset.py：BA32AA502FD9
  - src/model.py：D0592FACE6B1
  - src/loss.py：058D275505ED
  - scripts/run_ablation.py：F06D61334289
  - 根目录 model.py：2C2F81F65665
  - 根目录 loss.py：1A909F5F9A10

---

## 2. 必须让 Gemini 先处理的风险

### R1. 两套实现没有统一

生产版 A 有 pilot 和测试，但没有单调性；新版 B 有单调性和轨迹，但没有 pilot。继续并行开发会出现“论文里写的是 B，表格来自 A”的不可复现问题。

### R2. 测试 RUL 截断协议冲突

任务书写“训练和测试都截断 125”；当前生产 runner 用原始测试标签。两种协议的 RMSE/NASA 不可直接比较。必须二选一，并在 config、summary、论文表格里标注。

### R3. 官方测试集已经被观察

12 次 pilot 的 test 分数已经被用于报告和讨论。之后如果继续用这些分数选模型/调权重，就是测试集泄漏。后续应：
- 只用验证发动机开发；
- 设计冻结后，用新 seeds 做最终训练；
- 官方 test 只在最终版本评估一次，或明确声明这是 development pilot。

### R4. Full 不是最佳，不能强行改叙事

原任务书 Phase 4 写“若 Full 不是最佳，就调权重/结构直到最佳”。这在方法论上是用测试集反向选择，审稿风险极高。正确做法是：
- 在验证集上做有限、预注册的调参；
- 接受负结果，或换研究问题/数据集；
- 不能用 test 结果反复调整直到“故事成立”。

### R5. 消融对照不一定公平

- w/o SSM 的 LSTM 是 2 层，参数量 126,981；Full 是 144,264。LSTM 反而 RMSE 更低，不能证明 SSM 有效。
- w/o A-GCN 的 independent 分支仍有 per-sensor MLP + attention pooling，不是“无空间建模”，只是去掉图传播。
- 需要参数匹配、训练预算匹配、同 seed/同 split 的对照；最好再加一个“随机固定图/单位矩阵图”对照，区分“图结构”与“额外参数”的贡献。

### R6. 物理损失的两个版本不一致

生产版只有非对称项，没有 L_Mono；新版有 L_Mono，但未进入 pilot。论文若声称“非对称 + 单调退化约束”，必须把 L_Mono 接入生产版并做消融：MSE / +Asym / +Mono / full。

### R7. FD001 可能不适合证明图模块

FD001 是单一工况、单一故障模式；传感器图在 FD002/FD004 的多工况、多故障下更可能有用。只凭 FD001 否定 A-GCN 也为时过早。建议把 FD002/FD004 作为外部有效性检验，但先冻结 FD001 开发协议，避免边看 test 边改。

### R8. 文档状态不一致

README 还写 “Step 1 complete”；PROJECT_PROGRESS 同时写“尚无训练结果”和“已有三 seed pilot”；STEP2 哈希/测试数过期。Gemini 接手前应先统一状态文档。

---

## 3. 建议的下一步顺序

### P0：先冻结，不训练

1. **裁决测试标签协议**：clip125 还是 raw。建议以任务书为准选 clip125，但同时报告 raw 作为附录；全项目只用一个主协议。
2. **统一实现**：选 A 或 B 作为唯一生产代码。建议优先 A，因为已有 runner/tests/pilot；把 B 的 L_Mono/轨迹作为增量移植到 A，而不是另起炉灶。
3. **封存 test**：在新的 protocol 文档里写“test 只用于最终评估”，并把当前 12 次结果标记为 development pilot。
4. **更新文档**：README、PROJECT_PROGRESS、STEP2_VALIDATION 的哈希/测试数/协议。

### P1：只用验证集提升基础模型

1. 固定 engine-level split（每个 seed 固定 20% 验证发动机）。
2. 在验证集上做有限网格：
   - dropout 0.1 / 0.2 / 0.3
   - weight decay 1e-4 / 1e-2
   - d_model 48 / 64
   - temporal layers 1 / 2
   - 输入高斯噪声 0.01 / 0.02（可选）
3. 目标不是刷 test，而是把验证 RMSE 的 seed 方差降下来；当前 Full test std 0.71（raw）/0.71（clip），seeds 42/67 是短板。
4. 训练策略：early stopping patience 15–20、grad clip 1.0、cosine、checkpoint 按 validation RMSE。
5. 记录 calibration：mean bias、late fraction、severe late fraction；不要只看 RMSE。

### P2：把物理损失做成真正可消融的模块

1. 将根目录 loss.py 的 monotonic_violation 移植到 src/loss.py，保持旧 API 向后兼容：
   - PhysicsGuidedLoss(..., use_physics=True, lambda_mono=0.1)
   - forward(..., y_pred_traj=None, lengths=None)
2. 模型增加 return_trajectory=True，不破坏默认 (B,1)。
3. 消融四组：
   - MSE
   - MSE + Asym
   - MSE + Mono
   - MSE + Asym + Mono（full）
4. 在验证集上调 gamma / lambda_asym / lambda_mono，不要用 test。
5. 报告 NASA、RMSE、MAE、mean bias、late fraction；物理损失的价值应主要体现在安全侧指标，而不一定 RMSE。

### P3：重做公平消融

1. 参数匹配：
   - LSTM 对照调整层数/宽度，使参数量接近 Full；
   - 或报告 params 并做参数量回归/敏感性分析。
2. 加对照：
   - identity graph（A=I）
   - fixed random graph
   - uniform graph
   以区分“可学习图”与“任何图传播”。
3. seeds 至少 5 个（建议 42/67/123/2026/3407），报告 mean±std、paired delta、bootstrap CI。
4. 先 FD001；若图模块仍无收益，再在 FD002/FD004 上做外部验证，而不是反复改 FD001 test。

### P4：冻结后出最终表/图

1. 设计冻结，写 docs/FINAL_PROTOCOL.md。
2. 用新 seeds 重训；官方 test 只评估一次。
3. 生成：
   - 排序 RUL 预测图
   - 典型发动机生命周期图
   - 14x14 邻接矩阵热图
   - 消融柱状图
4. 表格必须来自 metrics.json / predictions.csv，不允许手填期望值。

---

## 4. 请 Gemini 明确回答的问题

1. 主协议到底用 clip125 还是 raw test RUL？为什么？是否两者都报告？
2. 生产实现选 A（src，已有 pilot）还是 B（根目录，公式更贴任务书）？迁移成本多大？
3. 论文的核心贡献是否仍然是“A-GCN + SSM + Physics Loss”三者并列？如果 FD001 不支持 A-GCN/SSM，是否改为“Physics-guided safety loss 为主，图/SSM 为架构探索”？
4. 是否接受“负结果/混合结果”作为论文的一部分？如果不接受，准备换哪个数据集/研究问题，而不是改 test？
5. LSTM 对照是否需要参数匹配？如何匹配才算公平？
6. 物理损失的主指标是什么：RMSE、NASA、late fraction、severe late fraction，还是组合？门禁阈值是否仍为 13.0 / 280？
7. 是否把 FD002/FD004 纳入下一轮？如果是，计算预算和协议怎么定？
8. 官方 test 已经看过 12 次，下一轮如何定义“最终评估”才可信？是否需要新的 held-out seeds 或额外数据集？

---

## 5. 可直接发给 Gemini 的审查 Prompt

~~~text
你是这个 C-MAPSS RUL 项目的 methodologist / senior reviewer。请先读：
- docs/HANDOFF_TO_GEMINI.md
- results/ablations_FD001_seeds42_67_123/REPORT.md
- docs/ABLATION_PROTOCOL.md
- src/model.py, src/loss.py, scripts/run_ablation.py

现状：三 seed FD001 pilot 显示 Full Model 不是最佳；w/o SSM 的 LSTM 平均 RMSE 更低，w/o A-GCN 的 NASA 更低；只有 physics loss 的方向在 3/3 seeds 上稳定。测试 RUL 是否截断 125 还没有冻结，官方 test 已经被观察多轮。

请你不要直接写论文，也不要用 test 分数调模型。请输出：
1) 一份协议裁决：clip125 vs raw，主指标，验证/测试边界；
2) 一份实现统一方案：保留 src 还是迁移根目录新版，如何无损接入 L_Mono；
3) 一份 validation-only 的实验计划（模型、正则、seed、预算、停止条件）；
4) 一份公平消融设计（参数匹配、identity/random graph 对照、统计检验）；
5) 一个明确的 go/no-go：当前证据是否还支持把 A-GCN 和 SSM 作为核心创新；如果不支持，给出替代论文叙事或替代数据集方案。
所有建议必须区分“已验证事实”“待验证假设”“个人判断”。
~~~

---

## 6. 常用命令与环境

远端项目：

~~~bash
cd /home/yndx/zxk/bearing-rul-pi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate math_modeling   # Python 3.11, torch 2.5.1+cu121
python -m unittest discover -s tests -v
~~~

单次消融：

~~~bash
python scripts/run_ablation.py \
  --subset FD001 \
  --seeds 42 67 123 \
  --variants wo_agcn wo_ssm wo_physics full \
  --epochs 80 --patience 20 --batch-size 256 \
  --num-workers 4 --device cuda
~~~

注意：上面的命令会写新的 run 目录；不要覆盖已有 pilot 结果。

根目录新版自检：

~~~bash
python sanity_check.py --device cuda
~~~

---

## 7. 绝对不要做

- 不要用官方 test 分数选择 epoch、超参数、损失权重或模型结构。
- 不要把 raw 与 clip125 的 RMSE/NASA 混在同一张表。
- 不要把根目录新版 B 的冒烟结果当成生产版 A 的 pilot 结果。
- 不要因为 Full 不是最佳就反复调 test 直到它最佳。
- 不要在没有参数匹配的情况下宣称 SSM 优于 LSTM。
- 不要在没有 L_Mono 消融的情况下宣称“单调物理约束有效”。
- 不要手填论文表格；所有数字必须来自 metrics.json / predictions.csv。
- 不要把 development pilot 写成 confirmatory final test。

---

## 8. 交接后的第一项具体动作（建议）

让 Gemini 先只做一件事：**写 docs/FINAL_PROTOCOL.md 的 v0 草案**，内容仅包含：
- 主协议（clip/raw）
- 唯一生产实现（A 或 B）
- 验证集划分与 seeds
- 主指标与安全指标
- 最终 test 只评估一次的定义
- Full 不最优时的处理规则（接受/换数据集/换叙事，禁止 test 调参）

这份草案通过后，再开始训练。不要先跑模型再补协议。
