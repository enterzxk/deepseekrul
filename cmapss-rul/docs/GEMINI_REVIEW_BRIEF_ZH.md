# C-MAPSS RUL 项目审阅简报（给 Gemini）

更新时间：2026-10-20

## 0. 请 Gemini 重点回答的问题

1. 在当前负向消融结果下，这个课题最合理、最有发表价值的科学问题是什么？
2. A-GCN 和 Multi-Scale SSM 是否应该继续保留为核心创新，还是应当删除/重构？
3. 官方 FD001 测试集已经在开发 pilot 中被观察过。后续怎样设计实验，才能最大程度恢复可辩护的独立测试协议？
4. 在不使用官方测试集调参的前提下，怎样把 FD001 的 RMSE 从当前约 14.9 提升到 11.5--12.5？
5. 物理非对称损失目前只在 NASA Score 上显示一致但有限的收益，怎样把它发展成真正有科学价值的 reliability/safety 机制？
6. 若目标仍是中科院二区期刊，最低限度需要补哪些外部基线、数据集和统计分析？
7. 当前模型是否有“模块堆叠”的审稿风险？应该怎样删减或重新定义贡献？
8. 三 seed 是否足够？发动机级划分、交叉验证和最终测试应如何组织？

---

## 1. 项目背景

研究对象为 NASA C-MAPSS 涡扇发动机剩余使用寿命（RUL）预测，目标期刊层次为中科院二区。

当前工作已经完成：

- FD001--FD004 数据预处理和 PyTorch Dataset
- 一个时空双分支模型 STG-MambaNet
- 物理引导非对称损失 PhysicsGuidedLoss
- 四变体、三 seed 的 FD001 消融实验
- 每个运行都保存配置、数据划分、权重、测试预测和指标

目前**没有任何外部论文基线复现结果**，也**没有 FD002--FD004 的结果**，因此还不能声称达到二区水平或 state-of-the-art。

---

## 2. 数据与标签

使用标准 C-MAPSS 原始文件：

- train_FD001.txt
- test_FD001.txt
- RUL_FD001.txt
- FD002--FD004 文件也已准备好，但当前只做了 FD001 消融

标准预处理：

- 输入只保留 14 个传感器：2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21
- 不使用 3 个 operating settings
- 训练集 Min-Max 拟合，验证集和测试集只做 transform
- 训练标签：min(125, last_cycle - current_cycle)
- 训练窗口：window_size=30，stride=1
- 测试输入：每台发动机最后 30 个周期
- 测试短序列：左侧复制首帧填充
- 测试标签：直接使用官方 RUL 文件，不截断
- 指标：RMSE、NASA 非对称分数、MAE、平均有符号误差、晚报比例、严重晚报比例

FD001 真实数据规模：

- 训练发动机：100
- 测试发动机：100
- 训练窗口：17,731
- 测试窗口：100
- 测试 RUL 范围：7--145

输入张量形状：

~~~text
(Batch_Size, 30, 14)
~~~

---

## 3. 当前模型 STG-MambaNet

总参数量：144,264（默认配置）。

### 3.1 自适应空间图卷积分支 A-GCN

可学习传感器节点嵌入：

~~~text
A = Softmax(ReLU(E1 @ E2^T), dim=-1)
~~~

其中 E1、E2 的形状为 14 x 16，A 为 14 x 14 行随机矩阵。

每一层图卷积：

~~~text
H_agg = A @ H
H_out = LayerNorm(GELU(Linear(H_agg)))
H_out = H + H_out
~~~

默认使用 2 层图卷积。

空间分支最后使用可学习的传感器节点注意力，把 14 个节点池化为一个时间片特征。

### 3.2 多尺度选择性状态空间时序分支

三个并行尺度，DWConv1d kernel 分别为 3、5、7。

每个尺度内部：

- Linear 投影生成 value 和 SiLU gate
- Depthwise Conv1d
- Linear 投影生成 input-dependent delta、B、C
- 对角负状态转移的选择性递推
- SiLU 输出门控
- 残差和 LayerNorm

选择性状态更新：

~~~text
state_t = exp(delta_t * A) * state_{t-1} + B_t * x_t
y_t = C_t * state_t
~~~

其中 A 为逐通道的负对角状态参数。

三个尺度的输出通过样本级 softmax 权重融合。

### 3.3 交叉注意力与门控融合

- 时间特征作为 Query
- 空间特征作为 Key/Value
- MultiheadAttention 交叉注意力
- 空间、时间、交叉特征通过样本级 softmax 权重融合
- 再通过 channel-wise sigmoid gate 调节融合结果

### 3.4 回归头

同时使用最后一个时间步特征和全窗口平均特征：

~~~text
prediction = MLP([H_last, mean(H)])
~~~

输出形状为 (B, 1)。

### 3.5 物理引导损失

~~~text
L = mse_weight * MSE
    + asymmetric_weight * mean(w * (pred - true)^2)

w = 2, 当 pred > true   （晚报）
w = 1, 当 pred <= true  （早报）
~~~

默认参数：

- mse_weight = 1.0
- asymmetric_weight = 1.0
- late_weight = 2.0
- early_weight = 1.0

也就是说，非对称项中晚报平方误差权重是早报的 2 倍，同时保留标准 MSE。

---

## 4. 消融实验协议

四个变体：

1. w/o A-GCN：去掉图传播，保留独立的逐传感器编码器和传感器池化
2. w/o SSM-Block：把完整多尺度 SSM 换成 2 层 LSTM
3. w/o Physics Loss：只把损失换成标准 MSE
4. Full Model：全部模块

训练协议：

- 官方训练发动机按发动机级别划分 80% train / 20% validation
- 每个 seed 使用相同划分和 DataLoader 顺序
- Min-Max 只在训练发动机上拟合
- AdamW
- learning rate = 1e-3
- weight decay = 1e-4
- batch size = 256
- cosine scheduler
- max epochs = 80
- early stopping patience = 20
- 梯度裁剪 = 1.0
- 按 validation RMSE 选择 best checkpoint
- 官方测试集只在训练结束后评估
- seeds：42、67、123

注意：当前早期 pilot 已经观察过官方 FD001 测试集。后续如果根据这些测试分数修改模型，会构成测试集泄漏。论文最终实验必须重新冻结设计并一次性评估独立测试集。

---

## 5. 真实结果

### 5.1 三 seed 均值

| Variant | RMSE | NASA Score | MAE | Params |
|---|---:|---:|---:|---:|
| w/o A-GCN | 15.048 +/- 0.308 | 347.370 +/- 41.614 | 11.423 +/- 0.376 | 143,816 |
| w/o SSM-Block (LSTM) | 14.756 +/- 0.304 | 383.466 +/- 60.829 | 11.007 +/- 0.056 | 126,981 |
| w/o Physics Loss | 15.308 +/- 0.442 | 450.918 +/- 101.568 | 11.248 +/- 0.254 | 144,264 |
| Full Model | 14.878 +/- 0.712 | 379.771 +/- 81.421 | 11.107 +/- 0.790 | 144,264 |

Full 的测试 RMSE 范围为 14.072--15.422，没有达到目标 11.5--12.5。

### 5.2 逐 seed RMSE

| Variant | seed 42 | seed 67 | seed 123 |
|---|---:|---:|---:|
| w/o A-GCN | 15.120 | 14.711 | 15.314 |
| w/o SSM-Block | 14.464 | 15.071 | 14.732 |
| w/o Physics Loss | 15.268 | 15.769 | 14.887 |
| Full Model | 15.422 | 15.140 | 14.072 |

### 5.3 相对 Full 的配对差值

正值表示被消融变体更差。

| Ablation | 平均 delta RMSE | 平均 delta NASA | 观察 |
|---|---:|---:|---|
| w/o A-GCN | +0.170 | -32.401 | RMSE 方向混合；NASA 反而略优 |
| w/o SSM-Block | -0.122 | +3.695 | LSTM 平均 RMSE 更好 |
| w/o Physics Loss | +0.430 | +71.147 | 3/3 seeds 的 NASA 都变差 |

---

## 6. 当前能支持与不能支持的结论

### 可以谨慎支持的结论

- 物理非对称损失在当前设定下，对 NASA Score 有一致方向的正向作用
- 去掉物理损失后，3/3 seeds 的 NASA Score 都变差
- 平均来看，RMSE 从 15.308 降到 14.878，NASA Score 从 450.918 降到 379.771

但“大幅降低危险晚报”仍然过强，当前 NASA 相对改善约 16%，不是数量级变化。

### 不能支持的结论

- 不能说 A-GCN 去掉后 RMSE 明显变差
- 不能说 SSM 明显优于 LSTM
- 不能说 Full Model 全场最佳
- 不能说已经达到 11.5--12.5 的二区水平
- 不能把当前结果和其他论文的公开分数直接比较，因为：
  - 没有复现外部基线
  - 训练/验证 split 不同
  - 没有统一 early stopping 和模型选择协议
  - 只做了 FD001

### 结果中的主要矛盾

1. A-GCN 对 RMSE 的平均收益很小，而且不同 seed 方向不一致。
2. LSTM 的平均 RMSE 甚至略好于 SSM。
3. 物理损失虽有一致收益，但模型选择是按 validation RMSE 做的，而不是按 NASA Score。
4. Full 的方差较大，说明对初始化/发动机划分敏感。
5. 验证集和测试集分布可能不一致，导致 validation RMSE 与 test RMSE 排名不完全一致。

---

## 7. 当前最需要解决的问题

### 问题 A：基础模型性能不足

当前 Full 约 14.9 RMSE，离目标 11.5--12.5 有较大差距。

可能原因：

- 只使用 80% 的训练发动机
- 模型对 30 长度窗口的长期退化建模不足
- 回归头和融合结构可能过复杂
- 没有 target scaling
- 没有使用 operating settings 或工况归一化
- 模型选择只看 validation RMSE
- 数据量/参数量/正则化不匹配

### 问题 B：创新模块没有通过自身消融

如果 A-GCN 和 SSM 都不能稳定优于简单控制组，审稿人会认为模块是凑数的。

需要 Gemini 判断：

- 是继续优化这两个模块，还是删除其中一个？
- 是否应该把论文重点从“新网络结构”转向“可靠性与非对称风险控制”？
- 物理损失是否足以成为核心贡献？

### 问题 C：测试集暴露

官方 FD001 测试集已经在 12 次运行中被观察。

需要在文档层面明确：

- 这些结果是 development pilot，不是最终论文表格
- 后续模型设计只能使用 validation 信息
- 最终需要冻结模型、超参数和 epoch 预算，再评估官方测试集
- 或者建立额外的训练发动机内部 outer test，用于开发阶段

### 问题 D：缺少公平外部基线

目前没有：

- CNN
- LSTM
- GRU
- Transformer
- 图神经网络基线
- 经典 C-MAPSS 方法
- 近年强基线

没有这些，无法证明方法有竞争力。

---

## 8. 可选后续路线

### 路线 1：先提升基础性能，再谈创新

- 只用 validation 做模型改进
- 尝试 d_model=128、target scaling、更强正则化、不同窗口长度
- 精简融合模块
- 冻结设计后再重跑消融

风险：可能最终发现 LSTM 或更简单模型已经足够。

### 路线 2：围绕物理损失重构论文

- 弱化 A-GCN/SSM 的结构创新叙事
- 把重点放在“面向安全维护的非对称 RUL 风险控制”
- 研究 late prediction、严重晚报、NASA Score、校准和不确定性
- 需要证明该机制跨 FD001--FD004、跨模型、跨工况有效

风险：单一损失函数创新可能不足以支撑二区论文。

### 路线 3：删除无效模块，重新设计真正的科学问题

- 放弃当前“双分支+门控”的堆叠式结构
- 从失败样本分析出发，研究工况变化、退化阶段和晚报机制
- 用诊断证据决定新模块

风险：需要更多时间，但科学可信度更高。

### 路线 4：以可靠性/泛化为核心

- 研究跨工况 FD002/FD004
- 研究预测不确定性、拒绝预测、风险敏感评价
- 把物理损失作为 reliability 机制的一部分

风险：需要更多实验设计和数据协议工作。

---

## 9. 请 Gemini 给出的具体建议

请 Gemini 不要只给泛泛的“加注意力/加 Transformer”建议，而是明确回答：

1. 当前负向消融下，最诚实的论文故事是什么？
2. 哪些模块应该保留，哪些应该删除？
3. 怎样设计一个不使用官方 FD001 测试集调参、又能快速迭代的验证协议？
4. 怎样在 validation 上系统提升基础模型，而不是随机试超参数？
5. 物理损失应如何重新定义或加权，才能既降低晚报又不过度牺牲 RMSE？
6. 模型选择应使用 validation RMSE、validation NASA Score，还是复合目标？
7. 三 seed、发动机级 split 是否足够？是否需要 5-fold engine cross-validation？
8. 二区论文最低限度的外部基线、跨数据集实验和统计分析是什么？
9. 是否应该放弃“所有模块都有益”的叙事，改为报告真实的混合结果和失败分析？
10. 下一周最值得执行的三个实验是什么？

---

## 10. 代码与结果位置

服务器项目根目录：

~~~text
/home/yndx/zxk/bearing-rul-pi
~~~

主要文件：

~~~text
src/dataset.py
src/model.py
src/loss.py
scripts/run_ablation.py
scripts/summarize_ablations.py
docs/ABLATION_PROTOCOL.md
results/ablations_FD001_seeds42_67_123/summary.md
results/ablations_FD001_seeds42_67_123/per_seed_metrics.csv
results/ablations_FD001_seeds42_67_123/paired_deltas_vs_full.csv
results/ablations_FD001_seeds42_67_123/REPORT.md
~~~

复现消融：

~~~bash
python scripts/run_ablation.py \
  --subset FD001 \
  --seeds 42 67 123 \
  --variants wo_agcn wo_ssm wo_physics full \
  --epochs 80 --patience 20 --batch-size 256 \
  --num-workers 4 --device cuda
~~~

汇总：

~~~bash
python scripts/summarize_ablations.py \
  experiments/ablations/RUN_A \
  experiments/ablations/RUN_B \
  --output-dir results/ablations_FD001
~~~

---

## 11. 一句话总结

当前项目已经完成可复现的数据管线、模型、损失和三 seed 消融，但真实结果没有证明 A-GCN 和 SSM 的有效性，Full Model 也不是最佳；物理损失是唯一方向较一致的模块，但效果有限。现在需要 Gemini 帮助判断：是继续优化现有结构，还是重构论文问题与核心贡献。
