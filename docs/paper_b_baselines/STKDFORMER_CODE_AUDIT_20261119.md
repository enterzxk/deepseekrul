# ST-KDFormer 作者 sample code 复现审计

> 日期：2026-11-19。对象：`ST-KDFormer_Sample_Code.zip` +
> `Enhanced_Remaining_Useful_Life_Prediction_for_Turbofan_Engines_Using_Spatiotemporal_Koopman_Dual-Branch_Transformer.pdf`
> （IEEE TIM 2025, DOI 10.1109/TIM.2025.3625335）。结论：**部分可复现；当前 zip 不能端到端跑。**

## 1. zip 里有什么

| 文件 | 状态 |
|---|---|
| `data_util.py` | 完整：`process_targets`、`process_input_data_with_targets`、`process_test_data`、`score_function`、`LossEMA`、`lambda_schedule`、`spectral_radius_tensor` |
| `ST-KDFormer_code.ipynb` | 数据管线、训练循环、测试/绘图、加载 checkpoint 的代码；**模型类不在 notebook 里** |
| `FD001/002/003/004_ST-KDFormer_revision.pt` | 4 个完整 pickled 模型（不是 state_dict） |
| `__MACOSX/` | macOS 元数据，忽略 |

## 2. 缺什么：两个模型源文件

notebook 第 0 个 cell：

```python
from data_util import *
from model_util import *
from transformer_model2 import *
```

zip 里**没有 `model_util.py`，也没有 `transformer_model2.py`**。

checkpoint 的 pickle 引用：

- `transformer_model2.dualTransformer_TS`
- `transformer_model2.Koopman_Test0` / `Koopman_Test1`
- `transformer_model2.LSTMKoopmanEncoder` / `LSTMDecoder` / `KoopmanDynamics`
- `model_util.Encoder` / `EncoderLayer` / `Norm` / `MultiHeadAttention` /
  `FeedForward` / `OrthoProjector` / `OutConvNet_spatiotemperal` / `PositionalEncoding`

直接 `torch.load` 报：

```
ModuleNotFoundError: No module named 'model_util'
```

我们用空 `nn.Module` stub 把两个模块名补齐后，**可以加载 state_dict**（说明 checkpoint 本身完整），
但没有 `forward`，不能跑推理/训练。stub 只能验证权重形状，不是复现。

## 3. 已经复现/验证的部分

### 3.1 数据管线：FD001 完全对上

在 3090 上用作者的 `data_util.py` + notebook 的 `CMAPSS_dataloader` 跑 FD001：

```
Train engines: 80
Validation engines: 20
Train windows: 13533
Validation windows: 3798
```

与 notebook 保存的输出**完全一致**。说明数据读取、特征选择、窗口化、train/val 划分这部分是完整可跑的。

### 3.2 checkpoint 元数据：与论文配置一致

用 stub 加载 4 个 checkpoint：

| 子集 | 参数量 | 时间分支输入 | 空间分支输入（窗口） |
|---|---:|---:|---:|
| FD001 | 1,697,133 | 14 | 30 |
| FD002 | 1,904,646 | 19 | 50 |
| FD003 | 1,697,133 | 14 | 30 |
| FD004 | 1,820,156 | 19 | 40 |

与论文 §IV-B/Table III 的“14/19 sensors、window 30/50/30/40、1.70M params”一致；
FD002/FD004 因输入维度更大而参数略多。

### 3.3 FD001 的已保存输出与论文 Table IV 一致

notebook cell 8 的保存输出：

```
RMSE 10.560733872511083 | SCORE 157.12120783959784
```

论文 Table IV 的 ST-KDFormer FD001：**RMSE 10.56 / Score 157.12**。  
这说明 checkpoint 是论文 FD001 的模型；但这是**作者环境里的保存输出**，不是我们在本机独立跑出来的。

## 4. 论文 Table IV 与我们的同协议结果

| 子集 | ST-KDFormer（论文，作者协议） | DSPT-V0（我们的匹配协议） | 备注 |
|---|---:|---:|---|
| FD001 | 10.56 / 157.12 | 10.60 | 基本持平 |
| FD002 | 13.20 / 844.85 | 11.03 | 作者协议下明显更差 |
| FD003 | 10.36 / 172.10 | 10.68 | 作者协议下更好 |
| FD004 | 15.55 / 1125.73 | 12.38 | 作者协议下明显更差 |

**不能直接写“我们更好/他们更好”**：两边协议不同（见 §5）。要公平比较，必须把 ST-KDFormer
放到我们的 matched protocol 下重训，而不是拿论文数字对表。

## 5. 作者协议 vs 我们的 matched protocol

| 项目 | ST-KDFormer sample code | 我们的 Paper B 匹配协议 |
|---|---|---|
| train/val 划分 | 随机 80/20，`random_state=23` | `split_seed=7, fold=0`，固定 engine 列表 |
| 归一化 | **MinMaxScaler 在全部 training 数据上 fit（含 validation engines）** | condition-wise z-score，只 fit training engines |
| 特征 | FD001/003 14 个；FD002/004 19 个 | 统一 14 个传感器 |
| 窗口 | 30/50/30/40；step=5；输入 (B,5,W,F) | span=70；K=3 因果窗口（LTAG）或单窗口 |
| 训练 | 100 epoch，batch 128，AdamW lr=1e-3，MultiStepLR [10,20,30,40] | 40 epoch，batch 512，patience 12 |
| 验证 | 随机 20% training engines，pooled RMSE | engine-level mean RMSE，5 cuts |
| test | 每机最后一个窗口（30/50/30/40），短序列尾部零填充 | 每机末周期一个窗口，左填充 |
| 指标 | RMSE + NASA score | engine-level RMSE |

**注意**：作者的 MinMaxScaler 在划分前对全部 training 数据 fit，validation engines 参与了
归一化统计；这是 validation 选择上的泄漏。test 的 scaler 也只 fit training engines，所以 test
不是标签泄漏，但 early stopping 的 validation 数字会偏乐观。复现时要修正这一点。

## 6. 结论与下一步

**结论**：当前 zip **不能端到端复现**；数据管线可复现（FD001 计数完全一致），checkpoint 可加载
并确认与论文配置一致，FD001 的保存输出与论文 Table IV 一致，但**模型 forward/训练代码缺失**。

**下一步（按优先级）**：

1. **向作者要 `model_util.py` 和 `transformer_model2.py`**（最小修复）。拿到后：
   - 先跑 FD001 notebook，确认本机能得到 10.56/157.12；
   - 再改 `window_length`/特征数跑 FD002/003/004，核对 Table IV；
   - 然后决定是否加入 matched comparison。
2. 如果作者不给：只能从论文 + checkpoint keys **重实现** ST-KDFormer；这是“复现方法”，
   不是“跑作者代码”，工作量和误差都更大。
3. 若要做公平比较：把 ST-KDFormer 的模型接到我们的数据管线（split_seed=7、14 传感器、
   condition-wise z-score、40 epoch），重训 4 子集 × 3 seed；不能直接拿论文数字对表。
4. 作者的 validation 归一化泄漏要在复现时修掉；否则 FD001 10.56 的 validation 选择不可比。
