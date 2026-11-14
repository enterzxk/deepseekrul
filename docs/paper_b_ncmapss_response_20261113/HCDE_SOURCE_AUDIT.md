# H-CDE：最近邻来源审查

结论：**PARTIAL（来源／静态审计），不进入当前 RUL 数值主表**。未执行作者脚本、训练、读取凭据或加载 checkpoint。

作者：[Zhao and Fink，RESS2026，10.1016/j.ress.2026.112943](https://infoscience.epfl.ch/entities/publication/5e208282-d8bf-4d09-9140-4874bf95b2a0)，[arXiv2509.00639v2](https://arxiv.org/abs/2509.00639v2)。官方代码固定 [b5cfa1e069bd824ad65fc0ad427dd0e828ebf936](https://github.com/EPFL-IMOS/hcde/tree/b5cfa1e069bd824ad65fc0ad427dd0e828ebf936)，51 个原文件完整保存在私有本地 clone，逐文件 SHA256 见 `evidence/HCDE_SOURCE_MANIFEST.json`；根 LICENSE 为 MIT。独立审查者为 Codex `audit_secondary_sources`，仅只读。

| 核查项 | 实际源码行为与可比性 |
|---|---|
| 主任务 | [NC config](https://github.com/EPFL-IMOS/hcde/blob/b5cfa1e069bd824ad65fc0ad427dd0e828ebf936/run/configs/ncmapss/hcde.yaml#L94) 设置 forecasting/regression/rollout；[dataset](https://github.com/EPFL-IMOS/hcde/blob/b5cfa1e069bd824ad65fc0ad427dd0e828ebf936/src/dataset/cmapss.py#L163) 的 target 为短窗 Xs14，非 flight-cycle RUL。不能加一个新 head 后冒称原论文 RUL 复现。 |
| 模型输入 | W4+Xs14；神经重构 loss 不使用真实 HPT health 值。不能误写为 health 直接进入输入或监督 loss。 |
| 额外健康量 | [task](https://github.com/EPFL-IMOS/hcde/blob/b5cfa1e069bd824ad65fc0ad427dd0e828ebf936/src/train/task.py#L63) 用真实 HPT_eff_mod 做潜变量健康对齐回归与相关评估；与我们的 raw-RUL/NASA 任务不同。 |
| 标签／时间边界 | dataset 用 `hs==0` 筛 test；切分前以整表最大寿命归一时间；scaler fit train+val。不能移植为严格 train-only/all-prefix 协议。 |
| 官方划分 | DS01 配置 train=[1,4,7]，test=[9]。若 CSV 保留 NASA 编号，unit7 属官方 TEST；但 CSV 与转换脚本未提供，编号映射未核实，故只记录明确的划分风险，不断言已证实泄漏。 |
| 未来工况 | fast reconstruction 窗口提供其 W 作为控制；这不等于未知未来任务下的 first-passage RUL。需明确 slow 观测截止和 fast 控制窗口的信息时刻。 |
| 完整核心 | 可学习 path transform、slow/fast CDE 分开积分与 sensor readout 均存在。`sigmoid(gamma*x)*tanh(x)` 是软单调偏置，负 x 时仍可负，不能称严格非负损伤增量。 |
| 工程阻断（静态） | dataset.type=cmapss 与 helper 接受的 sampling 类型不符；val_split=0 返回 None，而 trainer 仍无条件 eval。未实际执行，不填运行失败日志或虚构修复。 |
| 原训练 | 主 NC config：B256，Adam .005/wd1e-5，最多60轮，plateau .95/patience5，early-stop min20/patience10；H64/slow5/fast10/path10。原 GPU 卡时未核实。 |
| 依赖／checkpoint | README 要求 Python3.11，锁 torch2.2.0/numpy1.26.4、torchcde/torchdiffeq/torchsde 等；树内未见权重文件，README 未给权重下载链接。未核实外部 releases。 |

慢快状态、可学习退化驱动和健康响应重构已是明确最近邻。因此“工况解耦”“正损伤时钟”及简单条件残差化不能直接作为我们的新颖性主张。当前只完成 D1/D2 和已注册外部基线；H-CDE 如需 RUL 迁移，必须先另行给出目标、读出和信息边界规格。
