# 完整AIDGN：N-CMAPSS统一协议迁移（训练前登记）

这不是作者原始NC数据协议的数值复现，也不添加我们的temporal head。
完整`Model/AIDGN.py`按固定SHA直接加载，保留aging graph、GCN、全部decoder和abs输出。
作者原文件不修改；适配器只处理数据形状、可观察cycle与device属性。

作者实际NC默认：hidden128、AATE16、fc512、feature-fc=feature_num×4、dropout0.1、
hop1、单层、Linear聚合。统一18物理输入对应feature-fc72。DS02学习率5e-4，
DS05/06/07为1e-3；Exp实际Adam wd0，而不是未使用的hparams字典wd1e-4。

重要接口审计：原输出abs≥0，不能用零均值z-score标签直接训练。统一标签仍为
`min(rawY,125)`，训练数值除固定125、预测乘125；不改变原head。输入raw18保留作者MinMax[-1,1]变换，仅按当前四/五台训练机拟合；OCC为观测cycle/train-q95，披露替代作者cycle×1e-5数值缩放。
原aging输入并非从RUL或未知engine寿命推算；原全体dev/test maxRUL cap问题另标。

窗口目录在看新模型outer结果前固定为 `(history flights, points per flight)`：
`(1,50)、(1,512)、(20,16)、(70,4)、(70,512)`。所有点来自相同512点dev缓存，
按归一化航班位置等距取样并保留末点；不把它称作原始0.1Hz采样。
最后选项提供完整70×512观察机会；其余选项允许较短/较稀输入。
这不是各方法实际输入信息量完全相同的消融，不能由外部排名归因模块。

每个外折仅使用其他五台dev机：sorted units中外层heldout的cyclic-next作为inner验证，
另外四台重fit全部scaler。seed101对五窗口各训练固定200passes、B64；仅据inner
flight-end raw RMSE选择窗口，平局按点数、history、points。然后在五台机上从头
训练三个预设seed101/202/303，固定200passes后才构造outer heldout预测。

四子集共120次inner训练、72次outer训练，总192次；inner/outer单独目录和表格。
只在outer72上与当前候选DEVELOPMENT_LOEO比较。每格保存完整原模型权重、optimizer、
config/source/cache哈希、逐航班预测、更新数、时间与参数量。不选测试或outer epoch。
没有跨子集预训练、虚拟输入、健康概念标签或集成。正式TEST入口不存在。

硬件检查先实际运行五种原模型输入的forward/backward，含B64全分辨率
长度**35840**，不截短。检查仅合成数据、0optimizer，不冒称作者复现结果。
预算24共享GPU墙钟小时、每进程12GiB；超出时如实处理，不能删窗口或换小GNN。

状态：**PARTIAL，源码与适配已实现，等待独立审计和实际硬件验收后启动**。

统一fixed200用常学习率；作者原patience10/factor0.3验证驱动LR和early stop单独标注，不让outer验证驱动其训练，也不冒称原recipe忠实复现。

冻结前独立审计修正：作者feature_range明确[-1,1]。常数/范围<10*float64eps列用handled range1，训练常数映射-1；heldout越界不clip。新增纯合成四/五机预处理隔离、端点、constant及正标签/反变换合同。V1仅证明原类和硬件，不作为修后预处理证据；需V2真实验收。当前AIDGN真实训练仍0。

实际RuntimeV3验收PASS，原始类API/state/eval输出一致、完整五窗口B1/B64梯度有限、原source未改、预处理四/五机隔离与[-1,1]/positive125合同通过。最大T35840，峰值与预算见evidence/AIDGN_PREFLIGHT_V3.json。V2失败是constant测试点正好落在+1端点，fixture改成已知归一化3而非放松不clip判据，生产变换未改。当前正式AIDGN训练尚未启动。
