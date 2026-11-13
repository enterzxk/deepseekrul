# 独立实验审计与最终冻结前验收

三条Codex子代理完成来源、数学和实验流程的交叉审查。源码作者与审核者分开。
不调用DeepSeek Harness。各审计结论按任务对话留存；模型型号以实际载体为准，不伪造CLI日志。

已修复的阻断：balanced OT线性全局池化退化、ridge与投影混杂、70/75航班历史不一致、
resume未验封印、H权重提前detach、W-only有效参数及跳过无用谱诊断的报告口径。
数学最终范围：共同支持OLS、共同range与相对谱方向权重；非物理HI、非噪声置信、非一般坐标不变。

最终PREFLIGHT_V4：实际67项PASS，0 optimizer update、无TEST读取；执行SHA
485b3d98ec14629359e41d4cf09fd24af1a03aac18a3f99067b32385cd222f17。
含45项混合batch/query窗口外W/Xs/both扰动不变、5种生产FP32梯度、W-only无sensor旁路、
封印篡改拒绝、12组CPU FP32/FP64 fast/full输出及所有梯度/RNG逐位一致。
V2 preflight因a40缺h5py入口import失败，未执行检查或训练；改prepare惰性import后V3/V4通过。
性能优化审计子代理最后遭遇额度限制；其已落地改动由主控执行上述精确等价与GPU检查验收。
未伪称该失败调用返回了独立结论。先前独立数学/流程审查仍有效，后续数值检查是主控实测。

正式训练尚未启动。正式收益/机制门均待真实开发实验，不得由软件PASS推断。

2026-11-13 启动前硬件预算修订（L1，0次正式训练）：GPU现有其他任务负载升高，V4实测含25%余量预测26.95小时，超过自主设定24小时；扩为既有a40共享GPU36小时，不新增付费、不更改方法/seed/训练passes/数据/评价门，不操作其他任务。重新实测验收后才launch。原e00fef3源及V4未通过admission记录保留，不称训练结果。

最终启动证据：PREFLIGHT_V5真实67项PASS；CALIBRATION_V5实测含25%余量预计29.71小时，36小时共享GPU预算内。执行0d5ab920d31ec11767bb38f9ed717ce21f31ece3c6d981de7453cbd672d08009。尚未launch正式训练。
