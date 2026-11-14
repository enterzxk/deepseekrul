# Paper B 公开源码复现报告索引

C-MAPSS原任务已在独立分支完成并提交，详见[固定提交d7bd9c1的比较报告](https://github.com/enterzxk/deepseekrul/blob/d7bd9c17829b988ad867a57ba3c2968bde25f2b2/docs/paper_b_baselines/MATCHED_COMPARISON_REPORT_20261112.md)。其作者原协议与统一协议、TEST/OOF/validation口径保留原表，不与新数据集混合。

新N-CMAPSS DS02/05/06/07扩展见 [2026-11-14完整报告](../paper_b_ncmapss_response_20261113/OPEN_SOURCE_REPRODUCTION_REPORT.md)。本轮888次真实训练、888checkpoint全重载与2184点预测向量逐位一致；完整AIDGN/BayesRUL-HNN统一开发迁移执行SUCCESS、原作者数值复现PARTIAL；共同支撑谱加权模块五项门全FAIL，贡献判FAILED。MOO/Concept/H-CDE为PARTIAL，未虚构其新NC结果。官方TEST不参与本轮。

实验代码、真实预测、指标、配置哈希、训练时间、参数量、源commit、封印及逐seedcheckpoint位置均列于独立Paper B实验目录与完整报告。Paper A和冻结DSPT未修改。
