# CCT 实验的文献与评价口径核查

核查日期 2026-11-06。CCT 使用固定的 FD002、FD004 训练侧筛选。每个留出发动机的末端循环从 50 起全部计入，RUL 标签上限为 125。RMSE 对发动机内平均平方误差再做等权汇总，NASA 按发动机内及发动机间两次取平均。下列论文数字属于各自的官方测试协议，单列作背景。

| 方法及发表状态 | 作者报告的 RMSE，FD001 / FD002 / FD003 / FD004 | 作者报告的 Score，同序 | 核查到的协议 |
|---|---|---|---|
| DA-RUL，Results in Engineering，2026 | 11.52 / 12.59 / 11.21 / 12.96 | 183.90 / 630.70 / 245.47 / 785.94 | 公开 FD002 测试入口默认 W=40、cap125、逐工况 z-score、末窗口评分。加载器对短轨迹做前端重复补齐，并使用训练和测试传感器共同拟合归一化，特征集合也有差异。 |
| TCFT-BED，作者预印本 2511.04723，2025 | 11.53 / 13.65 / 9.31 / 14.24 | 261.26 / 1137.32 / 161.28 / 1209.45 | cap125，传感器按单调性和相关性选择。每台测试发动机按可用历史分配窗口，覆盖全部发动机。FD002 用 21/75/125，FD004 用 19/75/125。式 (23) 的指数符号与文字对高估惩罚的描述不一致，Score 按作者原表记录。 |
| MTFGN-SRL，作者预印本 2609.25179，2026-09 | 10.39 / 11.39 / 10.19 / 14.10 | 160 / 562 / 180 / 874 | 训练和测试均 cap125，14 项有效特征，按历史长度分组并通过样本关系图融合。FD002 用 20/40/60/80，FD004 用 18/40/62/84。Score 在测试样本上求和。 |
| Liu 等，IEEE TIM，2025，序列与特征注意力 | 公开页面与代码未核实到可引用的四子集数值 | 同左 | 公开实现先用配对子集预训练再微调，FD002 配 FD004、FD001 配 FD003。微调入口 cap125、W=45、末窗口评分，短轨迹前端补齐。公开加载器的归一化集合含训练与测试数据，预训练数据范围也与 CCT 不同。 |

DA-RUL 的数字来自作者仓库中的表 3，参数和处理取自公开测试入口及数据加载器。[作者表 3](https://github.com/TCCofWANG/DA-RUL/blob/main/pics/result.png) · [FD002 测试入口](https://github.com/TCCofWANG/DA-RUL/blob/main/test_script/FD002_TEST.py#L67) · [数据加载器](https://github.com/TCCofWANG/DA-RUL/blob/main/dataset/CMAPSSDataset.py#L271) · [出版社页面](https://www.sciencedirect.com/science/article/pii/S259012302504856X)

TCFT-BED 的窗口分配、式 (23) 和原报结果可在 §3.2–3.5 核查。[作者全文](https://arxiv.org/html/2511.04723v1)

MTFGN-SRL 的特征、cap、分组推断及原报结果可在 §IV-B、表 II 和表 III 核查。[作者全文](https://arxiv.org/html/2609.25179v1)

IEEE TIM 方法的配对子集预训练与微调处理来自作者实现。[出版记录](https://doi.org/10.1109/TIM.2025.3576953) · [预训练数据](https://github.com/mjliu-advertising/RUL/blob/main/dataset/CMAPSS/CMAPSSDataset_pre.py#L207) · [微调入口](https://github.com/mjliu-advertising/RUL/blob/main/Transformer_ft_serialatten.py#L44) · [微调归一化](https://github.com/mjliu-advertising/RUL/blob/main/dataset/CMAPSS/CMAPSSDataset_ft.py#L252)

TCFT-BED 和 MTFGN-SRL 已使用可用历史长度来分配推断窗口。CCT 的实验主张落在工况内趋势估计及其学习权重，分别由纯长窗、窗口调度与固定趋势进行对照。只有使用相同发动机、相同查询时点和相同标签的结果用于本轮增益判定。
