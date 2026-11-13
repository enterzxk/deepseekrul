# N-CMAPSS公开论文基线核查

日期2026-11-13。数据是实际飞行工况驱动的仿真失效，非实机失效记录。
[NASA官方记录](https://ntrs.nasa.gov/citations/20205001125)；
[数据原始论文，10.3390/data6010005](https://www.mdpi.com/2306-5729/6/1/5)。
W4、实测Xs14可作输入；Xv虚拟量、T健康参数、hs健康标签不得混入主比较。
Fc及duration不作默认输入；cycle仅作为公开年龄。预测时刻统一flight_end。

| 论文 | DOI与固定作者代码 | 核查结论/可比性 |
|---|---|---|
| A Benchmark on Uncertainty Quantification for Deep Learning Prognostics, RESS2025 | [10.1016/j.ress.2024.110513](https://doi.org/10.1016/j.ress.2024.110513)，[BayesRUL@2943f4d85a1bb71910c100c956b32c3f6b595ceb](https://github.com/lbasora/bayesrul/tree/2943f4d85a1bb71910c100c956b32c3f6b595ceb) | 首选Inception-HNN。原DS01–05联合训练、18维W+Xs、随机窗口验证、hs平台标签；原联合RMSE不能当DS02分数。须另做单子集/发动机验证/统一标签适配。 |
| Aging-Informed Dynamic Graph Network for Aero-Engine RUL Prediction, TII2026 | [10.1109/TII.2026.3694202](https://doi.org/10.1109/TII.2026.3694202)，[AIDGN@d1ac173b13659af1854cf500a9b1cba8367353fe](https://github.com/TCCofWANG/AIDGN-RUL/tree/d1ac173b13659af1854cf500a9b1cba8367353fe) | 作者有NC管线及DS02 checkpoint。原含T40/P30虚拟输入、合并dev/test后由maxRUL设cap、随机窗验证；原协议独立标注，统一适配不得保留这些访问。 |
| Multi-objective Optimization of Extreme Learning Machine for Remaining Useful Life Prediction, EvoApplications2022 | [10.1007/978-3-031-02462-7_13](https://doi.org/10.1007/978-3-031-02462-7_13)，[MOO_ELM@b19b2bf47bdfc3b7d4ae2a540b605caf4fd087e3](https://github.com/mohyunho/MOO_ELM/tree/b19b2bf47bdfc3b7d4ae2a540b605caf4fd087e3) | 明确DS02-006，完整ELM/NSGA-II搜索。原输入含两虚拟量、随机窗验证；不能用任意小ELM冒充完整方法，需保留搜索预算与目标。 |
| Interpretable Prognostics with Concept Bottleneck Models, Information Fusion2025 | [10.1016/j.inffus.2025.103427](https://doi.org/10.1016/j.inffus.2025.103427)，[Concept@25ec477c834e81440a288e161ec1f4461737ee49](https://github.com/EPFL-IMOS/concept-prognostics/tree/25ec477c834e81440a288e161ec1f4461737ee49) | 原DS01/04/05/07等而非DS02。普通CNN可作为独立论文来源；CBM/CEM用真实健康概念监督，须列特权标签条件。原按航班平均窗口预测，与单末窗结果不等。 |
| Performance-based probabilistic RUL prediction via particle filtering with neural-guided inference, MSSP2026 | [10.1016/j.ymssp.2026.115026](https://doi.org/10.1016/j.ymssp.2026.115026)，[源码@6e7cc72ba0fc616a57ac37d41796f45a51e7d0e8](https://github.com/4belito/deep-performance-rul/tree/6e7cc72ba0fc616a57ac37d41796f45a51e7d0e8) | 参考/特权条件。使用SmFan/SmLPC/SmHPC虚拟稳定裕度、Fc/整航班pos及hs健康参考；删掉核心变量不能称完整方法。 |
| Two-Phase Sequence-Feature Attention, TIM2025 | [10.1109/TIM.2025.3576953](https://doi.org/10.1109/TIM.2025.3576953)，[源码@5bd33b2b1bb044ab4be1a8cb67fdf8a5d96d1d0a](https://github.com/mjliu-advertising/RUL/tree/5bd33b2b1bb044ab4be1a8cb67fdf8a5d96d1d0a) | 论文包含N-CMAPSS，公开固定代码仅发现CMAPSS管线；NC忠实复现证据不完整，不能把旧C-MAPSS结果移入新表。 |

前四库309个作者原文件已在新包upstream保全，记录Git blob SHA1及SHA256；作者源码
未修改、未导入或训练。未核实整库许可的三个snapshot仅本地保留，受Git忽略保护；
提交其manifest及获取链接。MOO保留Apache与第三方GPL/LGPL声明。详见
`paper_b/ncmapss_response/upstream/PRESERVATION_ACCEPTANCE.json`及逐库SOURCE_MANIFEST.json。

目前前四篇复现状态均 **PARTIAL：DOI/官方来源/实现/管线已核，统一训练未执行**。
后两篇 **PARTIAL：仅作来源/额外条件核查**。本页不宣称任何NC论文已复现成功，
不把作者公开历史数字与本研究统一协议混表。

统一重训适配前须完成：完整核心/原损失保留审计，W+Xs+年龄同信息规则，发动机
验证、训练侧预处理、共同RUL与NASA定义、航班末预测规则、窗口范围和搜索预算。
尤其Bayes完整Inception固定18输入、输出loc/scale均softplus，不能偷偷改线性loc或
用简化CNN；年龄接入需明确外部适配，不得称逐字节原模型复现。
