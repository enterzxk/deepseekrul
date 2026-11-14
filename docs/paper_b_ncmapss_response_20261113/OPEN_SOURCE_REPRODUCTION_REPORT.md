# Paper B — N-CMAPSS来源核查与统一开发实验

状态：**PARTIAL，正式训练进行中**。此页不宣称新模块有效或四篇论文已复现成功。

既有C-MAPSS公开基线的已完成统一比较见
[2026-11-12报告](https://github.com/enterzxk/deepseekrul/blob/d7bd9c17829b988ad867a57ba3c2968bde25f2b2/docs/paper_b_baselines/MATCHED_COMPARISON_REPORT_20261112.md)。
本页是独立的N-CMAPSS扩展，不合并其历史OOF、旧NC验证或官方TEST数字。

## 已实际完成的工作

| 项目 | 判定 | 证据 |
|---|---|---|
| DS02/05/06/07数据与变量边界核查 | SUCCESS | data_manifests/、DEV_METADATA_VALIDATION.json；共1880个dev航班，每子集6台发动机 |
| 原HDF5和dev缓存完整性 | SUCCESS | 四文件重新计算SHA256；缓存逐字节SHA，保留采样源索引；仅解码A_dev前两列、W_dev、X_s_dev、Y_dev |
| 四库作者源码固定与保全 | SUCCESS（保全范围） | 309个作者原文件、逐文件Git blob/SHA256，排除清单与许可说明；不等于训练复现成功 |
| 新候选实现和信息边界 | SUCCESS（软件验收） | PREFLIGHT_V5真实67项PASS，含FP32梯度、严格70航班、无sensor旁路、封印恢复、fast/full精确等价 |
| 首批训练真实性独立审计 | PARTIAL | 33格快照/330封印文件验哈希；每子集深入抽查一格，完整200passes/实际1200或1400 optimizer steps |
| 初步checkpoint重载 | PARTIAL | 4个DS02 raw checkpoint，各3视图预测逐位一致，最大差0；不是全部504格审计 |
| 新模块净收益与机制 | PARTIAL | 等待全部预注册对照；软件PASS不能推导收益或新颖性 |
| 冻结DSPT上的最终模块适配 | PARTIAL | 本次先检验隔离响应编码候选，尚不作“冻结DSPT+模块已提升”结论 |

数据为真实飞行工况驱动的**仿真失效**。RUL单位是航班周期，当前预测时刻是完整航班结束。
没有把1Hz航班内时间当RUL单位；512点重采样后也不称逐秒等间隔数据。
Xv、T、hs、Fc、unit-ID、飞行持续时间不作为预测输入。cycle为公开可观察年龄。
文件字节SHA会扫描容器内的test字节，但不解码或查看任何test数组。

## 新论文与复现状态

完整DOI、固定commit、输入/标签/验证风险见 [NEW_DATASET_BASELINE_AUDIT.md](NEW_DATASET_BASELINE_AUDIT.md)。

| 论文/方法 | 当前判定 | 具体边界 |
|---|---|---|
| BayesRUL / Inception-HNN，RESS2025 | PARTIAL | 官方完整模型、损失、训练配置及适配器静态审计已核；统一192次迁移正在做0optimizer硬件验收，尚未训练。原DS01–05联合结果不是DS02分数 |
| AIDGN-RUL，TII2026 | PARTIAL | 完整作者网络统一迁移已通过真实V3验收并启动192次训练（120inner+72outer）；尚未全部完成，不能填最终分数。原实现两虚拟输入及全体label-derived cap另标 |
| MOO_ELM，EvoApplications2022 | PARTIAL | 完整ELM/NSGA-II源码已保全；尚未执行统一搜索，不能用任意小ELM顶替 |
| Concept-Prognostics，Information Fusion2025 | PARTIAL | 原作者代码完整；普通CNN是该论文的对照，不冒充其主CEM/CBM。概念监督模型属于额外健康标签条件 |

原作者公开checkpoint是否可加载、作者原始NC协议下数值差以及统一协议作者模型数值，
均未在本轮完成，不填虚构分数。公开历史结果不与本次DEVELOPMENT_LOEO直接排名。
Two-Phase论文虽报告NC，固定公开仓库尚未找到NC数据管线；也不将旧C-MAPSS训练称作NC复现。

## 已冻结的候选实验

训练源码：`2e57e8ae68f194d73ba0a94fcd45437507f33d58`。
五个生产文件执行SHA：`0d5ab920d31ec11767bb38f9ed717ce21f31ece3c6d981de7453cbd672d08009`。
实际执行源码archive SHA：`151ea967012a3644221ed379647b80bee1fae14aa6a0f24e0ac488fc74449ae8`。
实验根：`/media/sangfor/vdb/user/zxk/ncmapss_response_20261113/formal_2e57e8ae`。

四子集×六个发动机LOEO外折×seed101/202/303×七臂，**504次神经训练**：

- raw：同步W/Xs的神经编码；
- fe：共享条件观测函数和逐航班4维响应坐标；属于已有向量函数编码思想的RUL适配对照；
- delta：在fe上加入普通过去航班坐标差；
- supported：共同W支撑上的截断OLS差和共同range投影；
- common：在supported上只增加共同响应信息谱的相对方向权重；
- age：年龄MLP；
- w_only：工况与年龄参考，Xs全部置零，不使用sensor辅助监督。

另有24折确定性年龄先验，解析估计五台训练机平均寿命；不伪称三seed训练。
完整定义见 [HYPOTHESIS.md](HYPOTHESIS.md) 与 `paper_b/ncmapss_response/PROTOCOL.json`。

所有臂用同一observed-prefix信息范围、70航班上界、相同query集合和训练侧缩放。
严格按发动机分离；所有缩放量只fit当前五台训练机。固定200passes、batch64，
没有窗口或epoch搜索，因此不读取外层heldout曲线选轮。原始实测输入为每航班512个
W4/Xs14同步点；窗口内成员的lag参照越界时，同时关闭其innovation和coverage，避免75航班暗增信息。

训练label cap125，主评价raw flight-cycle RUL，cap125作为另列敏感性；预测只下界0，
不截断上界。NASA使用 `e=pred−true; exp(e/10)−1 (e>=0), exp(−e/13)−1 (e<0)`，
sum与mean分别保留。主RMSE先每engine计算再等权；flight-pooled数值另列。
按engine聚合/重采样，3seed不是18台独立发动机；六外折训练集合重叠，区间仅描述性。

每格保存initial/checkpoint/optimizer、scalers、config及哈希、源/cache/执行SHA、真实
逐航班预测、参数数目、时间、实际更新数和逐文件封印。五主臂总参数84075，结构
有效参数接近；W-only有效参数79937，明确作为信息参考，不伪称完全参数匹配。
时间为共享GPU争用下的进程时间，不能据此作模型速度排名。预算为单张共享GPU
36墙钟小时，成功格进程时间之和另报，不能重复计算并发时间。

## 独立审计快照

首批33个SUCCESS：DS02=9、DS05/06/07各8，共330个被封印文件哈希全部匹配。
深入抽查DS02_raw_u10_s101、DS05_raw_u1_s101、DS06_raw_u1_s101、DS07_raw_u1_s101：
200passes，实际optimizer步数1200/1400/1400/1400；全部参数84075，初始与最终张量
哈希不同且与checkpoint/metrics一致。五训练机scaler、四格三视图CSV标签/ID以及
raw/cap125 RMSE、MAE、NASA、bias独立重算差均为0。

上述独立审计没有新forward、训练或TEST读取。主控另外对4个checkpoint重载三视图，
预测逐位差0；不把这一主控检查冒称另一个独立模型评审。所有快照保留计数时点，
不把早先21格collector与稍后33格审计混称当前总进度。

## 尚不能作出的结论

共享神经basis、可微ridge和函数编码已有先例；Gram也不是纯W激励或校准噪声精度。
共同谱权重依赖所学坐标，trace归一化消除整体信息强弱。当前只有**候选机制**，
不是已认证的新模块贡献。必须通过净收益、相同support/OLS控制、重采样与同步破坏门，
然后另作新颖性判断；即使平均分数更好，也不能直接改称物理HI识别。

独立查新否决了另一个“between/within协方差相减”作为新方法的提案：其核心接近
cPCA/dPCA/MNF，且自相关噪声会伪造低频方向。该提案没有新增GPU实验，也没有修改
当前504格实验。

官方TEST曾在既有NC研究中开封。当前入口完全不读取它；将来的重复评价必须披露
历史开封，不能称全新的独立盲测。Paper A、冻结DSPT和旧实验文件均未改动。

## 收口路径

父守护自动等待四worker，确认全部预注册cell并运行独立证据collector，生成
`summary/DEVELOPMENT_LOEO.md`、CSV和机器gates。然后执行全部checkpoint预测重载，
将完整结果与本页的PARTIAL阶段分开保存，按实际门给出SUCCESS/PARTIAL/FAILED。
原作者协议、统一开发评价与官方TEST分别制表，不按早期结果换seed、窗口或模块。


## 2026-11-14 运行更新（有时点的快照）

UTC17:23，prototype166/504封印，AIDGN10/192封印，两父进程仍在运行。AIDGN源b6f8a07dbacd16196335fc57676ecc6dbf31c5dd，执行400306179c2b16df577eed02791e668c3c6d96436946102b021b0a3e5e8df8e6；窗口只按inner发动机选，outer72独立汇总。详见 AIDGN_UNIFIED_MIGRATION_PLAN.md。Bayes计划见 BAYES_UNIFIED_MIGRATION_PLAN.md，尚无正式训练。

传感器清单语义纠正：实际Xs顺序为T24,T30,T48,T50,P15,P2,P21,P24,Ps30,P40,P50,Nf,Nc,Wf；原cache directconcat与Bayes native完全相同，错误仅在旧manifest的input_columns文字。独立四原HDF5 schema证明见evidence/RAW_HDF5_SCHEMA_CORRECTION.json；输入bytes、冻结source和已有数学结果均不改。
