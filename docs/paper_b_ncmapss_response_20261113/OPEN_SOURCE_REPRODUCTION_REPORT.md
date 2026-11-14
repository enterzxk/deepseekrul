# Paper B：N-CMAPSS 公开基线与模块实验报告

日期：2026-11-14。**共同支撑谱加权模块：FAILED。完整AIDGN与BayesRUL-HNN统一开发协议执行：SUCCESS；作者原始协议数值复现：PARTIAL。**

本轮共888次真实神经训练：504模块消融＋两篇各192次（每篇120内层选窗＋72外层重训）。24个解析年龄参考另计。全部888个checkpoint已重载，2184组点预测、Bayes另336组sigma均与封印CSV逐位相同，最大误差0。未读取官方TEST，未改Paper A、冻结DSPT或旧实验。

旧C-MAPSS已完成结果保留在[独立2026-11-12报告](https://github.com/enterzxk/deepseekrul/blob/d7bd9c17829b988ad867a57ba3c2968bde25f2b2/docs/paper_b_baselines/MATCHED_COMPARISON_REPORT_20261112.md)。本报告不合并历史OOF、旧NC验证、作者历史分数或官方TEST。

## 相同信息下的模块消融

五个主臂raw/fe/delta/supported/common使用同一raw512同步点、最近70航班、相同Transformer头、同训练/外折query集合。W-only沿用该头但屏蔽Xs并禁用sensor辅助监督；age为只读当前年龄的MLP，age_prior为确定性年龄先验，两者不接收raw或相同历史头。每seed先平均六个heldout engine的RMSE，再报告三seed均值±样本SD；年龄先验为确定性估计，不制造seed。单位：航班周期。

|方法|DS02|DS05|DS06|DS07|
|---|---:|---:|---:|---:|
|raw|8.397 ± 1.015|8.819 ± 0.470|8.810 ± 0.224|9.029 ± 0.517|
|fe|9.406 ± 0.243|10.875 ± 0.314|9.822 ± 1.603|9.344 ± 0.098|
|delta|9.210 ± 0.654|8.104 ± 1.494|9.096 ± 0.785|7.469 ± 0.556|
|supported|10.640 ± 0.818|13.174 ± 1.797|10.782 ± 0.829|6.608 ± 0.627|
|common|9.855 ± 1.185|12.109 ± 1.242|11.078 ± 0.379|7.528 ± 0.725|
|age|9.164 ± 0.124|13.995 ± 0.361|11.203 ± 0.327|6.097 ± 0.113|
|w_only|9.775 ± 0.410|12.249 ± 0.301|10.244 ± 0.298|8.296 ± 0.290|
|age_prior|8.927（确定性）|13.593（确定性）|10.662（确定性）|5.896（确定性）|

五个预注册门 primary、age_references、sensor_specificity、attribution、mechanism **全部FAIL**。common相对raw在DS02/05/06分别恶化17.36%/37.31%/25.74%，仅DS07改善16.62%；DS07仍弱于delta、supported及年龄参考。不能将这个局部改善写成新模块成立。FE是已有函数编码对照，四子集也未胜raw。

共同rank均值约3.95/4，range投影作用约1e-7；谱方向相对变化约.65–.72，说明权重机制确实被改变，其收益与鲁棒性预测没有实现。载荷Gram不是纯工况激励或噪声精度，存在坐标gauge风险；不作物理HI识别主张。raw与FE辅助context路径也不同，不能把它们的全差值归因于系数尺度。

详细三视图、raw/cap125、配对发动机bootstrap区间与全部门见实验包summary。区间先在engine内对seed求均值再重采样6个engine；LOEO训练集重叠，仅作描述，不声称独立折显著性。

## 统一评价的完整公开模型

两篇均只用本次同一四子集、同dev发动机、同seed101/202/303、同cap125标签与flight-end预测规则。窗口只在训练机内部留一机验证后选，outer重新fit五机及三seed重训。以下每篇仅72个outer结果，120个inner成绩不进入表。raw是信息范围锚点；外部模型窗口与分辨率不同，表格不承担同信息模块因果归因。

|完整模型迁移|DS02|DS05|DS06|DS07|
|---|---:|---:|---:|---:|
|raw|8.397 ± 1.015|8.819 ± 0.470|8.810 ± 0.224|9.029 ± 0.517|
|AIDGN_full_migrated|9.027 ± 0.089|12.706 ± 0.449|10.625 ± 0.349|8.423 ± 0.186|
|BayesRUL_HNN_full_migrated|14.797 ± 3.362|14.741 ± 0.751|18.500 ± 6.998|10.778 ± 3.410|

BayesRUL在此固定200轮、小规模按机训练协议中有较大seed方差。保留所有seed与未上限截断的预测；这张表不证明其原论文结论失效，也不代表完成了全超参最优强度搜索。原Gaussian NLL与正mu/sigma头完整保留，没有为取得名次改成MSE。

NASA Score采用 e=prediction−truth，e≥0为exp(e/10)−1，e<0为exp(−e/13)−1。下表为每seed六机全部航班的Score总和，再对三seed求均值；NASA mean与SD、flight-pooled RMSE另见CSV，不与engine-equal RMSE混同。

|方法：NASA总和均值|DS02|DS05|DS06|DS07|
|---|---:|---:|---:|---:|
|raw|520.271|630.490|611.730|557.470|
|common|655.615|1067.255|934.393|439.881|
|AIDGN_full_migrated|587.720|1134.825|802.857|507.269|
|BayesRUL_HNN_full_migrated|5371.417|2833.519|11774.172|9404.827|

## 原作者协议与本次迁移分开判定

|论文|固定作者commit|原协议数值复现|统一NC执行|证据／限制|
|---|---|---|---|---|
|AIDGN-RUL，TII2026，10.1109/TII.2026.3694202|d1ac173b13659af1854cf500a9b1cba8367353fe|PARTIAL|SUCCESS|完整原网络、abs head、native defaults/MinMax保留；W+Xs18替代原含两virtual的20通道；cap125/125替代全label-derived cap；observed cycle/q95替代native1e-5；原随机窗/val自适应LR未迁移。192train、24inner选择、全192cp重载PASS。|
|BayesRUL-HNN，RESS2025，10.1016/j.ress.2024.110513|2943f4d85a1bb71910c100c956b32c3f6b595ceb|PARTIAL|SUCCESS|完整Inception/dense/双softplus、原weights_init与原HNN.step保留。单子集/按机/global训练侧scaler/正cap125/125/年龄metadata为显式迁移；未复现原DS01–05 joint/hs标签/随机窗/earlystop。192train及mu/sigma重载PASS。|
|MOO_ELM，EvoApplications2022，10.1007/978-3-031-02462-7_13|b19b2bf47bdfc3b7d4ae2a540b605caf4fd087e3|PARTIAL|PARTIAL|完整ELM+NSGA-II来源已固定，统一搜索未运行，不用任意小ELM冒充原方法。|
|Concept-Prognostics，Information Fusion2025，10.1016/j.inffus.2025.103427|25ec477c834e81440a288e161ec1f4461737ee49|PARTIAL|PARTIAL|原CBM/CEM依赖真实健康概念监督，须单列特权条件；不把其普通CNN对照当主方法。|
|H-CDE，RESS 2027卷期（2026公开记录），10.1016/j.ress.2026.112943|b5cfa1e069bd824ad65fc0ad427dd0e828ebf936|PARTIAL|PARTIAL|新增独立来源审查；实际为sensor重构/health对齐而非本协议RUL，健康筛选与划分/入口问题另记，不冒加head称原复现。|

原公开checkpoint路径、论文历史分数、原始协议RMSE/NASA差值未在本轮完成的项目，均不填伪造差值。DOI/官方来源和额外数据风险详见 NEW_DATASET_BASELINE_AUDIT.md 与 HCDE_SOURCE_AUDIT.md。

## 固定协议与窗口预算

四子集每6台dev机；按engine划分。各实验的缩放仅fit相应训练发动机：模块为outer五机，外部模型为inner四机或outer五机，官方TEST不参加方法、窗口、epoch选择。原始W4+Xs14、age可观察cycle；T/Xv/hs/Fc/unit-ID/duration不入模型。标签训练统一min(rawY,125)，原始航班周期RUL为主评价、cap125为敏感性列；预测只下界0，保留上界外推。

|实验|训练预算／候选|验证选择|
|---|---|---|
|模块七臂|固定70×512，无HPO；200passes/B64/AdamW .001/wd.0001；504train|无outer选轮或超参搜索|
|完整AIDGN|五窗口(1,50)/(1,512)/(20,16)/(70,4)/(70,512)，保留完整full70×512候选；200passes/B64/Adam，DS02 lr.0005，其余.001|每outer cyclic-next inner一机、train四机，seed101选最低inner rawRMSE，再outer五机三seed|
|完整BayesRUL-HNN|五窗口(1,30)/(1,50)/(1,512)/(20,16)/(70,4)，author T=H*P+1；200passes/B64，native Adam .001574/wd.001/Gaussian NLL|同一内外划分与五次选窗预算；不使用outer成绩选择|

Bayes固定18物理通道完整保留，前置一行[current age,1,0*16]非物理metadata；不称原生输入协议。原loss variance floor eps1e-6在归一坐标对应有效sigma .125周期，但CSV保存原预测sigma×125；没有校准保证主张。不同模型的可逆目标缩放用于保留各原正输出头，反变换后均为同一物理标签协议。

数据来自真实飞行工况驱动的仿真失效，不能称真实发动机失效实测。每航班512点为归一位置采样，不称逐秒等间隔。独立HDF5 schema核实实际Xs顺序为T24,T30,T48,T50,P15,P2,P21,P24,Ps30,P40,P50,Nf,Nc,Wf。旧manifest仅文字错名，真实directconcat未重排；旧逐sensor名称撤回，纠正保留于data_diagnostics_v2_schema_corrected。

## 真实训练与完整性证据

|实验|训练数|checkpoint重载|点预测向量|最大差|
|---|---:|---:|---:|---:|
|七臂模块|504|504 PASS|1512|0，全部bitwise|
|完整AIDGN|192|192 PASS|336|0，全部bitwise|
|完整BayesRUL-HNN|192|192 PASS|336，另336个sigma|0，均bitwise|

各格initial/final权重变化、200epochs、Adam实际step/moments、参数量、配置/源/数据/cache/checkpoint SHA及真实CSV已核。两篇collector独立重算train-only scaler、所有指标和24次窗口选择。完整逐格参数和时间在各collection/cells表；网络参数随窗口改变，不只报一个挑选的最小数。

模块共享GPU墙钟 **10.6105小时**，低于36小时上限；并发进程训练时间之和 **41.0094小时**另计，不能与36墙钟上限混比。AIDGN/Bayes各24共享GPU墙钟小时预算；其最终完成marker保存在原artifact包。所有时间受既有共享GPU争用影响，不据此排名模型速度。

训练producer冻结SHA：模块2e57e8ae68f194d73ba0a94fcd45437507f33d58 / execution0d5ab920d31ec11767bb38f9ed717ce21f31ece3c6d981de7453cbd672d08009；AIDGN b6f8a07dbacd16196335fc57676ecc6dbf31c5dd / 400306179c2b16df577eed02791e668c3c6d96436946102b021b0a3e5e8df8e6；Bayes 1cdfc7fb130e909877cab93067fd53b8cfc30318 / 3b76be258fc6829f4c54208d29c1a1131d44014119d754a78e8f971d752f53af。

本仓库 experiments/paper_b_ncmapss_response_20261113 保存轻量完整预测／配置／scaler／trace／metrics封印archive、逐文件manifest、collector表及全重载JSON。大checkpoint仅逐seed保存在现有私有a40原formal目录，当前没有公开权重下载，artifact manifest保留其SHA与原路径；不将原NASA观测或未获整库许可的作者源码上传Git。

## 负结果后的探索边界

D1完成120个训练机内部crossfit，残差std/rawstd中位仅约1.3–1.5%，尺度门MET；动态遗漏门NOT_MET，只有DS05达到预设10%。固定尺度没有改变年龄相关性。这提示预条件值得单独检验，不能证明残差是健康、也不能增加输入信息。

D2完成1120组零更新、固定训练query与四block的梯度审计。common末态共享response_body的weighted aux/task为.040/.052/.322/.233；没有支持aux>10的支配解释。完整mode/group/block值保留，不能把选定eval梯度当整个随机训练过程的因果解释。

下一轮只批准另行预注册的48次inner-only经典预处理×aux开关筛查，尚不认可新模块或创新性；其val分数不得混入上方outer表。H-CDE等最近邻已覆盖慢快状态与退化驱动。任何后续可学习模块须超越经典预处理强对照、通过机制门并另行审查。官方TEST在旧工作中曾开封，本次虽不访问，也不能叫全新盲测。

复跑入口、源码检查和原始artifact SHA见实验目录README；阶段决定见 POST_FAILURE_DECISION.md。
