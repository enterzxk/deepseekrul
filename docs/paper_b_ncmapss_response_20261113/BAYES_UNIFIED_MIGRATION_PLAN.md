# 完整 BayesRUL Inception-HNN 统一迁移（训练前登记）

日期：2026-11-14。只用 DS02/05/06/07 dev；原论文协议状态 PARTIAL。本登记不宣称模块增益。
作者 commit `2943f4d85a1bb71910c100c956b32c3f6b595ceb`；DOI [10.1016/j.ress.2024.110513](https://doi.org/10.1016/j.ress.2024.110513)。

完整原 Inception 四分支、降维分支、Flatten、80*T→64 dense、双 softplus+Threshold 输出，原 weights_init 及完整 HNN.step Gaussian NLL 保留。作者文件不改。不以 MSE 替代 HNN；负 NLL 合法。真实 torchinfo1.8.0 wheel 的 SHA256 为 `2e911c2918603f945c26ff21a3a838d12709223dc4ccf243407bce8b6e897b46`，仅在独立任务 `_deps` 解包，不改系统环境，无伪依赖。

实际原始 HDF5 列为 W4=[alt,Mach,TRA,T2]，Xs14=[T24,T30,T48,T50,P15,P2,P21,P24,Ps30,P40,P50,Nf,Nc,Wf]；四子集均与作者 native 顺序相同。独立 schema 审计只读取 W_var/X_s_var。旧 cache input_columns 清单列义错误在 RAW_HDF5_SCHEMA_CORRECTION.json 追加纠正，真实 directconcat 与已冻结输入 bytes 不改、不重排。

原网络固定18物理通道；前置一行明确的非物理 metadata `[当前可观察cycle/train_q95,1,0*16]`，其余全部物理观测保留。此为公开披露的年龄接入迁移，无额外自定义时序头，不称逐字节作者原始输入协议。全局物理通道 mean/std 与年龄q95只fit当前inner四机/outer五机；替代作者按窗口位置标准化及随机窗口验证时含val的拟合。

五窗口 (历史航班数,每航班点数) 为 (1,30)/(1,50)/(1,512)/(20,16)/(70,4)，原dense随作者T=H*P+1增长，不删dense/换池化。每个外层六折，内层验证机为排序中下一台，其余四机以seed101对五窗口各固定200passes/B64；按inner flight-end raw RMSE选，tie按总点数/H/P。再在五机重新fit scaler，选中窗口seed101/202/303各200passes。四子集共120inner+72outer=192次，只有72outer进入与模块LOEO的比较。

Native Adam lr=.001574/wd=.001保留；统一固定训练替代作者max500/patience50。标签固定min(rawY,125)/125，原mu/sigma均乘125保存周期单位；预测仅下界0，主指标raw flight-cycle RMSE、NASA sum/mean。GaussianNLL原full=False/eps1e-6/reduction=mean；normalized variance floor对应有效sigma .125cycle，预测CSV仍存原sigma*125，不能据此声称校准区间。每cell保存真实initial/final/optimizer/steps/trace/scaler/config及source/cache/execution/config/checkpoint hashes。

禁止T/Xv/hs/Fc/unitID/duration、跨子集预训练、集成、TEST选型。原论文是DS01–05联合、hs标签、随机窗验证；本次统一迁移与原报告分别登记。与模块的窗口/分辨率可不同，外部排名不能用作同信息模块因果归因。

验收门：完整核心/初始化/输出/NLL精确等价、全部五窗口B64有限前向反向、两输出头和当前年龄梯度、train-only预处理反事实、schema封印通过；0optimizer实测含25%余量不超24共享GPU小时/12GB。失败保留全部证据，不能通过简化模型或删除候选绕过。实际192训练及72outer全封印/重载后才可判统一执行SUCCESS；作者原协议未重现仍PARTIAL。

独立 Codex 子代理静态审计已PASS（无实际forward/训练），root正在运行隔离0optimizer硬件验收。冻结504与AIDGN实验不据此变更。
