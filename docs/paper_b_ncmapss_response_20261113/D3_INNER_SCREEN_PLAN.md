# D3：经典条件残差预处理的内层筛查

状态：训练前登记；看到原outer失败后的探索，不是新独立确认或已创新模块。顶层Codex L2及独立admission审查批准48次固定2×2筛查，原common贡献关闭。

每DS02/05/06/07：outer=最小dev机完全排除，innerVal=下一台，train=其余4台。seed101/202/303；raw/residual input×aux=.1/0，共48次；仅最后一次innerVal预测，不构造outer/TEST输入。所有四臂调用原冻结Model(raw)、70×512/200passes/B64/AdamW .001/wd.0001/clip1，相同block前向，仅aux系数不同，初态/完整head与encoder/scaler/processor/每epoch batch顺序/最终steps必须配对。

预处理只在train4拟合：保持原W的FP32全局zscore，FP32标准化Xs升FP64，以固定15项二次W ridge.001、无罚截距拟合m，再取训练拟合残差std(ddof0/floor1e-6)为经验s。应用(Xs−m(W))/s后转FP32，W/age/coverage/window/target保持；没有新增可训练网络参数。经验scale含模型失配与健康变化，不称noise。aux=0仍走相同前向；aux=.1的重建目标随输入坐标改变，因此不能把这个交互称纯loss尺度隔离。

唯一筛门：先平均三seed inner RMSE，再计算residual相对raw改善；同一至少3个子集在两种aux条件均改善≥2%，全部子集/aux均不得恶化>2%。48不齐不得判正门。仅一个inner机/子集，seed不是独立发动机；不作显著性或泛化确认。

预算：单进程8共享GPU墙钟小时、12GB；每batch计时硬截止，OOM/非有限/预算失败保留全部尝试，不删格/换短窗/改门。完整initial/final/optimizer/每step trace/实际inner预测/config/source/data/cache/processor和scaler哈希保存。通过仅建立经典预处理强参考，后续可学习模块须超越它并另审；未过不推进B1。

真实验收：NumPy13合同PASS（独立复核）；实际原Model四subset×四family B64前向反向16记录PASS、0optimizer，输入逆变换/NumPy-Torch/无RNG/四臂初态一致/排除机scaler隔离通过；见evidence/D3_PREFLIGHT_V1.json。含25%margin投影约1.06h<8h，峰值约465MiB。正式训练尚待以提交的完全同源文件启动。

执行SHA：d5f117e60734d99a02a56d7d3d5f50961ecaaedd24ab0c13ced57c24dfe80ced。检查程序SHA：2d57d9e8840b706ff0b3c7fbcec74f5a05b6943febbe953c68fa66cc01986aff。旧五文件execution0d5a不变。此处仅预处理参考，不是“DSPT+新模块已改善”的证据。


实际启动：source commit7bb98787459c3bf2b50fc64b84c5ec720fde243b，a40 /media/sangfor/vdb/user/zxk/ncmapss_conditioning_20261114/formal_7bb98787，PID3787224。仅48次inner_screen；formal_7bb98787.log/FAILED.json为恢复入口，禁止重复launch或改冻五文件。
