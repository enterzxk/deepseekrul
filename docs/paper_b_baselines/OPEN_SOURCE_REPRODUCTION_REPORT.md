# Paper B：公开源码强基线复现报告

数据与代码审计日期：2026-11-11。执行人为 Codex；三个独立 `gpt-6.1-sol/high` 子代理只读审计。
冻结 DSPT、Paper A、旧实验均未修改。以下数字全部由保存的真实预测与指标文件生成；缺测不补填。

## 1. 复现判定与证据覆盖

SUCCESS/PARTIAL/FAILED 的门见 [PROTOCOL.md](PROTOCOL.md)。SUCCESS 要求两协议各12次完整运行、
审计通过，以及每个子集的作者公开 RMSE 相对差在±10%以内。PARTIAL 不等于模型被简化：
也用于真实复现完成但公开数值不可核实、论文/代码协议有差异或数值门未达到。

| 方法 | 判定 | 作者协议 | 统一协议 | 未满足项 |
|---|---|---:|---:|---|
| AIDGN-RUL | **PARTIAL** | 12/12 | 12/12 | exact historical launcher source missing/unrecorded for 7 author runs; upstream/core/runtime patches retained |
| Two-Phase Attention | **PARTIAL** | 12/12 | 12/12 | published cycles-RMSE/NASA table unverified or not applicable; exact historical launcher source missing/unrecorded for 6 author runs; upstream/core/runtime patches retained |
| Dual-Mixer+FSGRI | **PARTIAL** | 12/12 | 12/12 | published cycles-RMSE/NASA table unverified or not applicable; public-code versus preprint hyperparameters differ; paper result is normalized/all-window; exact historical launcher source missing/unrecorded for 7 author runs; upstream/core/runtime patches retained |
| FFVE | **SUCCESS** | 12/12 | 12/12 | 全部预注册门通过 |
| FC-STGNN | **SUCCESS** | 12/12 | 12/12 | 全部预注册门通过 |
| CPT（第六候选） | **FAILED** | 原始预训练接口检查 | 未纳入优先训练矩阵 | forward输出[B,1]被再次传入期望36维的classifier；没有用普通回归模型替代 |

本表判定不代表模型相对 DSPT 获胜。数值差异与协议风险完整保留。

## 2. 论文身份、官方来源与固定源码

逐库实现/数据/训练/测试/checkpoint 审计见 [SOURCE_AUDIT.json](SOURCE_AUDIT.json)。
源锁与逐文件 Git blob/SHA256 见 [sources.lock.json](../../paper_b/baselines/open_source/sources.lock.json)。
原始源码按commit下载后逐字节保留，不在原文件上打补丁。因部分上游未授予再分发许可，
本仓库提交获取器、精确manifest、适配器与运行补丁，原始代码由获取器恢复。

| 方法 | DOI | 固定 commit | 可公开核查的结果来源 |
|---|---|---|---|
| AIDGN-RUL | [10.1109/TII.2026.3694202](https://doi.org/10.1109/TII.2026.3694202) | [`d1ac173b1365`](https://github.com/TCCofWANG/AIDGN-RUL/commit/d1ac173b13659af1854cf500a9b1cba8367353fe) | [来源](https://github.com/TCCofWANG/AIDGN-RUL/blob/d1ac173b13659af1854cf500a9b1cba8367353fe/README.md) |
| Two-Phase Attention | [10.1109/TIM.2025.3576953](https://doi.org/10.1109/TIM.2025.3576953) | [`5bd33b2b1bb0`](https://github.com/mjliu-advertising/RUL/commit/5bd33b2b1bb044ab4be1a8cb67fdf8a5d96d1d0a) | 数值表未取得 |
| Dual-Mixer+FSGRI | [10.1016/j.ress.2024.110398](https://doi.org/10.1016/j.ress.2024.110398) | [`727c020cabba`](https://github.com/fuen1590/PhmDeepLearningProjects/commit/727c020cabba2c6ae96e8f0e28f7f0121b292e81) | [来源](https://arxiv.org/html/2401.16462#S4.T4) |
| FFVE | [10.1016/j.aei.2023.102316](https://doi.org/10.1016/j.aei.2023.102316) | [`dd74264d8bb0`](https://github.com/LLerDagu/RUL-FFVE/commit/dd74264d8bb034e74ed7bc0133325f4e05ecebdf) | [来源](https://github.com/LLerDagu/RUL-FFVE/blob/dd74264d8bb034e74ed7bc0133325f4e05ecebdf/Benchmark.md) |
| FC-STGNN | [10.1609/aaai.v38i14.29500](https://doi.org/10.1609/aaai.v38i14.29500) | [`93f2387feff4`](https://github.com/Frank-Wang-oss/FCSTGNN/commit/93f2387feff4457b17016b33931a7dc6ec757719) | [来源](https://arxiv.org/pdf/2309.05305) |
| CPT | [10.1088/2631-8695/ae88b8](https://doi.org/10.1088/2631-8695/ae88b8) | [`0b4a32e8b163`](https://github.com/ZeyU1999/RUL-Prediction/commit/0b4a32e8b163709440b9c225ca1b4fffba7f1eea) | 数值表未取得 |

公开 checkpoint：AIDGN tree含checkpoint/result，但也有其他模型/数据集；Two-Phase发布两份paired预训练权重；
其余四库未见公开checkpoint。统一协议全部从头训练，包括排除双方验证机后的Two-Phase预训练。

## 3. 作者公开报告（引用值，非本次训练）

下表每格为 RMSE / NASA Sum，顺序FD001–004。AIDGN来自作者README，未取得正文结果表；
FFVE采用作者2024-01-22披露RMSE计算bug后的修正表；FC-STGNN为原文Table1的10次均值。

| 方法及来源口径 | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| AIDGN-RUL | 11.46 / 248.49 | 12.63 / 663.80 | 10.52 / 202.87 | 13.03 / 762.25 |
| FFVE | 12.31 / 280.21 | 12.55 / 772.17 | 12.43 / 302.10 | 14.08 / 1149.03 |
| FC-STGNN | 11.62 / 203.00 | 13.04 / 738.00 | 11.52 / 198.00 | 13.62 / 816.00 |
| Two-Phase | 未核实 | 未核实 | 未核实 | 未核实 |

Dual-Mixer-F作者预印本Table4报告**归一化全窗口RMSE**（3次均值）：0.1002/0.1338/0.0887/0.1141，
并报告MAPE，未报告NASA。该表不能转换成官方逐机RMSE排名，也不能直接乘125。

## 4. 忠实作者代码协议：实际训练

AIDGN、Two-Phase和FC-STGNN为逐机cycles指标。FFVE本表的RMSE与NASA从同一次实际随机预测重算；
下一表另保留作者代码所报的另一随机draw RMSE。Dual-Mixer单独列全窗口结果。所有均值±样本SD；
完整逐seed与训练时间、参数量、config/checkpoint hash见 [author_per_seed.csv](author_per_seed.csv)。

| 方法 | 子集 | RMSE cycles | NASA Sum | seeds |
|---|---|---:|---:|---:|
| AIDGN-RUL | FD001 | 11.38 ± 0.26 | 233.96 ± 7.85 | 3/3 |
| AIDGN-RUL | FD002 | 12.51 ± 0.13 | 656.39 ± 24.28 | 3/3 |
| AIDGN-RUL | FD003 | 10.43 ± 0.73 | 208.61 ± 16.80 | 3/3 |
| AIDGN-RUL | FD004 | 13.29 ± 0.08 | 841.79 ± 21.86 | 3/3 |
| Two-Phase Attention | FD001 | 12.85 ± 0.28 | 324.67 ± 17.14 | 3/3 |
| Two-Phase Attention | FD002 | 14.02 ± 0.36 | 961.56 ± 48.72 | 3/3 |
| Two-Phase Attention | FD003 | 11.93 ± 0.58 | 290.47 ± 48.48 | 3/3 |
| Two-Phase Attention | FD004 | 14.88 ± 1.00 | 1153.23 ± 225.49 | 3/3 |
| FFVE | FD001 | 11.87 ± 0.40 | 267.75 ± 35.73 | 3/3 |
| FFVE | FD002 | 12.79 ± 0.34 | 805.89 ± 46.86 | 3/3 |
| FFVE | FD003 | 12.17 ± 0.90 | 286.73 ± 47.22 | 3/3 |
| FFVE | FD004 | 14.19 ± 0.31 | 1227.83 ± 164.82 | 3/3 |
| FC-STGNN | FD001 | 12.39 ± 0.63 | 249.72 ± 44.17 | 3/3 |
| FC-STGNN | FD002 | 13.95 ± 0.69 | 848.47 ± 130.83 | 3/3 |
| FC-STGNN | FD003 | 12.37 ± 0.45 | 244.01 ± 46.85 | 3/3 |
| FC-STGNN | FD004 | 14.09 ± 0.35 | 1043.84 ± 180.29 | 3/3 |

作者公开值差异：每格为本次作者协议RMSE均值减公开值（cycles），括号内相对差。Two-Phase无已核实值不计算。

| 方法 | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| AIDGN-RUL | -0.08 (-0.7%) | -0.12 (-0.9%) | -0.09 (-0.8%) | +0.26 (+2.0%) |
| FFVE | -0.00 (-0.0%) | +0.30 (+2.4%) | +0.07 (+0.6%) | +0.22 (+1.5%) |
| FC-STGNN | +0.77 (+6.6%) | +0.91 (+7.0%) | +0.85 (+7.4%) | +0.47 (+3.4%) |

NASA Sum差异（本次作者协议均值减作者公开值，括号内相对差）：

| 方法 | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| AIDGN-RUL | -14.53 (-5.8%) | -7.41 (-1.1%) | +5.74 (+2.8%) | +79.54 (+10.4%) |
| FFVE | -12.46 (-4.4%) | +33.72 (+4.4%) | -15.37 (-5.1%) | +78.80 (+6.9%) |
| FC-STGNN | +46.72 (+23.0%) | +110.47 (+15.0%) | +46.01 (+23.2%) | +227.84 (+27.9%) |

| FFVE原代码实际报告（独立随机draw） | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| RMSE cycles | 12.31 ± 0.21 | 12.85 ± 0.26 | 12.50 ± 0.87 | 14.30 ± 0.29 |

### Dual-Mixer+FSGRI：原始全窗口口径单独列示

完整FSGRI已显式开启。保留原始Gaussian sampler、正样本噪声和weighted InfoNCE+MSE。
代码默认lr/batch/threshold与预印本Table3不同，不能声称严格复现其论文超参。

| 子集 | 原归一化RMSE | 相对原文差异 | 逆变换后全窗口RMSE cycles | 全窗口NASA（非逐机） |
|---|---:|---:|---:|---:|
| FD001 | 0.1068 ± 0.0020 | +6.5% | 13.45 ± 0.25 | 34201.06 ± 1962.58 |
| FD002 | 0.1417 ± 0.0026 | +5.9% | 17.85 ± 0.33 | 184338.64 ± 4583.27 |
| FD003 | 0.0904 ± 0.0024 | +1.9% | 11.39 ± 0.31 | 35647.99 ± 2375.94 |
| FD004 | 0.1334 ± 0.0017 | +16.9% | 16.81 ± 0.22 | 547051.17 ± 8993.77 |

作者标签为 `(capped_RUL+1)/denominator`；真实四子集的denominator均126，故逆变换为`126*x-1`。
更正后每个窗口真值已与raw unit/cycle/RUL核对。作者丢弃短机，endpoint仅FD002 253/259、FD004237/248；
这些retained-engine endpoint文件只作附录，不进入完整官方TEST表。统一协议保留全部发动机。

## 5. DSPT匹配的统一协议：官方TEST

相同engine-ID划分（80/20，split_seed7/fold0）、seeds101/202/303、cap125、cycles输出、NASA Sum。
NASA定义：d=预测cycles−真实cycles；逐机d<0取exp(−d/13)−1，否则exp(d/10)−1，最后求和。
Validation为每个独立验证机生命周期40/55/70/85/95%五截断点。所有统计仅fit训练机。按validation选checkpoint，
保存config/source/checkpoint freeze后才允许读取official test；访问审计禁止训练期读test或RUL文件。
统一输入采用DSPT的训练机拟合KMeans(k=6)工况簇+sensor z-score；FD001/3的六个簇并非六种真实工况。短验证/测试历史左零填充，训练使用完整方法窗口。
Two-Phase另保留标准化OC输入，FFVE另保留因果指数平滑。这些预处理与部分作者管道不同；本表比较统一输入协议下的完整架构/目标，非原端到端预处理性能。
每次run只评一次官方TEST；以下结果不与历史OOF或验证窗口指标混用。

窗口范围为作者默认的单点：AIDGN50、Two-Phase45、Dual30、FFVE30、FC30/50/50/50；调窗预算0次。
Two-Phase45由特征注意力硬编码决定。每个子集允许各自原始学习率与架构参数；完整搜索预算已预注册。

单点窗口预算未证明每个方法的最佳调优性能；不得因已看到TEST差异而追加调参。

### 5.1 单子集训练方法

| 方法 | 子集 | 官方RMSE cycles | 官方NASA Sum | 参数量 |
|---|---|---:|---:|---:|
| AIDGN-RUL | FD001 | 15.45 ± 2.96 | 7144.07 ± 11182.84 | 30070 |
| AIDGN-RUL | FD002 | 14.22 ± 0.21 | 912.18 ± 66.59 | 30070 |
| AIDGN-RUL | FD003 | 12.25 ± 0.62 | 287.72 ± 55.83 | 30070 |
| AIDGN-RUL | FD004 | 14.82 ± 0.61 | 1542.96 ± 368.39 | 30070 |
| Dual-Mixer+FSGRI | FD001 | 13.81 ± 0.78 | 299.44 ± 64.52 | 64123 |
| Dual-Mixer+FSGRI | FD002 | 14.94 ± 0.68 | 944.95 ± 81.62 | 64123 |
| Dual-Mixer+FSGRI | FD003 | 13.35 ± 0.77 | 380.88 ± 131.27 | 64123 |
| Dual-Mixer+FSGRI | FD004 | 17.23 ± 0.19 | 3113.11 ± 571.62 | 64123 |
| FFVE | FD001 | 12.43 ± 0.73 | 302.49 ± 49.08 | 207480 |
| FFVE | FD002 | 13.06 ± 0.51 | 753.23 ± 60.30 | 235873 |
| FFVE | FD003 | 12.85 ± 0.40 | 337.97 ± 24.16 | 207480 |
| FFVE | FD004 | 14.63 ± 0.30 | 1263.92 ± 101.37 | 412273 |
| FC-STGNN | FD001 | 13.50 ± 0.70 | 340.23 ± 79.82 | 19921 |
| FC-STGNN | FD002 | 17.14 ± 3.05 | 20222.47 ± 32949.59 | 28073 |
| FC-STGNN | FD003 | 13.82 ± 0.23 | 294.19 ± 30.94 | 592989 |
| FC-STGNN | FD004 | 15.35 ± 0.64 | 1450.94 ± 188.86 | 27293 |

### 5.2 跨子集预训练资源条件：Two-Phase单列

完整方法需要FD001+3或FD002+4预训练，双方验证机均排除；不把额外训练数据隐藏进单子集表。
encoder完整迁移并冻结，GRL与分类/回归预训练均保留。

统一实现按整条流水线设seed，先初始化FT再初始化PRE，阶段切换不重置RNG；作者两个脚本分别重置同一seed。
独立初始化审计确认该差异，未读取统一TEST数字，并决定保留冻结实现、不据已见结果切换随机流。
因此不主张逐阶段初始化/训练轨迹与作者同seed完全一致；旁注绑定原config哈希，原config不改。见[SEED_POLICY_AUDIT.json](SEED_POLICY_AUDIT.json)。

| 子集 | 官方RMSE cycles | 官方NASA Sum | 参数量（总量 / 可训练） |
|---|---:|---:|---:|
| FD001 | 13.34 ± 0.79 | 250.38 ± 29.74 | 12969 / 1649 |
| FD002 | 15.01 ± 0.45 | 858.41 ± 79.99 | 24289 / 1649 |
| FD003 | 13.29 ± 0.18 | 322.40 ± 26.05 | 12969 / 1649 |
| FD004 | 15.58 ± 1.48 | 2011.99 ± 1253.62 | 24289 / 1649 |

统一协议完整逐seed数据见 [unified_per_seed.csv](unified_per_seed.csv)。逐指标Student-t 95%CI、
训练时间、参数量见 [summary.json](../../paper_b/experiments/open_source_reproduction_20261111/summary.json)。
CI仅反映三次训练随机性，不增加独立发动机数量，也不消除固定开发集被重复使用的限制。

官方TEST在历史项目和本次作者协议中已经被观测，本次不是全新盲测。统一配置在新增统一测试前锁定，冻结后不依据TEST调参。

## 6. 核心一致性、协议差异与失败留痕

- AIDGN：原aging投影、动态图、图卷积保留；OCC=可见cycle/训练机平均寿命×1e-3。原代码滑窗验证改为engine验证。
- Two-Phase：原代码保留paired预训练和冻结encoder；作者协议fit train+test特征统计，统一协议仅fit训练机。
- Dual-Mixer：双路径gated exchange和完整FSGRI保留；统一标签为RUL/125，统一test对短机零padding、不丢engine。
- FFVE：按作者四子集脚本保留1/2/1/2 blocks和FD004长残差；保留随机重参数与KL+RMSE；统一validation/test用固定种子的单次实际随机draw，非取mu、非ensemble。
- FC-STGNN：两图分支与每FD patch/CNN参数完整保留；作者test-selected epoch只出现在作者表，统一表只按validation选模。
- 原始作者码始终独立保存；可选依赖导入、子集参数化、随机种子、纯可视化callback及test行序调整均有runtime patch。
- FFVE首次启动缺imageio：校验wheel后仅安装到隔离目录；FC-STGNN首次启动因IterableDataset抽象方法失败：仅为原数组容器加不可迭代stub。两者v3失败证据保留，v4完整重跑；独立审计确认网络和原数组处理未改。
- 早期Two-Phase行序和Dual×125导出记录保留；最终使用RECONCILED_metrics及已校验预测，旧导出不得引用为cycles/engine证据。
- 第六候选原预训练接口在真实训练输入上报shape错误，证据保存为candidate_execution.json；没有宣称完成训练。

## 7. 可复算与完整性

本次共收集并独立复算 **120** 次完整训练。所有数字源于逐run checkpoint/预测/配置；
RMSE与NASA由独立math.fsum/expm1实现复算，checkpoint/config hash核对，权重有限性检查，统一engine数量与官方标签核对。
20组真实train-only前向/反向检查及最终FFVE配置复检均有原始日志。原始失败、初始错误导出与修正链不删除。

历史执行器证据：作者协议60次中，40次保留了与config记录哈希完全一致的launcher源码；
16次有历史launcher哈希但未找回完全匹配文本，4次早期config未记录该哈希。
已从Git与初始部署的纯文本差异中按精确SHA恢复两版自身脚本；未执行候选文本或改写原config。缺口逐run列于
[EXECUTION_PROVENANCE.json](../../paper_b/experiments/open_source_reproduction_20261111/evidence/EXECUTION_PROVENANCE.json)。
固定上游代码、实际runtime patch、checkpoint及预测均保留，但不将这20次声称为完整逐字节launcher追溯；相关方法判为PARTIAL。
统一协议60次的四个冻结核心文件均有config哈希并经独立核对。

主要产物：

- [逐run记录](../../paper_b/experiments/open_source_reproduction_20261111/runs.csv)
- [指标与holdout审计](../../paper_b/experiments/open_source_reproduction_20261111/metric_audit.json)
- [工件SHA256目录](../../paper_b/experiments/open_source_reproduction_20261111/ARTIFACT_MANIFEST.json)
- [复现命令与目录说明](../../paper_b/baselines/open_source/README.md)

作者elapsed为包含预处理/最终测试的整段wall-time；统一training_time至冻结checkpoint（包含预训练/预处理、不含TEST）。
并发run的wall-time之和是process-hours，不冒充独占GPU卡时。
没有从官方test结果挑seed、删子集、改窗口或修改冻结DSPT。
