# 进度与下一步

> 权威源：docs/research_state.md（状态）、docs/decision_log.md（决策）。
> 本页只做阶段视图，不写未验证推测。

## 阶段视图

| 阶段 | 状态 | 说明 |
|---|---|---|
| A 方向探索 | 完成 | docs/idea_report.md Part 1 |
| B Idea 深化 | 完成 | PU-Net：冻结 IDA-PE + 尺度头 + 零初始化偏置头 |
| C 实验设计 | 完成 | Part 3 |
| D 实现设计 | 完成 | docs/implementation.md |
| E 编码 | 完成 | src/punet/、scripts/ |
| F 迭代 | 进行中 | 主矩阵 79/80；D001 关闭宽度路线，待新训练侧假设 |
| G 论文 | 进行中 | 稿件 + PDF 已编译；参考文献仍为占位，需 BibTeX |

## 下一步（按优先级）

1. 补 FD004 f4 s13，凑满 40 格（E001 补齐行）。
2. 由 L2（Codex）就"训练侧尺度-残差耦合"新假设给出取舍，写入 decision_log 后立项 H004。
3. 参考文献去占位：按 literature.md 登记表补 BibTeX，逐条核实。
4. 每完成一项，更新 research_state.md 与本页。

## 阻塞项

- reasoningEffort 实测 high（约定 max），属用户决定，未擅自改。

## 2026-11-09 02:xx · 本轮进展（主控线）

| 项 | 状态 | 产物 |
|---|---|---|
| D016 第 2 项：基线冻结 | ✅ 完成 | `docs/BASELINE_FREEZE.md`、`scripts/freeze_audit.py`（50/50）、D020 |
| D016 第 5、6 项：六件套草案 | ✅ 草案 v2 完成，**待 L2 立项门** | `docs/PROPOSAL_unit_coverage_identifiability.md` |
| 第一版「识别边界」框架 | ❌ 作废（文献占位） | `failures.md` F001 |
| 单位级失效跨 seed 稳定性 | ✅ 实测 | ICC 0.780、ρ +0.762、Jaccard 0.599（`scripts/engine_coverage_stability.py`） |
| 最近邻核实 | ✅ 10 篇 DOI 级核实 | `literature.md` 新增节 |

**下一步**：~~等 L2 立项门裁决~~ → **L2 已裁决：否决（D021）**。
- 不再沿 D018 立项；不启动 G5、不重跑实验。
- 已按 D016 第 8 项产出 `docs/INTERNAL_REPORT_baseline_audit.md`（内部技术报告 + 基线）。
- 若要做应用稿：先做具体期刊的贡献匹配审查，结论写 `decision_log.md`；不得当作投稿就绪认定。
- 失败登记：`failures.md` F001（第一版框架被占位）、F002（第二版计数对象混淆）。


- 2026-11-13：新N-CMAPSS机制原型已独立隔离，四缓存完成，作者代码来源保全；尚无正式训练收益证据。恢复见 docs/paper_b_ncmapss_response_20261113/。


## N-CMAPSS正式开发训练启动（2026-11-13）

源提交2e57e8ae68f194d73ba0a94fcd45437507f33d58；执行0d5ab920d31ec11767bb38f9ed717ce21f31ece3c6d981de7453cbd672d08009。
远端a40：/media/sangfor/vdb/user/zxk/ncmapss_response_20261113/formal_2e57e8ae。
父PID2331438；四子集worker2331541/2331542/2331543/2331544。七模式、六折、三seed504次真实训练，36共享GPU墙钟小时上限。
实际启动已核实；当前状态见docs/paper_b_ncmapss_response_20261113/evidence/STATUS.json，首批未完成，未作任何增益判断。
当前仅开发集，禁止TEST。原DSPT/Paper A/R2均无改动。源码后续只允许新增审计/文档，不改变冻结五个执行文件。
恢复：读远端STATUS.json；失败看formal_supervisor.log及formal_2e57e8ae/worker_*.log。不能重复launch（CLAIM独占）。
完成后核summary/DEVELOPMENT_LOEO.md与全checkpoint重载预测审计，再提交结果；四篇新NC论文统一重训尚未开始，仍PARTIAL。


2026-11-14 只读独立schema审计纠正：四HDF5 W_var+X_s_var均为 alt,Mach,TRA,T2,T24,T30,T48,T50,P15,P2,P21,P24,Ps30,P40,P50,Nf,Nc,Wf，与Bayes native完全一致。冻结prepare实际directconcat保持此顺序，但其input_columns清单错误把Wf/Nf/Nc放前；此前清单列语义撤回，以 evidence/RAW_HDF5_SCHEMA_CORRECTION.json为准。不重排数据，不改变504/AIDGN执行文件、任何数学输入或既有结果。未读取TEST观测或标签。


2026-11-14 Bayes完整Inception-HNN统一迁移：独立static审计PASS，真实PREFLIGHT_V1 source3b76be258fc6829f4c54208d29c1a1131d44014119d754a78e8f971d752f53af PASS、0optimizer，5窗口B64/原始state-output-NLL精确等价、双head及current-age有效梯度、train-only变更隔离均实际通过，峰值221MiB，含25%margin预计17.535h<24h。允许按预注册120inner+72outer启动；原作者协议仍PARTIAL，来源与schema纠正不改变504/AIDGN。真实依赖torchinfo1.8.0隔离_deps，未改系统环境。


2026-11-14 PB-NC-response闭环：原common贡献判FAILED，504训练/24年龄/全504checkpoint真实重载，primary/age/sensor/attribution/mechanism全FAIL。证据见docs/paper_b_ncmapss_response_20261113/OPEN_SOURCE_REPRODUCTION_REPORT.md及experiments/paper_b_ncmapss_response_20261113；L2记录PB-NC-POSTFAIL-20261114。D1/D2是失败后训练侧探索，非RUL增益/HI识别；旧data_diagnostics_v1逐sensor错名已撤回，数字不变的schema纠正见data_diagnostics_v2_schema_corrected。


2026-11-14 888次开发训练闭环完成：module504原科学门FAILED；完整AIDGN192和Bayes192统一迁移均执行SUCCESS，native数值PARTIAL。全888cp/2184点预测、另336sigma逐位重载0差。独立报告核60数字格及7707归档文件SHA，PASS；公开Git只含代码/预测/配置/指标/审计，大cp仍在私有a40并全SHA留存。master报告docs/paper_b_baselines/OPEN_SOURCE_REPRODUCTION_REPORT.md和NC完整报告已落；D3仍仅inner探索另表。


2026-11-14 PB-NC后续：D3已launch，source7bb98787/executiond5f117e60734d99a02a56d7d3d5f50961ecaaedd24ab0c13ced57c24dfe80ced，a40 /media/sangfor/vdb/user/zxk/ncmapss_conditioning_20261114/formal_7bb98787，PID3787224，48inner-only经典预处理筛查，不混入888outer表。source五文件连checks冻结，PREFLIGHT16PASS0optimizer；最多8共享GPU小时。新增两fresh-context数学/查新代理拒绝‘函数范数白化/解码差’作为新贡献：精确FE/KRR等价且background歧义未解，未写代码/未起GPU，见FUNCTIONAL_CONTRAST_REJECTION.md。H-CDE书目按出版社更正为2027卷期/2026公开记录，旧简写RESS2026仅指作者记录年。
