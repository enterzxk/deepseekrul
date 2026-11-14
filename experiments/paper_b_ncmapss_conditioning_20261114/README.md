# D3 INNER_SCREEN：48次经典预处理强对照

本目录与888主矩阵分离。只评每子集固定一个inner发动机，outer及官方TEST不构造；不得向`UNIFIED_DEVELOPMENT_COMPARISON_V2.csv`追加本内层成绩。

实际48/48训练、12配对组、48checkpoint全量重载PASS，unbounded预测全逐位0差；预设方法筛查门NOT_MET。只有DS02达到两aux条件≥2%改善，不能挑该子集认定模块有效。完整表见docs/paper_b_ncmapss_response_20261113/D3_INNER_SCREEN_RESULT.md。

- artifacts/conditioning_light_artifacts_v1.tar.gz：含conditioning/顶层前缀的完整真实CSV/config/scaler/processor/step与epoch trace/metrics/CLAIM/SUCCESS/配对及总SUMMARY；不含NASA原始观测或二进制checkpoint。
- manifests/：archive与447个内文件逐字节SHA/bytes。
- audit_full_v1/：实际完整软件/产物/optimizer/配对/真实CSV指标核验与48次重载对比。audit由producer实现者编写；独立静态与数据审查的范围另行标明。
- source_manifests/：原runtime精确Git字节manifest；生产commit7bb98787459c3bf2b50fc64b84c5ec720fde243b，execution d5f117e60734d99a02a56d7d3d5f50961ecaaedd24ab0c13ced57c24dfe80ced。
- summary/、launch/：真实固定门及启动receipt。

大权重保留在私有a40 `/media/sangfor/vdb/user/zxk/ncmapss_conditioning_20261114/formal_7bb98787/cells/<cell>/{initial.pt,checkpoint.pt}`。每个SHA在SUCCESS/metrics/AUDIT中。当前没有公开权重下载；Git仅有轻量完整预测证据。所有结果不覆盖旧实验。

复核入口：`python -B paper_b/ncmapss_conditioning_20261114/audit.py --runtime <runtime_7bb98787> --cache <四devNPZ缓存> --runs <formal_7bb98787> --output <新目录> --replay --device cuda:0`。必须匹配原缓存/manifest，不要将新run或最新main冒称原源。该命令不更新optimizer，不读TEST；输出目录拒绝覆盖并保护输入。

源码/数据/标签/预算/门预先固定在包内PROTOCOL。FP32物理标准化后才升FP64拟合二次条件模型和经验残差尺度；W/age/coverage/窗口/标签不变，四臂同初态/更新/顺序/block。两aux条件都保留相同前向，仅损失系数不同；原Model/冻结DSPT/Paper A均未改。

独立真实结果复核PASS：functional_method_review未调用producer audit，自行核13Git源码、542封印文件、48组CPU初末权重/AdamW、50,400条step、12配对组及48CSV原/cap125全部指标和NOT_MET门，指标差0；它未另做forward或原HDF5重建。root全48原forward重载则另由AUDIT证实逐位0差，两类证据范围分开。
