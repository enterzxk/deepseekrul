# Paper B N-CMAPSS：可复核的真实开发实验

本目录是独立Paper B产物。888次真实训练=504模块消融+192完整AIDGN+192完整BayesRUL-HNN。每篇192包含120 inner选窗和72 outer重训。inner不能进入统一outer主表。旧C-MAPSS/NC-R2/官方TEST不在本表中。

- `UNIFIED_DEVELOPMENT_COMPARISON_V2.csv`：当前主对照，40行=四subset×10方法/参考；V1快照保留。
- `artifacts/*_light_artifacts_v1.tar.gz`：完整真实预测、metrics、config、scalers、trace、CLAIM/SUCCESS、选窗和完成标记；没有原始NASA观测/作者私有源码/checkpoint二进制。三个tar分别以prototype/、aidgn/、bayes/为顶层前缀；manifest内部path相对该前缀。
- `manifests/`：每个archive及内部逐文件SHA256。解包后逐文件验SHA，不接受修改后结果；`.gitattributes`保留CRLF/LF字节以维持封印。
- `*_collection_v1/`：完整源码/初末状态/真实步数/scaler/指标/inner选窗核验及独立汇总；`outer_cases.csv`与`inner_cases.csv`分开。
- `*_replay_full_v1/REPLAY_AUDIT.json`：全部888 checkpoint重载；2184个点预测向量及336个Bayes sigma逐位一致，最大差0。
- `source_manifests/`：原实际runtime精确manifest。生产文件commit/执行hash与报告一致，当前新增审计文件不会改producer身份。

## 原checkpoint位置

现有授权a40（未改SSH或权限）逐seed真实initial/checkpoint/optimizer持续保存在：

```text
/media/sangfor/vdb/user/zxk/ncmapss_response_20261113/formal_2e57e8ae/runs/<cell>/
/media/sangfor/vdb/user/zxk/ncmapss_public_aidgn_20261114/formal_b6f8a07d/cells/<cell>/
/media/sangfor/vdb/user/zxk/ncmapss_bayes_hnn_20261114/formal_1cdfc7fb/cells/<cell>/
```

完整initial.pt/checkpoint.pt SHA在各cell的SUCCESS与metrics、collector/replay中。大权重没有写入Git；本地/远端原结果不覆盖、不删除。复核需有现有a40访问或另行取得这些权重，不能把只下载本Git当成已下载所有checkpoint。

## 复跑与校验

实际环境Python3.9.25/PyTorch2.5.1+cu121、CUDA12.1/NumPy1.26.4。Bayes另需真实torchinfo1.8.0 wheel，隔离 `_deps`，其SHA见docs/paper_b_ncmapss_response_20261113/evidence/TORCHINFO_DEPENDENCY.json。没有伪造依赖、修改作者文件或系统profile。当前producer与作者native环境不同，详见各protocol。

1. 从生产commit建立独立runtime；将对应`*_GIT_SOURCE_MANIFEST.json`复制为runtime根`GIT_SOURCE_MANIFEST.json`并逐文件验证。外部作者源码按原完整commit和已提交`SOURCE_MANIFEST.json`取回到独立只读目录，不用最新main代替。
2. dev cache需按固定prepare与四原HDF5 SHA重建，只允许A_dev前两列、W_dev、Xs_dev、Y_dev。cache与manifest的SHA、原schema、flight采样索引均需核验。当前repo忽略原NASA观测，不包含四dev NPZ。
3. 按冻结包PROTOCOL运行前重新执行硬件/因果/梯度preflight，使用新输出目录。模块入口`paper_b.ncmapss_response.run`，AIDGN入口`paper_b.ncmapss_public_20261113.run_aidgn`，Bayes入口`paper_b.ncmapss_bayes_20261114.run_bayes`；参数见各入口`--help`与已保留LAUNCH证据。原结果绝不覆盖。
4. 复核可直接运行审计脚本，以下为已有a40目录示例（不训练、不读TEST）。审计脚本放在冻结runtime外，Bayes需将真正 `_deps` 加入PYTHONPATH。

```text
python paper_b/ncmapss_public_20261113/collect.py --runtime /media/sangfor/vdb/user/zxk/ncmapss_public_aidgn_20261114/runtime_b6f8a07d --cache /media/sangfor/vdb/user/zxk/ncmapss_response_20261113/data --runs /media/sangfor/vdb/user/zxk/ncmapss_public_aidgn_20261114/formal_b6f8a07d --author-file /media/sangfor/vdb/user/zxk/ncmapss_public_aidgn_20261114/sources/aidgn/Model/AIDGN.py --output <NEW_AUDIT_DIRECTORY> --require-complete
```

同目录`replay.py`接受相同参数；Bayes的collect/replay在其独立包，source路径必须是完整`bayesrul/models/nets/inception.py`。它同时重放原mu和sigma。模块全量重载入口为`paper_b/ncmapss_response/replay.py`，`--runs`指向formal的`runs/`，不是包含解析age参考的顶层。实际所有命令与exit状态已保留，不以静态PASS冒充训练成功。

## 解释边界

统一表为DEVELOPMENT_LOEO：每seed先对六engine RMSE等权，再报三seed均值/SD。NASA总和按所有相同flight query求和，mean与SD分列。训练cap125、主评raw flight-cycle，cap125敏感性另存；预测只下界0，不上限裁剪。作者原协议和原分数未重现的部分判PARTIAL；完整迁移执行SUCCESS不等于作者原协议数值SUCCESS。

注册common模块五门失败，贡献判FAILED，全部负结果保留。D1/D2以及未来D3是看到outer开发失败后的探索，不能冒称新独立确认；D3内层结果绝不拼到outer主表。详见`docs/paper_b_ncmapss_response_20261113/OPEN_SOURCE_REPRODUCTION_REPORT.md`。
