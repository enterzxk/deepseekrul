# N-CMAPSS DS02 registry notes（append-only）

状态：进行中
最后更新：2026-11-13 04:22 (+08)
适用范围：`docs/paper_b_ncmapss/registry/` —— **与 C-MAPSS 的 `docs/paper_b_hc/registry/` 完全独立**，
不共享文件、不共享 run_id 空间、不互相引用行。

写入人：`runs_<subset>.csv` 只由实验结果管理员写；`launch/`、`close/` 由 3090 上的训练进程写。

---

## 0. 变更记录

| 版本 | 日期 | 改了什么 | 为什么 |
|---|---|---|---|
| **v20** | 2026-11-13 | ① 测试集在**第 15 代锚**下开封，时序可机械核验：`freeze_ts_utc` `20:01:43Z` → `eval_start_utc` `20:05:25Z`（**+222 秒**），opens 记录的 `protocol_blob` `6ec1c014…`、`data_sha256`、`test_units` 三项与清单一致。`scripts/nc_freeze.py verify` 现在自动核 opens 记录。② opens 记录在 3090 上是**未跟踪文件**（且不在 `.gitignore` 里），一次 `git clean -fdx` 就会毁掉唯一的时序证据；已按 base64 **逐字节**拉回并提交 `freeze/opens/NCM-FREEZE-20261112T2001Z.opens-snapshot-20261112T200525Z.json`（sha256 `047ba0a4…`，与远端实算一致）。③ **`code_md5` 对不上，登记在案**：清单冻结的是 `b889bd2` 的代码，测试批实际跑在 `ad53371` 上——`train.py` `b749f611…`→`7a961345…`、`run_loeo.py` `3da6f9f8…`→`86cbaca0…`，`data.py` / `features.py` / `evaluation.py` **逐字节未变**。④ 登记前必须先解决三件：运行器写出的 `runs_test.csv` 是 **62 列**（缺 `subset` / `coincident`），`code_md5` 与 `eval_start_utc` **两列为空**。⑤ 测试行**不进**已封存的 dev 表，另开 `runs_<subset>_test.csv`（同 64 列）。`SCHEMA` §8 第 6 条同步。 | ① 「先冻后开」不能靠自己声明，必须一条命令能从文件算出来。② 证据只活在一台机器的一个未跟踪文件里，就不是证据——**封存只有推上去才算数**。③ 清单的 `code_md5` 描述的是跑 dev 那批的代码，而测试路径在 `b889bd2` 时还不存在（§6.10 是之后才定义的），两份 md5 **必然**不同；把「哪些文件变了、变了什么」钉住比假装一致安全——diff 只涉及 `load_split(..., fitted=)`（test 的变换必须在 dev 上拟合）和 `fit_rung(..., no_holdout=True)`（验证曲线记 NaN、关早停），损失、优化器、超参、架构、标签构造一行没动。④ 62 列 + 两个空列直接封，等于把窟窿钉死，而封存后只许追加更正行。⑤ 已封存的 dev 表只许追加 superseded 行，测试行往里塞就是改历史。 |
| **v19** | 2026-11-13 | ① 协议换**第 15 代锚** `7e4fc04` / `6ec1c014…`：§6.10 第 2 条的 N0 撞地板计数由 `20/24` 改为 **`14/24（DS02 5/6 + DS05 3/6 + DS06 4/6 + DS07 2/6）`**，并写明那个 20 是 DS02 四档合计的 20/60。② 冻结清单重签为 `freeze/NCM-FREEZE-20261112T2001Z.json`（`5c9b15a5…`），`protocol_blob` / `freeze_commit` 跟到新锚；`1955Z` 移入 `superseded/`。**40 格 `epoch_star_values` 与其余字段逐字节未变**。③ 新增两条**写作规则**（§1 末）：聚合计数必须带分组明细且明细相加等于总数；冻结清单的哈希规则（`freeze_ts_utc` 进哈希 + 固定规范化形式）。④ `SCHEMA` §3.3 锚表补第 14、15 代，并逐条对照「改动必须由测量失效证明」这条分界线。 | ① 这句话是**可机械复算的精确断言**，复算只需四份已封存台账，任何人两分钟就能发现；留着它等于在一个已冻结的文档里放一个已知错数。批还没开跑，换锚的代价是几分钟。② 运行器的闸门会拿清单的 `protocol_blob` 与本次核到的锚比，对不上就拒——所以正文与清单必须同一步到位，中间不留缝。③ 裸聚合必然要求读者复算，而人复算会错；这个错出现三次，每次都是分子分母来自不同聚合，**结构性的，不是粗心**。④ 两次前移都不是结果触发的：14 代补的是「全 dev 训练时 `epoch*` 从哪来」（测量缺失），15 代改的是一个算错的计数（测量错误），都没有一行测试结果参与。 |
| **v18** | 2026-11-13 | ① 作废清单移到 `freeze/superseded/`，**字节未动**（移后按 v16 规则复算摘要仍为 `3c27009e…`）；确立不变量「`freeze/*.json` 恰好一份」，`SCHEMA` §5 目录结构与 §9 自检第 5 条同步。② 规范化形式写死进 `freeze/INDEX.md`：`ensure_ascii=False` + `separators=(',',':')` + `sort_keys=True` + UTF-8，并写明闸门与签名器必须同形。③ 明确闸门应拒绝「运行器算出的 `epoch*` ≠ 清单 `epoch_star_values`」的开封请求。 | ① 作废后 `freeze/` 下有两份 `frozen:true`，闸门那条「唯一 live 清单」会当场卡住——而往已签清单里加 `status` 字段会让它自己的摘要失效。移动目录不改内容，摘要照样可复算，且「哪份算数」一条 `ls` 就够，不用让闸门去猜字段。② `json.dumps` 默认 `ensure_ascii=True`，签名器与闸门用不同形式时，已提交的清单会被判成「被改过」——这种误判会把整批 run 挡在门外，而且看起来像篡改。③ 清单里放了 `epoch*` 而运行器自己重算，等于冻结白做。 |
| **v17** | 2026-11-13 | ① 冻结清单**重签**为 `freeze/NCM-FREEZE-20261112T1955Z.json`（40 格），哈希规则升 **v17**：`freeze_ts_utc` **纳入**哈希，只排除 `eval_start_utc` 与哈希自身。② 新增三条结构性字段 `sequencing` / `completeness` / `epoch_star_values`（附 `floor_statistics`）；40 格 `epoch*` 在签名时从四份封存台账机械复算写入。③ 旧清单 `NCM-FREEZE-20261112T1945Z` 记 `superseded`，**字节未动**，其摘要只能按 v16 规则复算；新增 `freeze/INDEX.md` 记录状态与复算方式。④ **更正一处数字**：§6.10 的「N0 的 24 折里 20 折撞地板」应为 **14 折**（DS02 5/6、DS05 3/6、DS06 4/6、DS07 2/6）；20 是 DS02 `inner_val` 行数（20/60），不是 N0 折数。⑤ 新增 `scripts/nc_freeze.py`（`sign` / `verify`），`SCHEMA` §6 与 §9 同步。 | ① 摘要不保护 `freeze_ts_utc`，等于「先冻后开」的唯一证据可以被改早而摘要照样对得上——这是我们最不能留的洞。② §6.10 说 `epoch*` 在开封前冻结，可这些数不在被哈希的清单里就执行不了：谁都能在开封后按自己的算法重算一遍，而且看起来完全合规。③ 改已签清单的字节会让它自己的摘要失效，作废只能记在外部索引，不能回头改历史文件。④ 同一个数换一个分母就变成另一个结论，而两个版本在正文里长得一模一样——这类错不会自己暴露，只能靠机械复算；协议正文是郑总的、改它要动锚，所以先记进清单的 `known_erratum` 并提请裁决，不擅自改。⑤ 「清单自证」应该是一条命令，不是一段解释。 |
| **v16** | 2026-11-13 | ① 协议第 14 代锚：`5a6b1e7` / `497afe23…`，新增 §6.10（全 dev 模型的 `epoch*` 定义）。② 冻结清单 `freeze/NCM-FREEZE-20261112T1945Z.json`：4 子集 × 40 格 × 2 评价点 = 80 行。③ **披露一条 dry run**：代码验证期在旧锚 `a3c40c63` 下跑过一格（DS02 N0，test rmse 8.915 / 9.310、bias +6.48），数值不作为证据、不参与任何选择、不影响任何规则。 | ① 全 dev 训练没有内层划分，`epoch*` 不是选出来的，必须由 dev 六折估计并在开封前冻结，否则测试集一开就没有可复算的定义。② 清单不含 `eval_start_utc`，改为运行器另写 `<freeze_id>.eval_start.json`——写进清单会让文件在开封后变化，哈希就失效了。③ 那个 test 读数是真的存在过的，藏起来比披露危险；披露并声明用途，才不会变成事后可用的证据。 |
| **v15** | 2026-11-13 | 四个子集全部封存：`runs_DS05/06/07.csv` 各 120 行、64 列，`audit_verdict=pass`；`SUBSETS.md` 记录四份文件 sha256、`git_sha`、`protocol_blob` 与审计结论。封表前做了**480 行全量机械审计**（ckpt sha256、`rmse_raw_unit` 从 `pred.csv` 复算、`n_flights`），0 失败。另记一条**跨批次复现**：DS02 试点批（`49f6939`，无 ckpt）与四子集批（`b889bd2`，有 ckpt）的 480 个指标值逐位相同，最大差 0.0。 | 封表前的全量审计是「能不能给 pass」的依据——DS02 试点批因为没存权重只能给 `partial`，四子集存了权重且全部核对通过，才够 `pass`。跨批次复现比单批内部检查更强，值得单独留档。 |
| **v14** | 2026-11-13 | ① 列数 63 → **64**：新增 `subset`（放 `run_id` 之后），台账改为**每子集一张** `runs_<subset>.csv`，`runs.csv` 退役；新增 `SUBSETS.md` 索引（含 `protocol_blob`、行数、sha256、状态）。② `run_id` 格式改成**从产物反推的正则** `^(N0|N1|N1c|N2|N3)_LOEO-<engine>(_s<seed>)?_ev<stop>$`，不再用带时间戳的 `NCM-…`。③ `audit_verdict` 增加 `partial`：指标可复算、权重不可验（无 checkpoint）。④ 登记流程加**出汇总前的 seed 数硬门禁**。⑤ 活规则引用全部改为 `runs_<subset>.csv`；历史变更记录不动，v13 行加一句消歧。 | ① 范围是开放的（后续可能加 DS01/03/04），一张表意味着每加一个子集都要重写证据文件，已封存子集的哈希就不稳定；而规则要求已跑完子集的证据此后一个字节都不动。② `preds_path` 里嵌着 run_id（120/120），改名要动 120 个路径与目录名；改 schema 是零风险。③ DS02 试点批没存权重，若标 `pass` 会被读成完全通过。④ 116/120 行时用中间数据出表，1 seed 的均值与 3 seed 的均值在表里长得一样，只有断言能拦。 |
| **v13** | 2026-11-12 | ① 冻结模板 `max_dev_cycle: 89` / `max_test_cycle: 76` 拆成 `max_dev_flights: 89` + `max_dev_rul: 88`、`max_test_flights: 76` + `max_test_rul: 75`；② `capped_note` 与 §3.5 的「89/76 周期」改为「89/76 飞行，`max(Y)` = 88/75」，并加口径区分说明；③ 顺带修正 `measured_scale.updates_total` 的过期值 240 → **1200**（200 遍 × 6 次/遍）。列数仍 63。 | `unit_lifetime()` 返回的是 `max(Y)`（`data.py:184`），相对三分位的分母也是它（`evaluation.py:359`）；官方标签在最后一个飞行上是 0，所以飞行数 89 对应 `max(Y)` 88。**一个叫 `max_dev_cycle: 89` 的字段，谁读都会拿 89 去和 `rul_cap=125` 比——比得过，但量不对。** `updates_total` 同理：协议改到 200 遍之后它还是 240，是漏改。 |（v14 补注：v13 之前的变更记录里出现的 `runs.csv` 指当时唯一的那张表；v14 起为 `runs_<subset>.csv`。）
| **v12** | 2026-11-12 | ① `coincident` **删掉 `n/a`**，只留 `true`/`false`；§1.6 关闭「N0 是否有内层划分」的活扣。② `N1c` 不进 `runs.csv`，改落 `experiments/ncmapss/diag/runs_diag.csv`（同 63 列，`phase=diag`）。③ `phase` 枚举加 `diag`；§1.6 显式写「test 路径尚未实现，本批不产生 test 行」，并把 test 模型来源写死为「全 6 台 dev 重训、每 seed 一 ckpt」。④ 锚链统一表述为 **13 代**，并注明审计早期写的「五次前移」是错的。⑤ 写入 `REGISTRY_NOTES` 单一写入者规则（owner = 实验结果管理员）。列数仍 63。 | ① 代码对 N0 也产出两个评价点，`n/a` 是不可达值——留着等于给写入端一个错的合法值，审计会放它过去。② `runs.csv` 是冻结网格的证据表，加一个「不进网格」的诊断臂会让主表包含非预注册行。③ 180 行是模型不是现状，不写清会让人以为已经能跑。④ 锚链有缺口就没法证明某个 run 用的是哪版协议。⑤ 两次撞车都是靠事后发现版本号不对才暴露的。 |
| **v11** | 2026-11-12 | ① 新增 `coincident` 列（62 → **63 列**），落地「双评价点必须同时存在且可区分」的机械检查；② `run_id` 增加 `<stop>` 段（`iv`/`f200`），一次训练产出两行，新增 §1.6 说明 120 + 60 = 180 行的行模型；③ 新增 §3.7：作废用 `status=superseded` + `audit_verdict=fail` + `registry_note` 以 `VOID:` 开头，**不新增枚举值**；④ §3.6 按 `9ac1744` 的现行口径重写（200 遍 + 内层早停），此前写的 40 遍已过时；⑤ 冻结锚历史改为**从 git 重建的 13 代**（此前审计只记了五代，中间有缺口）。 | ① 「两列都必须存在」原来是承诺，`coincident` 让它在文件层面可检查——巧合与漏报在表里长得一样；② 双评价点让「一个 run 一行指标」的假设失效；③ 作废各写各的，读者就看不到这个对照存在过；④ 台账与协议正文对齐；⑤ 锚链有缺口就没法证明某个 run 用的是哪版协议。 |
| **v10** | 2026-11-12 | 补记（据 commit `9ac1744`）：`epoch*` 改为取 **5 条内层曲线均值的 argmin**，不再取各折最小值的 median（后者会塌到 1–2 遍）。锚前移到 `9ac1744` / `a3c40c63…`（当前）。 | median of per-split minima 在个别折塌陷时会被拉低，均值曲线稳得多；实测 N2 因此从 5.29 掉到 18.09 的失效形态被消除。 |
| **v9** | 2026-11-12 | 补记（据 commit `7e9415e`）：训练预算从固定 200 遍改为**内层验证早停**——`epoch*` 由 5 条内层曲线决定，外层留出机不再参与选停止点；内层训练上限 400 遍，`patience 12` 记录但不作为停止规则。锚随协议正文前移到 `7e9415e` / `21af549e…`。 | 固定预算只能报「训练停在哪」，报不出「模型能达到什么」；而拿外层留出机早停等于用留出机选模型。 |
| **v8** | 2026-11-12 | 锚前移到 `1c76b37` / `443462ca…`；训练预算 40 → **200 遍**（200 × 6 ≈ 1200 次参数更新，与 C-MAPSS 的 ~1560 同量级）；新增**折内目标标准化**（`target_norm: fold_local_zscore`，`y` 用训练折 mean/std 标准化，预测反变换回周期后才写 `pred.csv`）；审计第 19 条改为核 `epochs_run == 200`；模板 `train_plan.values` 加 `target_norm` / `target_norm_note` | 40 遍只有 240 次更新，四档全部欠训练：fold 1 实测 N0 15.23 / N1 21.71，而同特征线性 ridge 是 5.19；加目标标准化后**同样 40 遍** N0 就到 2.45。800 遍开始记忆化（N2 6.81 → 13.99），故取 200 遍。两项都是跑前按诊断定的，不是看结果调参 |
| **v7** | 2026-11-12 | ① 冻结锚第三次前移到 `e7ad15f` / `35b40d8b…`（`922a2d1` / `ebfbfcb8…` 进 `superseded_anchor`，现共三条）。② 训练预算改为 **`epochs=40`（遍数，对齐 C-MAPSS）、`batch=min(64,N_train)=64` → 每遍 6 次更新、共 240 次**；`selection_rule` 记 `fixed_passes:40`；审计第 19 条按新公式核。新增 §3.6 说明「对齐遍数不对齐步数」。③ 新增两个**诊断字段** `holdout_min` / `holdout_min_epoch`，进 `metrics.json` 不进台账，标 `diagnostic only`。列数仍 62。 | ① 1000 个优化步在单折 380 窗口、batch=380 下等于 **1000 遍**过数据，是 C-MAPSS 40 遍的 25 倍——四档公平但问的问题从「谁预测更准」变成「谁过拟合得慢」。决定泛化的是遍数不是步数，所以对齐量换掉、batch 缩到 64 让一遍不是一次更新。② 不做早停就必须把曲线一起报，否则过拟合被固定终点藏起来。**这是开跑前最后一次锚前移**，之后直到第一批 LOEO 出结果不再动 §4/§6。 |
| **v6** | 2026-11-12 | ① 冻结锚前移到 `922a2d1` / `ebfbfcb8…`（旧 `7f01909` / `4458d9ea…` 进 `superseded_anchor`）。② `analysis_rule.capped_criterion = "inactive (no unit exceeds the cap)"`，新增 §3.5：capped 不并排、不称更严格、差异算标签属性不算模型属性。③ `train_plan.epochs_overridden = 1000 optimizer steps`，`protocol.json` 的 `epochs: 40` 在 DS02 侧不使用；审计加第 19 条按 `ceil(1000/ceil(N_train/batch))` 核对。④ 实测规模（dev 446 飞行 / test 202 飞行 / 单折 380 窗口 / unit 14 每周期 1,912 行）写进冻结模板 `measured_scale`。列数仍 62。 | 两条都是**按实测文件、在跑之前**做的决定，不是看结果调参：40 epochs 在 380 窗口上等于 40 个梯度步，等于没训练；capped 因最长 dev 仅 89 周期而处处等于 raw。**故意不加 `n_train_windows` 列**——开跑在即，加列会让写入端与台账错位，该值从 `config.json` 取即可。 |
| **v5** | 2026-11-12 | ① **N0 的描述按代码改**：`N0 = age-only 小 MLP（4,353 参数），无 seed 轴（固定 seed 101，每折 1 格）`；§1.1/§1.2 两处「无参数、无初始化」的说法删掉，`selection_rule` 改为 N0 记 `fixed_epochs:40`、`ckpt` 要求存或写明为何不存。② **`preds.csv` 固定 5 列** `unit, cycle, y_true_raw, y_true_cap125, y_pred`，**删掉裸 `y_true`**，`engine_id`/`flight_cycle` 统一成 `unit`/`cycle`；审计加第 18 条。列数仍 62。 | ① 原文三句（无参数、无初始化、无训练随机性）和代码都不符，而**格数 6 是从「无 seed 轴」来的，不是从「无参数」来的**——说错理由，以后有人会照着「无参数」去改代码。② 裸 `y_true` 装的是 `min(raw,125)`，**是一个默认走 capped 的隐式口径**，和 C-MAPSS 那晚「池化当发动机级」是同一类错误：有个裸名字在偷偷兜底。D043 定 raw 为主口径，留这么一列迟早有人拿它当主。 |
| **v4** | 2026-11-12 | 新增 `protocol_blob`（61 → **62 列**），审计清单加第 17 条；冻结模板加 `freeze_commit` / `protocol_blob` / `known_protocol_blob_at_freeze` / `n3_scope`。**冻结锚随后整体前移到 `7f01909` / `4458d9ea…`**（旧 `5b228ad` / `6c184bbf…` 作废），并把「N3 = MS+SA，`hi_source=none`，本批不含 HI 臂」写死在 §3.3。 | ① **commit 可以被 amend 或 rebase，blob 不会。** 协议冻结若只锚在 commit 上，一次 amend 就能让「冻结」失效而不留痕迹；记 blob 后一条 `git rev-parse` 就能核。② **blob 对换行符敏感**：`.gitattributes` 把 `*.md/*.json/*.csv` 定为 `eol=lf`，Windows 上 CRLF 重存会让 blob 变而内容看起来没变，所以核验必须在仓库内跑 `git rev-parse`，不许拿工作区文件算 sha256。③ 锚前移是因为协议在开跑前修了三处（主指标标注、相对分段分母、HI 范围收窄）；④ N3 不含 HI 必须写死，否则以后有人会从台账反推「N3 测过 HI」，那就是伪造证据。 |
| **v3** | 2026-11-12 | 分段误差 12 列重建为 **3 段 × 2 规则（relative / absolute）× 2 口径（capped / raw）**，指标只留 RMSE；去掉 v2 的分段 MAE（改留在每次 run 的逐段 CSV）；`seg_boundaries` 改为同时装两套边界 `{"relative":[0.3333,0.6667],"absolute":[75,25]}`；审计清单加第 16 条。列数仍 61。 | **v2 的 12 列漏了 segment_rule 这一维**——相对/绝对双规则是本轮新增的核心，漏了两套边界只能留一套，等于把新增的东西丢掉。另一条：测试集每机只评最后一个飞行，相对分段若用被评估子集的 `block.max()` 会整体退化成 `early` 且不报错，所以归一化尺度必须来自该机完整寿命（`unit_lifetime`），缺了要报错。 |
| **v2** | 2026-11-12 | `runs.csv` 从 50 列扩到 **61 列**：新增 `train_plan_sha256`；新增 `kmeans_fit_units`、`kmeans_seed`；新增 `freeze_source`、`freeze_source_sha256`；分段误差从 6 列（仅 RMSE）扩到 **12 列**（3 段 × 2 口径 × RMSE+MAE），并把原 `seg_*_raw` 更名为 `seg_*_rmse_raw` 以区分指标。审计清单从 12 条扩到 **15 条**。 | ① 训练计划与工况编码在 2026-11-12 冻结（`c69db46` 待推），台账必须能证明「跑的是冻结的那份」，所以记 `protocol.json` 的哈希而不是手抄数值——**数值抄错看不出来，哈希对不上能看出来**。② `n_cond=6` 在 DS02 上重新定义（`W` 上 k-means，只在当前折训练机拟合），必须留拟合机清单才能查「DSPT 是否偷吃信息」。③ N3 必须从冻结文件读取而非手抄，否则「N3 是冻结的」无法证明。④ 分段误差只报 RMSE 会让「误差大小」和「偏差方向」混在一起，加 MAE 是补齐。 |
| v1 | 2026-11-12 | 初版，50 列 | 建立台账骨架 |

> 改 schema 必须同时改本表并跳版本号。只改 `SCHEMA.md` 不改这里，等于让后来人不知道列是怎么来的。

---

## 1. 规则

- **单一写入者（2026-11-12 裁定）**：`docs/paper_b_ncmapss/registry/` 下四个文件（`runs_<subset>.csv`、`SUBSETS.md`、`REGISTRY_NOTES.md`、`SCHEMA.md`、`FREEZE_TEMPLATE.json`）**只由实验结果管理员写**。其他人**提 patch 或口述**，不直接改文件。此前发生过两次并发撞车（v9/v10 未进变更记录），靠事后发现版本号不对才发现，代价太大。
- **一个 run 一行，只追加不删。** 作废用 `status=superseded`，不删行。
- `run_id` 格式见 [SCHEMA.md](SCHEMA.md) §2。**同一 run_id 不得重跑**；重跑换时间戳，
  并在 `registry_note` 写「重跑原因 + 前一次 run_id」。
- **`launch/` 用 `O_CREAT|O_EXCL` 独占创建**，写完回读自检，带 `written_by` 与 `process_start_utc`。
  手写或复制上一份改字段的 launch 视为伪造。
- **N3 冻结**：MS/SA/HI 开关用 C-MAPSS 已冻结值，**不许因 DS02 结果改**。改了就不是 N3。
- **测试机 `[11,14,15]` 只在冻结清单提交后评估**，`eval_start_utc > freeze_ts_utc` 才合规。
- **最终测试模型 ≠ 任何一折模型**：测试集只用「全部 6 台开发机重训」的 9 个 checkpoint；
  LOEO 的 54 个 checkpoint 只用于开发集报告，**不参与任何测试评价**。冻结清单里出现 LOEO checkpoint 即冻结无效。
- **测试集只有 3 台机**：每格只能报 seed 均值 ± seed sd（n=3 seeds, 1 unit），
  **不得写跨机 95% CI**；三台机双侧符号检验最小 p = 0.25，不能表述为「显著优于」。
- **`audited_by` 只放人名**，判断放 `audit_verdict`，溯源信息放 `registry_note`。
- **训练计划冻结**：四档共用 `paper_b/configs/protocol.json`（40ep / p12 / b=min(512,N) / lr1e-3 / wd1e-4 / d64 / L2 / do0.2）。
  只许改输入宽度（51 列）。哪一档不收敛是要报的发现，**不是改超参的许可证**。台账记 `train_plan_sha256`。
- **工况编码 `n_cond=6`**：`W` 上 k-means(6)，**只在当前折训练机拟合**；`kmeans_fit_units` 出现留出机或测试机 → 该 run 作废。
- **N3 冻结来源**：`paper_b/results/FREEZE_ABLATION_SEED101.json` + `paper_b/configs/protocol.json`，
  代码直接 `json.load`，哈希记 `freeze_source_sha256`；**DS02 侧不许手抄参数值**。
- **口径列不许混**：`_unit`（逐机等权）≠ `_flightpooled`（逐飞行段混算）；`_raw` ≠ `_cap125`。
  引用任何数字必须带上这四个限定中的两个。
- **聚合计数必须带分组明细**（2026-11-13 裁定，与下面那条哈希规则同级）：正文、结果文档、
  清单字段里任何聚合计数，**必须把分组明细一起写，且明细相加等于总数**——
  `14/24（DS02 5/6 + DS05 3/6 + DS06 4/6 + DS07 2/6）`，不许写裸的 `20/24`。
  **没有分组明细的一律退回。**
  理由：同一个数换一个分母就变成另一个结论，而两个版本在正文里长得一模一样。
  已经因此错了三次——`ρ=1.0`（该是 0.943）、`16.61`（1 个 seed 当 3 个）、`20/24`（该是 14/24），
  **三次都发生在「手里已有正确数据、写结论时重新口述一遍」的环节**。
  光提醒自己小心没有用：裸聚合必然要求读者复算，而人复算会错；摊开分子分母，错不可能发生。
- **冻结清单的哈希规则**：`freeze_ts_utc` **必须在哈希内**，只排除 `eval_start_utc` 与哈希自身；
  规范化形式固定为 `ensure_ascii=False` + `separators=(',',':')` + `sort_keys=True` + UTF-8。
  签名器与闸门必须同形，否则已提交的清单会被判成被改过。复算：`python scripts/nc_freeze.py verify <清单>`。
  细则见 [SCHEMA.md](SCHEMA.md) §6 与 [freeze/INDEX.md](freeze/INDEX.md)。

## 2. 有效样本数

- 独立单位是**发动机**（开发集 6 台），`independent_unit=engine`。
- `6 折 × 3 seed = 18 个数字，但不是 18 个独立样本`；跨 seed 区间用 `df = n_engines − 1`。
- seed 只衡量优化随机性，不能补足发动机数量。三台测试机全改善也不足以证明总体优越。

## 3. 同步方向

- 台账（`runs_<subset>.csv`、本文件、`freeze/`）以**本地仓库为准**，单向镜像到 3090 同一相对路径。
- `launch/`、`close/` 在 3090 上产生，单向拉回本地。
- 两边都不双向写。C-MAPSS 台账出现过两个写入者同时改同一文件，这条就是为了避免重演。

## 4. 写回守卫（哈希乐观并发 + 互斥）

对着 `HEAD` 比 blob **不能**拦住第二个写入者——两个写入者读到同一份工作区时看到的是同一个 HEAD，
两边都会通过。真正的互斥靠锁。写回按三层执行：

1. **独占锁**：`runs_<subset>.csv.lock` 以 `O_CREAT|O_EXCL` 创建，整段读—改—写持有；抢不到直接 abort。
2. **双锚校验**：写入前重新比对「读到的 blob」与「HEAD blob」，任一变化即 abort；
   HEAD blob 记入日志（落后于工作区时可见）。
3. **写后回读**：`os.replace` 后重读，字节与刚写的不一致则留 `.conflict` 副本并 abort。

## 5. 登记记录

（空。每次登记追加一段：日期、run_id 列表、来源 launch/close、审计结论、异常。）

## 6. 删除/作废记录

（空。格式：`run_id | 原因 | 裁决人 | 日期`。）
