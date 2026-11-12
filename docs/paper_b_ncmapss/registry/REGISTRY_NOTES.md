# N-CMAPSS DS02 registry notes（append-only）

状态：进行中
最后更新：2026-11-12 20:40 (+08)
适用范围：`docs/paper_b_ncmapss/registry/` —— **与 C-MAPSS 的 `docs/paper_b_hc/registry/` 完全独立**，
不共享文件、不共享 run_id 空间、不互相引用行。

写入人：`runs.csv` 只由实验结果管理员写；`launch/`、`close/` 由 3090 上的训练进程写。

---

## 0. 变更记录

| 版本 | 日期 | 改了什么 | 为什么 |
|---|---|---|---|
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

## 2. 有效样本数

- 独立单位是**发动机**（开发集 6 台），`independent_unit=engine`。
- `6 折 × 3 seed = 18 个数字，但不是 18 个独立样本`；跨 seed 区间用 `df = n_engines − 1`。
- seed 只衡量优化随机性，不能补足发动机数量。三台测试机全改善也不足以证明总体优越。

## 3. 同步方向

- 台账（`runs.csv`、本文件、`freeze/`）以**本地仓库为准**，单向镜像到 3090 同一相对路径。
- `launch/`、`close/` 在 3090 上产生，单向拉回本地。
- 两边都不双向写。C-MAPSS 台账出现过两个写入者同时改同一文件，这条就是为了避免重演。

## 4. 写回守卫（哈希乐观并发 + 互斥）

对着 `HEAD` 比 blob **不能**拦住第二个写入者——两个写入者读到同一份工作区时看到的是同一个 HEAD，
两边都会通过。真正的互斥靠锁。写回按三层执行：

1. **独占锁**：`runs.csv.lock` 以 `O_CREAT|O_EXCL` 创建，整段读—改—写持有；抢不到直接 abort。
2. **双锚校验**：写入前重新比对「读到的 blob」与「HEAD blob」，任一变化即 abort；
   HEAD blob 记入日志（落后于工作区时可见）。
3. **写后回读**：`os.replace` 后重读，字节与刚写的不一致则留 `.conflict` 副本并 abort。

## 5. 登记记录

（空。每次登记追加一段：日期、run_id 列表、来源 launch/close、审计结论、异常。）

## 6. 删除/作废记录

（空。格式：`run_id | 原因 | 裁决人 | 日期`。）
