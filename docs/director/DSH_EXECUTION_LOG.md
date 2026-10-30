# DSH 执行日志（native goal，会话 codex-rul-native-goal-20261022）

追加式记录；详细证据见 `docs/director/DSH_PROJECT_AUDIT.md`、`results/dsh/order03/REPORT.md`、`results/dsh/control.json`。

## Round 0（首次用户回合）
- 创建目标 `goal-8f36b594-6e02-4320-bb90-12be7b93755b`（默认 256 轮），`get_goal` 确认 active/armed；写交接文件。
- 未审查仓库、未连服务器。

## Round 0b（传输层重启）
- `get_goal` 得到 `activation=disarmed`；对同一目标 `update_goal resume` → revision 2，active/armed。
- 未创建重复目标。

## Round 1（本回合，2026-10-22 00:55–01:35 +0800）
1. 接管原审查 worker 的 66 条 tool_result，导出 `results/dsh/audit_raw/01–67.txt` 与索引。
2. 核对 GitHub 快照 4 个 blob SHA：全部与自报值相符；`model.py`、两份 ORDER_03 报告与本地字节一致；`run_candidate.py` 不一致（GitHub 为第三版本）。
3. 服务器实测（`-o UpdateHostKeys=no`）：FD002 12 单元完成；M1_global 3/3 崩于 `build_cond_abs`；FD001 全崩于 `run_candidate.py:650`；发现四代启动器并存、M4c_noid 与 M4 同配置、RACDR 死参数 704 个、`use_sensor_id=False`。
4. 写 `results/dsh/analysis/aggregate_order03.py` 并在服务器用 a40_env 运行：per-seed / pooled / macro / 发动机聚类 bootstrap（20k）全部重算，输出 `results/dsh/order03/{cells.csv,group_summary.csv,aggregate.json}`。
5. 关键结果：matched M4−M0 = −1.975 [−4.047,+0.291]（跨零）；M4−M3 = −0.982 [−2.096,+0.131]；仅 Protocol B M4−M0 显著；matched severe 0.250 vs 0.218（更差）；Protocol B late 0.521 超 0.52 门禁；M4b（n=3）16.21；M4c 与 M4 三位小数逐位相同。
6. 文档：`DSH_PROJECT_AUDIT.md` 追加“接管复核 A–F”；`DSH_EXECUTION_PLAN.md` 追加阻塞与 v3 计划；`results/dsh/order03/REPORT.md` 主表。
7. 未终止 `o3d.sh`、未改训练源码、未推送 GitHub、未做防御性测试。

## 发现的环境事实
- 本工作区存在第二个 DSH 会话 `desktop-ptc-deepseek-rul`（goal `goal-96c7053b…`），先写入了 audit/plan/control 并启动 o3d；本 session 只做增量，不覆盖其证据。
- 服务器 `o3d.sh` 在 17:23Z 已是新 PID（1387427），FD001 单元继续崩溃。

## Round 2（2026-10-22 01:40–02:05 +0800）
1. 服务器状态：`o3d.sh` 已退出；另一会话继续改代码（`run_candidate.py` 依次出现 `2393dc8e`、`9074d6ca`；`model.py` 变为 `8331a355`）。
2. 代码复核发现：
   - `gate_mode`（evidence/ones/half/const）在 `model.py` 只被构造、forward 从不读取 → `RACDR_fixed1/half/const` 与 RACDR 完全等价，是假消融；
   - 新 FD001 `build_windows_abs/matched` 自行重算证据（常量 age、trend 代替 jitter）→ 训练/评估特征偏移；
   - matched 单工况用一次性顺序 rng，与 FD002 的逐机 seed+unit 不一致。
3. 生成冻结 runner `scripts/frozen/run_candidate_v3_8fce61ff.py`：复用 `_single_evidence`、matched 逐机 rng、ROOT 自适应、metrics 写 `code_sha256/command/dataset`；连同 `frozen_src/candidates/model.py`、`frozen_scripts/baselines.py` 一起冻结，`py_compile` 通过。
4. 聚合旧 FD001 开发数据（n=2–3）：M0 14.084、M3 12.215、M4 12.879、M4b/M4c 12.387（同配置）；M4−M3 = +0.661 [−1.740,+3.256]，跨零。标注 pre-fix，不作结论。
5. 写 `results/dsh/freeze.json`（runner/model/baselines/数据哈希、协议、环境）并上传服务器；写 `docs/director/DSH_AUDIT_ADDENDUM.md`（gate_mode 死参数、特征偏移、版本时间线、M4b FD001 同配置）。
6. 启动 Level B：`split_seed=2027`、seeds 2029/2030/2031、FD001+FD002、M0_tf/M4_racdr，12 单元，`xargs -P 2`；FD001 s2029 的 M0 已完成（val 12.21），RACDR 在训。启动时重复启动了一个 RACDR s2029（同配置），已定向 kill 重复进程，保留一个。
7. 未终止另一会话、未覆盖其文档、未推送 GitHub、未做防御性测试。

## 下一步（Round 3）
1. 收 Level B 12 单元（不忙轮询），用冻结脚本自带的 `code_sha256` 核对来源。
2. 写 Level B 汇总：matched 主表、Protocol C、late/severe、发动机聚类 bootstrap（20 引擎/52 引擎）。
3. 按 LEVEL_B 门禁判定：① matched RMSE 优于 tf；② FD001 [0,25) severe ≤ 1.1×tf；③ 记录结果，不得回改候选。
4. 把 Level B 结果写入 `results/dsh/order03/REPORT.md` 与 `DSH_AUDIT_ADDENDUM.md`，给出诚实结论与下一阶段建议；通过或不通过都冻结归档。

## Round 3（2026-10-22 02:00 +0800，进行中）
1. Level B 首次启动后被并行会话的 pkill 误杀（launcher 与 xargs 均终止），已重命名 runner 为 `scripts/frozen/levelb_runner_8fce61ff.py`（字节相同、SHA 不变）并重启；skip 逻辑避免重跑已完成的 2 个单元。
2. 已记录复验切分与开发切分的发动机重合：FD001 20/20 中重合 7 台，FD002 52/52 中重合 6 台；因此只能称“预登记复验”，不是独立盲测。
3. 官方 test 文件存在但代码从不引用；FD003/FD004 的 test/RUL 仍为 `chmod 000`。
4. 已启动后台等待任务（job pwsh-de8f2f1fb621422b），12/12 完成后自动跑 `aggregate_levelb.py`；下一轮读取结果并判定门禁。

## Round 4（2026-10-22 02:45–02:55 +0800）：Level B 完成并交付结论
1. 12/12 单元完成（FD001/FD002 × M0_tf/M4_racdr × seeds 2029/2030/2031），全部 `code_sha256=8fce61ff…`，保存 `best_model.pt`。
2. 结果（matched，均值±sd）：FD001 M4 8.735±0.841 vs tf 9.876±1.120；FD002 M4 9.762±0.574 vs tf 12.859±0.476。
3. 配对 bootstrap（M4−tf）：FD002 matched −3.092 [−5.465,−1.000] 显著；FD001 matched −1.157 [−3.420,+1.391] 跨零；FD001/2 Protocol B 均显著。
4. 晚报：FD001 late 0.367 vs 0.550、severe 0.117 vs 0.233；FD002 late 0.455 vs 0.564、severe 0.096 vs 0.231；均过 0.52 门禁与 severe 门禁。
5. 交付 `docs/director/DSH_STAGE_CONCLUSION.md`；`results/dsh/order03/REPORT.md` 增 §7；`control_native_goal.json` 更新。
6. 限制已披露：tf 基线早停（FD002 best_epoch 2–6）、复验非独立盲测（发动机重合 7/20、6/52）、n=3、旧矩阵不可重跑、真门控消融未实现。
7. 遗留缺口：FD001 开发 M4 s2029、M1_global 两个 seed；不影响 Level B 结论，列为下一阶段。
