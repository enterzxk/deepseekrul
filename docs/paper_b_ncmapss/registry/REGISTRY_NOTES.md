# N-CMAPSS DS02 registry notes（append-only）

状态：进行中
最后更新：2026-11-12 20:40 (+08)
适用范围：`docs/paper_b_ncmapss/registry/` —— **与 C-MAPSS 的 `docs/paper_b_hc/registry/` 完全独立**，
不共享文件、不共享 run_id 空间、不互相引用行。

写入人：`runs.csv` 只由实验结果管理员写；`launch/`、`close/` 由 3090 上的训练进程写。

---

## 1. 规则

- **一个 run 一行，只追加不删。** 作废用 `status=superseded`，不删行。
- `run_id` 格式见 [SCHEMA.md](SCHEMA.md) §2。**同一 run_id 不得重跑**；重跑换时间戳，
  并在 `registry_note` 写「重跑原因 + 前一次 run_id」。
- **`launch/` 用 `O_CREAT|O_EXCL` 独占创建**，写完回读自检，带 `written_by` 与 `process_start_utc`。
  手写或复制上一份改字段的 launch 视为伪造。
- **N3 冻结**：MS/SA/HI 开关用 C-MAPSS 已冻结值，**不许因 DS02 结果改**。改了就不是 N3。
- **测试机 `[11,14,15]` 只在冻结清单提交后评估**，`eval_start_utc > freeze_ts_utc` 才合规。
- **`audited_by` 只放人名**，判断放 `audit_verdict`，溯源信息放 `registry_note`。
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
