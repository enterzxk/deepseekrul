# Paper C registry notes（append-only）

状态：**v2**，2026-11-22
适用范围：`docs/paper_c/registry/` —— 与 `docs/paper_b_hc/registry/`、`docs/paper_b_ncmapss/registry/` 各自独立，
不共享文件、不共享 run_id 空间、不互相引用行。

写入人：本目录只由实验结果管理员写；`launch/`、`close/` 由 runner 写。

---

## 0. 变更记录

| 版本 | 日期 | 改了什么 | 为什么 |
|---|---|---|---|
| **v1** | 2026-11-22 | 建目录：45 列 schema、`grid.json` + `gen_queue.py` 生成 30 格队列、空 `runs.csv`、`check_registry.py` 九项断言。基线钉在 `fixed40`（30 格），`author_limit` 10 行明确不得配对。 | 见 §1–§4 |
| **v2** | 2026-11-22 | **60 格两 arm**（A 段 30 V0 + B 段 30 +CADF），52 列；`segment`/`arm` 分列，两条 `run_id` 格式；主臂 `k0a`→**`k0p`**；B 段基线改成**本批 A 段那一格**；`gate/alpha` 轨迹三列 + 开闸 epoch 为 B 段必填；**去重键补 `base_model`**（v1 的键是错的）；检查器 12 项、退出码三档；新增 `schema_cols.py`、`selftest_registry.py`（23 用例；同日把 B 段闸门从「台账里重算 `|Δ| ≤ 0.2`」改成「记录 `gate1.py` 的 verdict + 判据脚本哈希」，因为 0.2 打在 argmin 上会假 ABORT）、`FIXED40_ANCHORS.csv/.md`、`FREEZE_20261122.json`。 | 经理裁定 60 格（跨机器漂移）+ 预注册 v3 `k0p` |

---

## 1. 这 60 格是什么

```
A 段 30 格 = 5 方法 × {FD001, FD004} × {101,202,303} × V0
B 段 30 格 = 5 方法 × {FD001, FD004} × {101,202,303} × +CADF(k0p)
```

**为什么 V0 也要重跑**（v1 的「复用 V0」已撤销）：V0 是 A40 + Python 3.9 + torch 2.5.1 跑的，
3090 环境不同，RMSE 差 0.1–0.3 很正常，而 Gate 1 要判的效应就是这个量级。拿跨机器的数当基线，
等于用比效应还大的噪声做配对。60 格买到的是**同机、同环境、同种子、同 40 轮**的干净配对。

A 段不是「顺手复现」，它是一道**闸门**：A 段对不上锚点，B 段 30 格的钱就省下来。

---

## 2. v1 的两个未定项，现在都有答案

### 2.1 `K0` 用绝对还是按比例 → **`k0p`（`ceil(0.2·L)`）**

`k0a=15` 在 L=30 的窗口里占 50%、L=45 占 33%——跨骨干比的根本不是同一个模块。`k0p` 让窗口占比
恒为 20%，SOTA 五篇内部 Δ 的量级才可比。**5 篇 SOTA 只跑 `k0p`，不跑 `k0a`**。

`grid.json` 现在是 `"variant_tags": ["k0p"]`，队列 60 行，逐格 `k0_resolved`：

| 方法 | span | `k0p` | 备注 |
|---|---:|---:|---|
| aidgn | 50 | 10 | |
| two_phase | 45 | 9 | |
| dual_mixer | 30 | 6 | |
| ffve | 30 | 6 | |
| fcstgnn | 30 / 50 | **6 / 10** | FD001–3 是 30、FD004 是 50 → 「fcstgnn 过了」必须逐子集说 |

**B_min 臂不在本批**：它沿用 v1.2 的 `k0a=15`（L=50 占 30%），已跑完、FD004 −0.061 不过门。
**不得因结果换 `k0p` 重跑**（那是重抽折叠的同类错误），Gate 1 若只差它一票，如实写成限制。

### 2.2 预注册 → 已落地 `5255b1b6`

`docs/P2_CROSS_ARCH_PREREG.md` @ `5255b1b6`（215 行，sha256 `1f4be02274db…`）。
A 段 `launch_allowed` 因此转 `true`；B 段仍封，闸门是 `grid.json::a_segment_reconciliation.verdict`
（`pending` → `pass` 只能由 `gate1.py` 的输出改写，台账不自己判）。

**冻结第 ③ 步由本目录完成**：`FREEZE_20261122.json` 记录预注册 / `paper_c/wrapper.py` /
`frozen_v0_fixed40.json` 的 sha256，**全部从已提交 blob 算**，不从工作树算。
`wrapper.py` 的哈希 `d3e4dbbc…` 与牢莫在群里报的一致（8000 B），是独立复算。

> 踩过的坑：用 PowerShell 把 `git show` 管道给 python 算 sha256，中文会被管道按 GBK 转码，
> 得到 `4ccc0db6…` 而不是真实的 `1f4be022…`。**算哈希只能用 `subprocess(capture_output=True)`
> 取字节**，不能过 PS 文本管道。

---

## 3. 检查器：从 9 个口头用例变成 21 个可重跑用例

v1 的九项断言是用 9 个**临时脚本**验的，没进库——所以「检查器会真的拦」在文档里是一句声明。
现在 `selftest_registry.py` 就是那句话的可执行版本，**21/21 全过**：

```
30 segment-A cells, no B yet                        exit=0 want=0 OK
30 A + 1 B                                          exit=0 want=0 OK
argmin delta 0.5 does not block (verdict is the gate) exit=0 want=0 OK
A missing one cell + a B result                     exit=2 want=2 OK
B with the reconciliation still pending             exit=2 want=2 OK
B with a record that says fail                      exit=2 want=2 OK
B with a judging-script hash mismatch               exit=2 want=2 OK
B paired against the cross-machine ledger           exit=1 want=1 OK
B delta sign reversed                               exit=1 want=1 OK
B with TEST filled while sealed                     exit=1 want=1 OK
B without the gate/alpha trace                      exit=1 want=1 OK
B gate never opened but in the Gate table           exit=1 want=1 OK
B gate never opened, kept out of the Gate table     exit=0 want=0 OK
duplicate ok cell                                   exit=1 want=1 OK
same variant/subset/seed, different model           exit=0 want=0 OK
superseded correction shares the cell               exit=0 want=0 OK
bad status enum                                     exit=1 want=1 OK
run_id outside the grid                             exit=1 want=1 OK
ok row whose queue cell was never authorized        exit=1 want=1 OK
k0p resolved to 15 instead of 10                    exit=1 want=1 OK
epochs_run above the fixed40 ceiling                exit=1 want=1 OK
prereg commit no longer holds the prereg            exit=2 want=2 OK
header lost a column                                exit=2 want=2 OK
23/23 cases behaved as expected
```

> **第 3 条是「必须不拦」**：argmin 差 0.5 时台账放行。军师指出 0.2 不能打在 argmin 上——
> 曲线上 sd≈0.94，argmin 漂 0.4 是必然（zxk 首格实测：argmin −0.398、曲线水平 −0.125）。
> 把 0.2 打在 argmin 上，A 段跑完那一刻会给出一个**假的 ABORT**，而假的原因会被记成
> 「环境不一致」。现在 0.2 只在 `gate1.py` 的曲线水平与逐机秩上；台账只核
> 「30 格齐 + verdict=pass + 判据脚本哈希」。第 11 项因此不再需要那 5 格预算截断例外。

### 3.1 v1 里一个**会真的炸**的 bug（在它炸之前抓到了）

v1 的去重键是 `(baseline_version, cadf_variant, subset, seed)`——**漏了 `base_model`**。
30 格一落盘，同一 (子集, seed) 的 5 个方法会互相判成「重复 ok 格」，一次报 4 条。
它没在 v1 炸，只因为 `runs.csv` 当时是空的。

现在的键是 `(baseline_version, arm, base_model, cadf_variant, subset, seed)`，自测里
「same variant/subset/seed, different model」这条用例专门守它（旧键下会 exit 1，现在 exit 0）。

### 3.2 三个用例本身写错过（也是自测抓的）

1. B 段的用例全在「对账还没记录」的 grid 上跑，于是每个都先被第 11 项拦成
   exit 2，测的根本不是它要测的东西 → 加了 `grid_open.json` 夹具。
2. 「epochs_run 超上限」的用例用 `dict(a_ok[0])` 追加，先撞「重复格」→ 改成**替换**那一格。
3. 「预注册哈希漂移」用例改的是 `grid.json::prereg.commit`，而第 12 项当时只读 FREEZE 里的
   commit——**用例根本碰不到被测代码**。这暴露了第 12 项的一个真缺口：grid 指向的预注册
   commit 与冻结记录不一致时不会报。现在第 12 项显式比对两者。

---

## 4. 四条硬规则（来自已发生的教训）

1. **TEST 封存是列级的**：`status=ok` 且 `stage != Terminal` 时，`test_rmse` / `test_score`
   任何一格非空即拒。Paper C 的规则是「开发期不得读取官方 test 标签」，必须由检查器挡。
2. **两段的基线不能混**：A 段配冻结台账（`fixed40`），B 段配**本批 A 段那一格**。
   B 段指回冻结台账 = 检查器拒。`author_limit` 10 行永远不得配对。
3. **delta 符号写死**：`基线 − 本格`，正 = 本格更好。逐行验证恒等式。
4. **gate 从未打开 ≠ 模块无效**：`gate_open_epoch=none` 的行可以登记，但 `in_gate_table`
   必须是 `false`。检查器拒「没醒却进 Gate 表」。

---

## 5. 远端现状（2026-11-22 23:40）

```
origin/research/paper-c-cadf  已含 5 篇 adapter + fixed40 30 格 × 7 文件（210 个）
                              + paper_c/{wrapper,gate1} + 预注册 5255b1b6
3090：zxk 走本地打包落 180 个验证文件（origin 在该机 TLS 中断，fetch 挂住）
      /home 可用 92G（守卫 ≥30G 通过），两卡空闲
```

`frozen_v0_fixed40.json` 是牢莫提交进本目录的唯一一个他人文件（`68551059`），
保留为只读参照物；锚点权威件是 `FIXED40_ANCHORS.csv`，两者 30 格 `val_engine` 交叉核对
**最大差 4.6e-07**（他 JSON 保留 6 位小数），30/30 一致。
