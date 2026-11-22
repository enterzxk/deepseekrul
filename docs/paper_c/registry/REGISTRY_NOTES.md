# Paper C registry notes（append-only）

状态：v1，2026-11-22
适用范围：`docs/paper_c/registry/` —— 与 `docs/paper_b_hc/registry/`、`docs/paper_b_ncmapss/registry/` 各自独立，
不共享文件、不共享 run_id 空间、不互相引用行。

写入人：本目录只由实验结果管理员写；`launch/`、`close/` 由 runner 写。

---

## 0. 变更记录

| 版本 | 日期 | 改了什么 | 为什么 |
|---|---|---|---|
| **v1** | 2026-11-22 | 建目录：45 列 schema、`grid.json` + `gen_queue.py` 生成 30 格队列、空 `runs.csv`、`check_registry.py` 九项断言。基线钉在 `fixed40`（30 格），`author_limit` 10 行明确不得配对。 | 见 §1–§4 |

---

## 1. 这 30 格是什么

`5 方法 × {FD001, FD004} × {101,202,303} = 30 格`，全部是 **+CADF**，V0 基线不重跑。

- 基线不重跑：V0 已完成（94 次物理训练、168 个条件），且**在另一条分支**
  （`codex/paper-b-open-source-reproduction`，`d7bd9c17`）。台账用 `baseline_ref` 指过去，
  值从基线台账**读**，不手抄。
- 只用 80/20：经理裁定走计划书 1.2 的「或 3 paired seeds」，V0 基线直接复用，
  过门后再补 5-fold 终表。`split_protocol` 列写死这一点，避免以后有人把 5-fold 的数和它并排。
- 本批只跑 FD001/FD004。FD002/FD003 是 Gate 1 PASS 之后的补齐动作，不是现在。

---

## 2. 两个未定项（**开跑前必须解决**）

### 2.1 `K0` 用绝对还是按比例 —— 队列现在只有一种

`grid.json` 的 `variant_tags` 目前是 `["k0a"]`（绝对 `K0=15`），所以队列是 30 格。

经理指出 `K0=15` 是绝对周期数，而 5 篇的窗口只有 30/45/50——窗口 30 时 K0 吃掉一半，
基线就变成「窗口前半段」而不是健康期基线。**两种都要跑，别混。** 所以一旦预注册定了，
改 `grid.json`：

```json
"variant_tags": ["k0a", "k0p"]     # 或 ["k0p"]、或加 "eng"
```

再 `python docs/paper_c/registry/gen_queue.py` —— 30 格自动变 60 格（或 90）。
**不要手改 `run_queue.csv`**，它每次都会被重新生成。

### 2.2 预注册文档还没提交

`docs/P2_CROSS_ARCH_PREREG.md`（郑总的活）**目前不存在**。
所以队列里每一行的 `launch_allowed` 都是 `false`，`note` 写着这条依赖。

这不是形式主义：Gate 1 的判据（4/6 骨干、+0.40、胜率 60%、EOL 不退化）必须先写死，
否则 30 格跑完再定判据，就是事后挑标准。

---

## 3. 检查器验证过会真的拦

`check_registry.py` 写完先用 9 个用例打过一遍（正例 2 个、反例 7 个），全过：

```
clean ok row                               exit=0 want=0 OK
wrong delta sign                           exit=1 want=1 OK   delta 9.9 != baseline - cadf
TEST filled while sealed                   exit=1 want=1 OK   test_rmse is filled
wrong baseline condition                   exit=1 want=1 OK   source_condition='author_limit', need 'fixed40'
subset outside grid                        exit=1 want=1 OK   run_id FD002 不在网格内
bad status enum                            exit=1 want=1 OK   status='done'
duplicate ok cell                          exit=1 want=1 OK   同键两条 ok
superseded duplicate is allowed            exit=0 want=0 OK
baseline value disagrees with ledger       exit=1 want=1 OK   10.0 != 11.100379622
```

**第一版有个真 bug**：我把「非 ok 行与 ok 行同键」也判成错，于是 `superseded` 更正行被拒——
和这条规则本身矛盾。负例测试把它抓出来了。现在的规则是：**同键只允许一条 `ok`；
`failed` 重试与 `superseded` 更正都可以共存**。

---

## 4. 三条硬规则（来自已发生的教训）

1. **TEST 封存是列级的**：`status=ok` 且 `stage != Terminal` 时，`test_rmse` / `test_score`
   任何一格非空即拒。Paper C 的规则是「开发期不得读取官方 test 标签」，所以这条必须由检查器挡，
   不能靠自觉。
2. **基线口径不能混**：`fixed40` 30 格 vs `author_limit` 10 行。混了得到的 delta
   是两个变量的差。检查器按 `source_condition` 拒。
3. **delta 符号写死在 schema 里**：`基线 − 本格`，正 = CADF 更好。检查器逐行验证恒等式，
   防止有人把方向写反后一路错到 Gate。

---

## 5. 远端现状（2026-11-22 22:20 实测）

```
3090: 分支 ncmapss-ds02 @ 51307439（**没有 paper_c**），/home 可用 92G（守卫 ≥30G 通过）
      GPU0 677MiB/24576MiB 0%，GPU1 15MiB/24576MiB 0%（两卡空闲）
      无 docs/paper_c/registry，无 results/paper_c，无 P2_CROSS_ARCH_PREREG.md
```

**这不是我能解决的阻塞**：Paper C 的代码和 5 篇基线在不同分支上，
3090 现在 checkout 的是 N-CMAPSS 分支。开跑前需要把 `paper_c` + 基线代码落到同一棵树上
（或起独立 worktree），并把本 registry 同步过去。登记时以**台账每行的 `git_sha`** 为准，不看日志。
