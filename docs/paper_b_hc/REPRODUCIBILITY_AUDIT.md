# Paper B 可复现性审计（REPRODUCIBILITY_AUDIT）

最后更新：2026-11-12 · 维护：实验结果管理员
本文从台账（`registry/runs.csv`、`registry/{launch,close}/`）与既有审计文档提炼，**不引入新数字**。

---

## 1. 代码指纹链

### 1.1 当前生效基线

| 角色 | SHA | 说明 |
|---|---|---|
| 代码提交 | `57a4570` | 标定期强制 `eval()` 并恢复原模式；RNG 断言纳入 CUDA 生成器 |
| 锁提交 | `21bc2a2` | 16-blob code lock（lock-only commit） |
| `baseline_version` | **v12** | 唯一有效基线 |

`registry/code_lock.json` 自身 md5 = `b7e0bf01c974c724778aa3353511111c`，`git_sha = 57a4570`，16 blob，a40 上 `CODE LOCK OK`。

**历史版本**（不再作为准入基线，仅供追溯）：v9 `d1e0527` / 锁 `6c405ad`（15 blob，锁 md5 `c04c958600ca26fed9ea6944cbe12cbc`）；v10 `95a5898`（16 blob，`scale_align.py` 入 OWN）；v11 `9601320`（16 blob，删 `train_hc` 内联 remap + 守卫与 `--check` 同口径 + `.gitattributes`）。**W2 那 24 格 cal 格是在 v11 下跑的**，标定缺陷见 §4 事故 14。

### 1.2 指纹的算法口径（**必须按此，否则会被假不符骗**）

```
md5( git cat-file blob <ref>:<path> )      # 取 blob 字节，不取工作树
```

- 锁里 `definition` 写明「md5 of `git cat-file blob <ref>:<path>` bytes」，`git_sha` 记**被锁代码所在提交**，不等于 HEAD。
- 一个版本 = **两个提交**：代码提交 + 只含 `code_lock.json` 的锁提交。
- **禁止**用工作树字节算基准：本机 `core.autocrlf=true` 且仓库无 `.gitattributes`，
  blob 是 LF、工作树是 CRLF。实测 `train_hc.py` blob 15063 B / 工作树 15373 B（多 310 个 CR），
  把 blob 转成 CRLF 后 md5 **正好等于**工作树 md5——内容相同、`git status` 也判 clean，
  但直接比 md5 会报假不符。这是本项目踩过的最多次的坑。

### 1.3 三处比对（v9 实测）

| 比对 | 结果 |
|---|---|
| 锁 vs `git cat-file blob d1e0527:<path>` | **0/15 不符** |
| 锁 vs `git cat-file blob 6c405ad:<path>`（HEAD） | **0/15 不符** |
| 锁 vs a40 实测 `md5sum`（LF 工作树） | **0/15 不符** |

a40 侧另可跑 `python -m paper_b.hc.code_lock --check`，退出码 0。

### 1.4 冻结代码清单（15 个 blob）

`models/dspt.py`、`training/{data,train,metrics}.py`、`configs/protocol.json`、
`hc/{config_hc,data_hc,model_hc,cycle_encoder,train_hc,run_hc,verify_loop,snapshot_val,compare_val_snapshot,code_lock}.py`。
`__init__.py`、`make_control_init.py`、`run_v0.py` **不计入**。

---

## 2. 确定性证据（σ = 0）

同一格、同 seed、同 init 的重复 run 在**并发条件下**仍逐位相同：

| 证据 | 结果 |
|---|---|
| `A3a-0030Z` vs `A3a-1640Z` | `ckpt/best.pt` `07afec2a…`、`preds.csv` `65f7f107…` **逐字节相同**，val_engine 均 `11.316705661052403` |
| `A3a-FD001-W1`（补跑，W1 批次） | `11.3167`，与前两条相同 → 跨批次也一致 |
| `A3p` 跨 v6 / v8 / v9 三次 | 均 `10.895096839289387` |
| `A3p-0030Z` / `A3p-R1` / `A3p-FD001-W1` | 三条全 `10.895096839289387` |
| `A1-FD001/002/003` 的 `-W1r2` vs 原 `-W1` | 值一致（原 run 因 OOM/中断未落 close） |

**这排除了实现抖动，但不构成 seed 级带宽**：真正决定能否推广的变异来源是 seed / 初始化，
那要 ≥3 seed 才看得到。所以「σ=0」只能用于说明**实现确定性**，不能用于声称结论稳健。

**但「σ=0」有一个精度口径必须写明**：W2 那 24 格 cal 格的标定是在 dropout 开着时做的，
`gain` 带 **1e-11–1e-8** 的 seed 依赖（§4 事故 14）。所以在 W2 上，
「同格重复逐位相同」和「seed 极差 = 训练随机性下界」只在 **1e-8 精度内**成立；
v12 起标定逐位 seed 无关（spread `0.000e+00`），这句话才无保留成立。
结论不受影响（1e-8 比任何判据低 5 个数量级），但不能把它写成字面意义的 0。

### 2.1 早停 vs 打满预算的判别

判据：`epochs_run − best_ep == patience` 即早停；等于预算上限即打满。
实测 `A1-FD002-W1r2`：`epochs=40`、`patience=12`、`best_ep=8`、`epochs_run=20`，gap = 12 → **早停**，非截断。

---

## 2b. 测量精度上限：预测值是 bf16 量化的

**`val_engine_mean_rmse` 是在 bf16 量化的预测上算出来的。**
`predict_hc` 在 `autocast(bf16)` 下取 `.float()`，那只是**加宽类型、不增加精度**。

- `preds.csv` 里的值是 `115.0 / 71.5 / 7.8125 / 2.59375 / 3.296875` 这类二进分数，
  都是 bf16 可精确表示的数（8 位有效位）。
- bf16 在数值 1–10 上的粒度是 **0.008–0.06**，比 5e-9 大五六个数量级。

**影响**：对 RMSE ≈ 10 的结论无实质影响，但这是一条**测量精度上限**——
任何小于 bf16 粒度的差异（例如 5e-9 量级的 preds 差）都不可能是模型输出的真实变化，
只能是存盘/读回的十进制表示。因此：

- 判"代码改动是否改变数值"时，阈值取 bf16 粒度（≥1e-3），**不要用 1e-9**；
- 报告 Δ 时，低于该粒度的差异一律记为 0。

**逐位校验必须在模型输出精度上做。** 用更高精度（float64）去比模型输出（float32/bf16）
会**系统性产生假阳性**：CSV 存的是 float32 的最短往返十进制（8 位有效数字），
用 float64 读回会与加宽后的 float32 差「半个末位」（实测 5.000e-09 / 2.813e-10）。
实测 28 格中 4 格出现此类残差，把读回值转回 float32 再比，**4970/4970 逐位相同**。
因此 `preds_bitcheck.py` 一律在 **float32**（模型自身精度）上比对，重跑结果
`28 cells: 28 exact, 0 with a non-zero diff`。

**证据**：28 个 canonical checkpoint 重算 preds，24 格逐位相同、4 格差 ≤5e-9；
而同一份代码 pass1 vs pass2 **全部 0.000e+00**（排除 kernel 非确定性）。

---

## 3. 台账准入规则（Gate 表）

**按格判，不按精确 `run_id`。**

1. 格键 = `(baseline_version, variant, subset, seed)`；`baseline_version` **是承重件**。
   实测反例：省掉它会选中 pre-versioning 的 `…T1505Z`（10.5455、`audit_verdict=fail`、已 retired），
   而不是 `0030Z`（11.3167），**差 0.77 RMSE**。
2. `run_id` 必须匹配 `^HC-C-{variant}-{subset}-s101-W1(r\d+)?$`——只写"前缀 `-W1`"会误纳 `-W1foo`，
   只写"以 `-W1` 结尾"会把 `r2/r3` 排除。
3. 同格取 `status=ok` 中 `closed_utc` **最早**者；`failed` / 中断的 run **不占位**；其余标 `superseded`。
4. `0030Z` / `1640Z` 不匹配上述正则 → **永不是 Gate A 证据**，只进 `CYCLE_DEPENDENCE_STUDY.md` preliminary。
5. 同 seed 重复（`-R1` / `-R2`）标 `determinism_only`，**不进任何 Gate 表**。

### 3.1 枚举纪律

`status` 只用 `ok / failed / smoke / running / superseded`；`verdict` 只用 `pass / fail / n/a`。
**不存在 `unclosed`**：进程在 = `running`，进程没了且无 close = `failed`。
`determinism_only`、`in_gate_a_table`、`retired_version` 都是**独立布尔列**，不得塞进 `verdict`。
`retired_version` 只给**版本号复用**的 run（v6/v7），正常成功的 run 不得标 retired。

---

## 4. 事故与处置（全部留痕）

| # | 事故 | 根因 | 处置 |
|---|---|---|---|
| 1 | v8 的共享初始化必炸 `missing=['pe_periods']` | remap 用 `pop`，A3a 是 61 键、两个 period 字段都在 | 升 v9 改 `copy, never pop`；v8 整版作废 |
| 2 | 「9 倍权重」表述错误 | 冻结 val 下每引擎恰好 5 窗口，池化等权 | 三处 docstring + 任务书更正，冻结代码留正确说法 |
| 3 | `A1-FD003-W1` OOM，0 epoch | 8 进程把 GPU 顶到 47.33 GiB 满 | 改显存门 `used ≤ 40 GB` 才起格；`-W1r2/r3` 重试成功 |
| 4 | 同格两个执行者并发起同一格（`A3a-1640Z`） | `run_id` 的 `O_EXCL` 只防同名、不防同格 | 判 `duplicate`、排除出表；台账按格去重 |
| 5 | 第三条同格重复（`A3p-0105Z`） | 临时起的格不在预登记清单 | 判 `duplicate`；**只许一个发射器** |
| 6 | 文档被 PowerShell 反引号转义吃字符 | here-doc 里 `` ` `` 是转义符 | 从 `HEAD~2` 取原始字节整体恢复，按字节级重贴；`§12.5` 5 个控制字符还原 |
| 7 | 审计文档写入未来时间戳 | 预写了"02:5x / 03:0x" | 全部改为真实提交时间；加规则「时间戳一律以 a40 为准」 |
| 8 | `runs.csv` 被并发改写、判决被回写 | 两个写入方 | 主控指定唯一写入人；以**最近一次 commit 的 blob** 为准，不看工作树 |
| 9 | 本地 `all_runs.csv` 的 A8 族只有 78 行 | 本地聚合滞后，权威源在 lab3090 | 统计一律取 lab3090 产物（96/96）；已在 `ABLATION_RESULTS.md` §0 标注 |
| 10 | 文件名与论文对不上（`Enhanced…` ≠ SFI-STHGAT） | 本地 PDF 文本层命名误导 | 建「文件名→论文」对应表，**以标题为准** |
| 11 | 「作者必裁测试标签」推论 | 训练标签 cap 不约束推理期输出 | 两处文档追加撤回（保留原文），主表改称「本工作冻结口径」 |

| 12 | a40 上 `train_hc.py` / `run_hc.py` 变成 CRLF，`run_hc` 启动守卫拒跑（`REFUSED: code differs from the committed lock`），但 `code_lock --check` 报 OK | **两条校验路径口径不一致**：`code_lock.check()` 做换行规范化，`run_hc` 的守卫比原始字节；Windows→a40 同步即复发 | `sed -i 's/\r$//'` 转回 LF（备份 `experiments/hc/tmp/crlf_backup_194641/`），验证 **16/16 原始字节 md5 与锁相等**；根治：`run_hc` 守卫同口径规范化 + `.gitattributes`（`*.py text eol=lf`） |
| 13 | `train_hc` 内联了一份与 `scale_align.load_shared_init()` 重复的 remap | **代码重复**：函数版排除了 `cycle_gain`，内联版没有，cal 臂在 ep=0 即炸 `missing=['cycle_gain']` | 删内联、改调函数（单一真相源）；cal 臂冒烟补进发车前检查 |
| 14 | W2 24 个 cal 格的 `gain` 带 seed 依赖（spread **1.3e-11–1.1e-08**），与「标定是 (init, subset, 编码) 的纯函数」不符 | `calibrate_init_rms` 在建模型后**直接标定、未 `net.eval()`**，dropout 开着；dropout 在 GPU 上消耗 **CUDA 生成器**（按 seed 播种），而旧 RNG 断言只快照了 **CPU 生成器**，因此「不消耗训练随机数」是**被误证**的（断言通过，但什么都没证明） | **不重跑**：新 gain 与旧值相对差 2.8e-8（`0.0142921377` vs `0.0142921381`），比任何有意义阈值低 5 个数量级。v12（`57a4570`）标定期强制 `eval()` 并在结束后恢复原模式，RNG 断言加 `torch.cuda.get_rng_state()` 比对；修后 **train / eval 两模式 spread 均 `0.000e+00`**。留痕口径：**W2 的 24 格标定时 dropout 开着，gain 带 1e-11–1e-8 的 seed 依赖，v12 起修正** |

**事故 14 的决定性实验**（`paper_b/hc/cal_determinism.py`，FD001 A2cal，两 seed × 两模式）：

| 模式 | gain s101 | gain s202 | spread |
|---|---|---|---|
| train（W2 实际状态） | 0.0142921381 | 0.0142921381 | **7.137e-11** |
| eval | 0.0142921377 | 0.0142921377 | **0.000e+00** |

train 模式的 7.137e-11 恰好等于 W2 实测的 A2cal FD001 spread —— 这证明那 24 格**确实是在 dropout 开着时标的**，不是推断。

**这条为什么要修而不是忽略**：seed 极差被解释为「训练随机性的下界」，而这个解释要求标定不贡献**任何** seed 相关成分。当时它贡献了（虽极小）。修完这句话才是真的；不修就是用一个未验证的前提去支撑一条结论。

---

## 5. 台账当前状态

| 项 | 值 |
|---|---|
| `registry/runs.csv` | **128 行 / 38 列** |
| `check_runs.py` | `rows=128 flags=0`，退出码 **0** |
| `stage` 分布 | C **100** / B **28**（D 已预留命名空间，零行） |
| W2 行 | **33**（32 `ok` + 1 `failed`；失败格 `HC-C-A2cal-FD001-s101-W2` 保留不动，`W2r2` 为该格证据） |
| 已准入格 | **28**（主表 24 + frozen `A3p` 4），全部 `preliminary (PENDING-L2)` |
| 远端一致性 | a40 `runs.csv` 内容陈旧；**准入一律直读 a40 的 launch/close，不读那份 csv** |

台账行数的算式（防止再被转述成 134）：本轮 24 格中 **6 个 A1（FD002/003/004 × s202/s303）上一轮已登记**，
实际新增 **18 个 cal 格**（A2cal×9 + A3bcal×9），110 + 18 = **128**。

---

## 6. 复现一次 Gate A 格的最小步骤

```bash
# 1) 代码：取 v9 的 15 个 blob，逐文件核对锁
git cat-file blob d1e0527:paper_b/hc/train_hc.py | md5sum   # 期望 069f8977…
# 2) 环境
source /media/sangfor/vdb/user/zxk/env.sh
PY=/home/yndx/miniconda3/envs/a40_env/bin/python            # 系统 python3.8 无 numpy，会直接崩
# 3) 单格
$PY -m paper_b.hc.run_hc --variant A1 --subset FD002 --seed 101 \
    --run-id HC-C-A1-FD002-s101-W1r4 \
    --init-from experiments/hc/tmp/wave1_init_A1_s101.pt \
    --baseline-version v9 --gate A
# 4) 期望：close.json status=ok；val_engine_mean_rmse 与 11.8031 同量级（重跑不保证逐位，见 §2）
```

> 重跑**不保证**逐位相同：σ=0 只在同 init、同 seed、同代码且调度条件一致时成立。
> 显存门、并发度、`cudnn.benchmark` 都会影响算法选择，复现时应记录这些条件。
