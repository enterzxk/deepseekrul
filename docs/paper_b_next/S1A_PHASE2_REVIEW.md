# S1-A 第一半独立复核（军师）

复核对象：`docs/paper_b_next/S1A_CYCLE_MATCHED.md`（§7–§9，commit `351a71af`）
冻结件：`docs/paper_b_next/S1A_POSITION_PREREG.md`（`29f337f9`，frozen_at 2026-11-14T18:19:44+08:00）
方法：**不 import `analyse_s1a.py`**，按冻结件文字独立重写一遍（`scripts/audit_s1a_independent.py`），
逐格读 `paper_b/results/s1a_collected/**/val_pred.csv`，不读 `metrics.json::val_engine_rmse`。

---

## 0. 结论

**四项全部通过，判定成立：无唯一胜出，`pre_norm` 为默认而非证据。**
`head_concat` 可排除。Cycle 的正效应本身是稳的（三个位置对 `none` 全部合格，5/5 seed 同号）。

唯一一处**可核性限制**（不是错误）：§9 的「首轮打印措辞修正、重跑逐位不变」中，
「逐位不变」无法从 git 证明——初版脚本未入库，没有 before 状态。但 after 状态正确，
且我的独立重算与报告一致，见 §5。

---

## 1. §7.1 的 4×4 绝对表：16/16 格一致

我重算了全部 16 个（位置 × 子集）格，不是只抽查 4 格：

| 子集 | `none` | `pre_norm` | `post_norm` | `head_concat` |
|---|---:|---:|---:|---:|
| FD001 | 11.7198 | 10.5248 | 11.1654 | 11.7711 |
| FD002 | 11.3998 | 10.6992 | 10.7296 | 11.0795 |
| FD003 | 11.9353 | 10.2740 | 10.5964 | 10.3683 |
| FD004 | 13.5791 | 11.8872 | 12.0917 | 12.4120 |

**max |我的重算 − §7.1| = 0.00005**（16 格，四舍五入到 4 位后全同）。
seed sd 也逐格对上（例：FD001 `pre_norm` 0.6287、FD004 `none` 0.3165）。
验证机数 20/52/20/50 = 142，与 §7.1 一致。

## 2. 合并 CI、方向计数、加权敏感性：按冻结件算法复现

独立实现：逐机配对差 → 子集内对发动机有放回重采样（10,000 次）→ seed 先平均 →
四子集**等权**平均 → 2.5/97.5 与 0.835/99.165 分位；点估计用观测值；RNG `20261114`。
按脚本同样的配对顺序消费随机数，结果逐项对上：

| 对比 | 点估计 | 95% CI | 98.33% CI | 方向 | 机数加权 |
|---|---:|---|---|---|---:|
| `pre_norm − none` | −1.3122 | [−1.8898, −0.7583] | [−2.0230, −0.6441] | 4/4 | −1.2546 |
| `post_norm − none` | −1.0127 | [−1.4042, −0.6244] | [−1.4878, −0.5426] | 4/4 | −1.0358 |
| `head_concat − none` | −0.7508 | [−1.1284, −0.4012] | [−1.2058, −0.3265] | 3/4 | −0.7417 |
| `post_norm − pre_norm` | +0.2994 | [−0.1409, +0.7303] | [−0.2494, +0.8134] | 0/4 | +0.2187 |
| `head_concat − pre_norm` | +0.5614 | [+0.1420, +0.9817] | [+0.0504, +1.0857] | 0/4 | +0.5129 |
| `head_concat − post_norm` | +0.2620 | [+0.0146, +0.5059] | [−0.0365, +0.5574] | 1/4 | +0.2941 |

最大偏差：点估计 **4e-5**、95% 端点 **4e-5**、98.33% 端点 **4e-4**（分位数插值/浮点归约），
方向计数 6/6 相同，加权值与 §7.2 一致。**算法与冻结件一致。**

## 3. 两两 CI 齐全，支配判定没有越界

- 三对齐全：`post−pre`、`head−pre`、`head−post`，都在 §7.3 表里。
- 挡住支配的只有 `post_norm − pre_norm`（95% CI 跨 0）。`head_concat − pre_norm`
  的 95%/98.33% CI 都不含 0，方向 0/4，用于**排除** `head_concat`，不是用于宣布胜出。
- 支配检验的方向：冻结件要求 `best − other` 的 CI 上限 < 0；§7.3 表是 `other − best`。
  `post−pre` 跨 0 ⇒ 镜像也跨 0，结论不变；`head−pre` 全正 ⇒ `pre−head` 全负，支配成立。
  判定代码用的是镜像后的有序对，**没有拿错方向**。
- **建议（不阻塞）**：§8 的支配句最好把镜像 CI 直接写出来
  （`pre_norm − post_norm` = [−0.7303, +0.1409]，`pre_norm − head_concat` = [−0.9817, −0.1420]），
  否则读者按「CI 上限 < 0」直接套 §7.3 的 `head_concat − pre_norm` 行，会误以为 `pre_norm`
  连 `head_concat` 都没支配。文档文字本身没错，只是少一行显式镜像。

## 4. `Q` 与胜出判定：严格按冻结条款

`paper_b/reports/s1a_report.json::decision` 原文：

```
rule_version v1, prereg_commit 29f337f9...
eligible [pre_norm, post_norm, head_concat], blocked {}
qualified [pre_norm, post_norm, head_concat]
winner null, selected_position pre_norm, fallback_used true, cycle_is_negative_effect false
notes: "3 positions qualify ... but pre_norm does not dominate all of them pairwise"
       "no unique winner; stage two continues with pre_norm as the implementation default ..."
```

- `Q` 三个成员都满足「95% CI 上限 < 0 且 n_neg ≥ 3」，与 §7.2 一致。
- `|Q| ≥ 2` → 取 `Δ` 最负者 `pre_norm`；它对 `post_norm` 的成对 CI 跨 0 → 不支配 →
  判无唯一胜出 → 回落 `pre_norm`。**这正是冻结件 §3.4 的字面执行，没有超出。**
- `missing 0 / bad_cuts 0 / bad_config 0`，`complete=true`，没有缺格导致的假合格。
- 全文 6 处「胜出」：第 4、198、238、241 行是否定/条件句；第 141 行是一般规则
  （「胜出位置在算 TEST 前冻结」）；第 272 行明写「不得把 `pre_norm` 当成胜出位置」。
  **没有任何一处把 `pre_norm` 写成胜出。**

## 5. §9 披露：成立，但「逐位不变」不可从 git 证明

- **pooled 字段**：我独立重算 `S1A_none/FD004/s303`，`metrics.val_engine_rmse = 19.7102`、
  engine-level = 13.5721，差 **6.13816**，与 §9 的 6.1382 一致。
- **措辞修正**：当前脚本与 `s1a_console.txt` 第 27 行都是修正后的文本
  （明确区分 optimizer-visible totals 与 gradient-path count，指向 §4.1），旧句
  「总参数 = 可训练参数」已不存在。
- **但**：`scripts/analyse_s1a.py` 是 `351a71af` 新增的单个文件，git 里没有初版，
  所以「重跑后 absolute/pairs/decision 逐位不变」无法从提交历史证明。我能证明的是
  **after 状态正确**：我的独立重算与报告 16 格 + 6 对比一致（§1–§2），判定字段与冻结件一致。
- **是否构成「开跑后修改预登记判据」：不构成。** 改的是控制台打印文字，不是判据、阈值、
  聚合权重或指标定义；容量门两种口径（总参数 0.95%、有效参数 0.44%）都 <1%，判定不受影响。
  若要挑毛病：预注册原文写「有效（可训练）参数」，这个括号本身有歧义；文档把它定为
  「前向有梯度通路」，方向是更保守的（有效值更小）。本次两种口径都过门，所以无实质影响；
  若将来某一位置总参数 >1% 而有效参数 <1%，这个定义就需要一次带日期的 amendment 才能用。

## 6. 判定

**S1-A 第一半 pass。** 可以写进论文的表述只有：

1. **Cycle 作为插件在 DSPT 上有效**：三个位置对 `none` 的合并 95% CI 全部不含 0，
   方向 4/4 或 3/4，5/5 seed 同号；`pre_norm` 合并 −1.3122。
2. **位置之间没有唯一胜出**：`pre_norm` 点估计最好，但未支配 `post_norm`
   （`post−pre` 95% CI 跨 0）；按预注册取 `pre_norm` 为**默认**，不是证据。
3. **`head_concat` 排除**：`pre_norm − head_concat` 的 95%/98.33% CI 都不含 0，方向 0/4；
   `head_concat` 还在 FD001 上比 `none` 差。

第二阶段（五个公开方法）可以用 `pre_norm` 开工，但叙事必须写「默认位置」，
不得写「最优位置」。若要分开 `pre_norm` 与 `post_norm`，需要更多 seed 或更多发动机，
本阶段不做。

---

## 7. 复核证据

```bash
# 独立重算（不 import analyse_s1a.py）
$ python scripts/audit_s1a_independent.py
## 7.1 absolute table
max |mine-doc| over 16 cells = 0.00005
## 7.2/7.3 pairs (replicated draw order)
pre_norm - none     -1.3122 [-1.8898,-0.7583] [-2.0227,-0.6445] 4/4 -1.2546
post_norm - none    -1.0127 [-1.4042,-0.6244] [-1.4877,-0.5426] 4/4 -1.0358
head_concat - none  -0.7508 [-1.1284,-0.4012] [-1.2055,-0.3267] 3/4 -0.7417
post_norm - pre_norm   +0.2994 [-0.1409,+0.7303] [-0.2492,+0.8133] 0/4 +0.2187
head_concat - pre_norm +0.5614 [+0.1420,+0.9817] [+0.0504,+1.0857] 0/4 +0.5129
head_concat - post_norm +0.2620 [+0.0146,+0.5059] [-0.0364,+0.5574] 1/4 +0.2941

# 报告完整性
missing 0  bad_cuts 0  bad_config 0
prereg_commit 29f337f9575877b59bd7ca1fcd6f0c84f516dcea
winner null  selected_position pre_norm  fallback_used true

# pooled 字段
metrics.val_engine_rmse 19.710219688697528
engine_mean 13.572055402710761
diff 6.1381642859867664
```

改动文件：`scripts/audit_s1a_independent.py`（新增，独立复核脚本）、
`docs/paper_b_next/S1A_PHASE2_REVIEW.md`（本文件）。未改动任何已有结果或判定文件。
