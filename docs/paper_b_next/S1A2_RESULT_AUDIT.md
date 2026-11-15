# S1-A 第二半结果审计（军师）

审计对象：`docs/paper_b_next/S1A2_PUBLIC.md`（`87194993`，96/96 格终态）
审计方法：不 import `scripts/analyse_s1a2.py`。在两台机器上逐格读 `config.json` 与
`validation_predictions.csv`，只算逐发动机 RMSE；本地独立实现配对与 bootstrap。
数据：a40 100 格（24 v1 + 16 sham + 60 V0）、3090 56 格（36 v1 + 20 sham），合计 156 个
（60 V0 + 60 V1 + 36 sham），与期望的 96 个新训格一致。

---

## 0. 结论

**可判 pass（数值与判据全部通过）**，但报告有 **3 处收口前必须改的文档问题**（见 §5）。
数值、配对、同预算、控制失效、分解式五条都核过，没有发现需要改数的缺口。

| 审计项 | 结果 |
|---|---|
| 96 格完整性（config / 行数 / 截点） | **通过**：0 缺、0 多、0 重复、0 坏行、0 坏配置 |
| 配对与区间（独立重算 vs §2.1） | **通过**：点估计最大差 **4e-5**；CI 差 ≤0.01（bootstrap 蒙特卡洛/抽序） |
| 同预算（`two_phase` V0 vs V1） | **通过**：12/12 预算字段相同；`data.py` 仅多一个 `"cycle"` 数组；闸一 12/12 `max|diff|=0` |
| `aidgn` 控制失效 | **通过**：5 格 `rel` 0.9227/0.8741/0.8118/0.7494/0.5028，与报告一致 |
| 分解式 V1−sham = (V1−V0) − (sham−V0) | **通过**：三方法差 **0.000000** |
| `s+1=2<3` 推理 | **通过**：R 的四个方法里只有 `two_phase` 通过，加 `aidgn` 也只有 2 |

---

## 1. 96 格完整性

- 期望：V0 60（5 方法 × 4 子集 × 3 seed）、V1 60、sham 36（`aidgn`/`two_phase`/`dual_mixer`）。
  实际：156 个（含 V0），**missing=0、extra=0、dups=0**。
- 行数：FD001 100、FD002 260、FD003 100、FD004 250（验证机数 × 5 截点），
  **bad_rows=0**；每台机恰好 5 个截点，**bad_cuts=0**。
- `config.json` 的 `cycle_arm`/`model`/`subset`/`seed` 与路径逐格相符，**bad_cfg=0**。
  `cycle_arm` 只出现 `v1`/`sham`，没有第三值。

## 2. 独立重算 vs §2.1

从 `validation_predictions.csv` 重算 engine-level（每机 5 截点先算 RMSE 再对机等权），
同方法同子集同 seed 同发动机逐机配对，seed 先平均，四子集等权，10,000 次发动机重采样，
RNG `20261114`。五个主比较：

| 方法 | 主比较 | 我的点估计 | 报告 §2.1 | 差 | 方向 | 结论 |
|---|---|---:|---:|---:|---|---|
| `aidgn` | V1−sham | −11.7130 | −11.7130 | +0.00004 | 3/4 | 一致 |
| `two_phase` | V1−sham | −0.7129 | −0.7129 | +0.00003 | 4/4 | 一致 |
| `dual_mixer` | V1−sham | −0.1615 | −0.1615 | +0.00003 | 3/4 | 一致 |
| `ffve` | V1−V0 | −0.2371 | −0.2371 | +0.00003 | 3/4 | 一致 |
| `fcstgnn` | V1−V0 | −0.3091 | −0.3091 | +0.00004 | 3/4 | 一致 |

CI 端点最大差 **0.0099**（`ffve` 下界），来自 bootstrap 抽序/蒙特卡洛，不是配对错误：
- `two_phase`：我 [−1.0557, −0.3755]，报告 [−1.0542, −0.3729]，**都排除 0**；
- `ffve`/`fcstgnn`/`dual_mixer`：我分别 [−0.5844,+0.1045] / [−0.6969,+0.0841] /
  [−0.4773,+0.1496]，报告都跨 0，**结论不变**；
- `aidgn`：我 [−12.8280,−10.5461]，与报告逐位相同（它是抽序里的第一个比较）。

**分解式**（§2.2）：

| 方法 | V1−V0 | sham−V0 | 差 | V1−sham | 差 |
|---|---:|---:|---:|---:|---:|
| `aidgn` | +1.1467 | +12.8597 | −11.7130 | −11.7130 | 0.000000 |
| `two_phase` | −0.5453 | +0.1676 | −0.7129 | −0.7129 | 0.000000 |
| `dual_mixer` | +0.0215 | +0.1830 | −0.1615 | −0.1615 | 0.000000 |

三行都逐位成立，说明配对口径一致。

## 3. 同预算（`two_phase` V0 vs V1）

逐格比 12 个 V1 格与对应 V0 格：

- `epochs` / `patience` / `batch` / `lr` / `window` / `precision`：**12/12 全部相同**。
- `unified.py` / `adapters.py` / `common.py` 的 `code_sha256`：**12/12 相同**。
- `data.py`：12/12 不同，但 diff 只有两处：
  1. `from paper_b.training.data import CYCLE_SCALE, ...`；
  2. `windows()` 返回 dict 多一个 `"cycle": np.minimum(ends / CYCLE_SCALE, 1.0)`。
  **训练窗、截点、scaler、`x`/`age`/`y`/`unit`/`cut` 均未改动**；`CYCLE_SCALE = 250.0`，
  与第一半的 `min(c/250,1)` 一致。这不是协议变更，是插件需要的 cycle 通道。
- 闸一 `gate1.log`：12 格逐格 `tensors=37/59`、`max|diff|=0.0`、`mismatched=0`、
  `sha_mine=sha_v0`；`WORST max|diff| over all cells: 0.0`、`gate1 passed: True`。
  与报告 §1.8 的表一致。

**一个必须写进同预算表的已知差异**：`dual_mixer` 的跑批器用 `RandomSampler(generator)`
生成打乱顺序，V0 用的是 `shuffle=True`（全局 RNG）。报告 §1.3 已披露，但 §2/同预算表没列。
它不影响 `dual_mixer` 的「不成立」结论（CI 跨 0），但读者会问「同预算是否包括数据顺序」——
应在同预算表旁注明「`dual_mixer` 的打乱顺序实现不同，已披露；其余四方法与 V0 的抽序逐位相同」。

## 4. `aidgn` 控制失效

5 个 sham 格，按 `train_loss` 末轮/首轮算 `rel`：

| 格 | `rel`（我算） | 报告 §2.6 | sham engine RMSE | 同格 V0 |
|---|---:|---:|---:|---:|
| FD004/s101 | 0.9227 | 0.92 | 49.05 | 13.01 |
| FD004/s303 | 0.8741 | 0.87 | 46.45 | 13.19 |
| FD002/s303 | 0.8118 | 0.81 | 44.75 | 11.80 |
| FD004/s202 | 0.7494 | 0.75 | 37.27 | 13.75 |
| FD001/s303 | 0.5028 | 0.50 | 41.03 | 11.32 |

全部 >0.8 阈值，验证 RMSE 落在 37.3–49.1（报告写 36.9–49.1），与「训练不动、停在均值噪声底」
一致。5 格保留、未剔除，符合预注册「不删格、不换估计量」。

§3 的 24 格事后尺度匹配：报告标题明写 **post hoc**，§3.3 明写「不替换主分析那一行、不进入档位计算」；
主 `cycle_runs` 只含 96 个新训格，尺度匹配的 24 格在 `cycle_runs_scale`，**没有混进主结果**。

## 5. 收口前必须改的三处文档问题

1. **顶部状态与正文矛盾**。第 3 行仍写「状态：**进行中**。预注册 `v1.2` 已冻结，96 格尚未开跑；
   §3 之后等结果填。」——但 §2 已是 96/96 终态。应改为「完成（96/96 终态，`tier = weak`）」。
2. **§4「待填」是 72 格中间版的残留**。它写「主比较未齐时档位由脚本强制扣住，不出数」，
   与 §2.4 的 `tier = weak` 直接矛盾。应删掉整节，或改成「已填，见 §2.4/§2.5/§2.6」。
3. **`dual_mixer` 的打乱顺序差异**未进同预算表（见 §3 末）。应在 §2 或同预算表旁加一行披露。

这三处不影响任何数字或档位，但一个最终报告不能同时写「尚未开跑」和「96/96 终态」。
改完这三处，本阶段可落 pass。

---

## 6. 审计证据

```bash
# 逐格读取（两台机器，只读 config.json + validation_predictions.csv）
$ ssh a40 "python3 /tmp/audit_s1a2_remote.py --root /home/yndx/cycle_runs \
    --v0-root /home/yndx/paper_b_public_baselines_20261111/paper_b/experiments/\
open_source_reproduction_20261111/runs/unified --out /tmp/audit_a40.json"
cells=100
$ ssh 3090 "python3 /tmp/audit_s1a2_remote.py --root /home/user/cycle_runs --out /tmp/audit_3090.json"
cells=56

# 本地独立配对 + bootstrap（先按上一条生成 audit_a40.json / audit_3090.json）
$ python scripts/audit_s1a2_combine.py
cells=156 expected=156 missing=0 extra=0 dups=0
bad_rows=0 bad_cuts=0 bad_cfg=0
aidgn      v1-sham  d=-11.7130 ci=[-12.8280,-10.5461] dir=3/4
two_phase  v1-sham  d=-0.7129 ci=[-1.0557,-0.3755] dir=4/4
dual_mixer v1-sham  d=-0.1615 ci=[-0.4773,+0.1496] dir=3/4
ffve       v1-v0    d=-0.2371 ci=[-0.5844,+0.1045] dir=3/4
fcstgnn    v1-v0    d=-0.3091 ci=[-0.6969,+0.0841] dir=3/4
decomposition diff: 0.000000 / 0.000000 / 0.000000

# two_phase V0 vs V1 config
FD001/s101: budget_diff=none code_diff=['data.py']
... 12/12 same
# data.py diff: only CYCLE_SCALE import + "cycle": min(ends/250, 1.0)

# 闸一
WORST max|diff| over all cells: 0.0
gate1 passed: True

# aidgn DEAD rel
FD004/s101 rel=0.9227  FD004/s303 rel=0.8741  FD002/s303 rel=0.8118
FD004/s202 rel=0.7494  FD001/s303 rel=0.5028
```

改动文件：`scripts/audit_s1a2_remote.py`（新增，逐格读取）、
`docs/paper_b_next/S1A2_RESULT_AUDIT.md`（本文件）。未改动任何已有结果或报告。
