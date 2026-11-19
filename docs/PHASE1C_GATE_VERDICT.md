# Phase 1c Gate 1 判定（2026-11-20）

> 依据：`results/FREEZE_PHASE1C.json`、`experiments/phase1c/logs/`、`phase1c_gate`。
> **test 全程封存。** 本判定在 FD003 剩余 cell 跑完前已可由 clean 门确定。

## 1. 判定

**Gate 1：FAIL。**

| 条件 | 结果 |
|---|---|
| clean cost ≤0.15（per subset, seed） | **FAIL：FD003 s303 = 0.254** |
| EOL per-sensor-cycle ≤2% | PASS（FD001 0.24%，FD003 0.00%） |
| FD001 bias/drift/stuck recovery | PASS（0.700–0.872） |
| FD001 bias/drift/stuck recovery | PASS（0.700–0.872） |
| FD003 recovery | bias PASS（0.798–0.952）；drift mag3/5 PASS（0.772/0.841）；**drift mag2 FAIL 0.640**；**stuck FAIL 0.570**；noise 0.413–0.653 |
| noise | FD001 no_harm/mixed；FD003 damaging；均单列 |
| 完整性 | **62/62 complete，0 failed** |

## 2. 关键证据

### clean cost（逐 seed）

| subset | seed | R0 | R3c | cost |
|---|---:|---:|---:|---:|
| FD001 | 101 | 12.647 | 12.641 | −0.006 |
| FD001 | 202 | 12.424 | 12.394 | −0.030 |
| FD001 | 303 | 12.887 | 12.894 | +0.007 |
| FD003 | 101 | 14.299 | 14.448 | +0.149 |
| FD003 | 202 | 14.507 | 14.466 | −0.041 |
| FD003 | 303 | 13.110 | 13.364 | **+0.254** |

### FD001 非 noise（3-seed）

| fault | mag | R3 mean (min–max) | R5 mean |
|---|---:|---|---:|
| bias | 2/3/5 | 0.784 / 0.826 / 0.782 | 0.053 / 0.078 / 0.015 |
| drift | 2/3/5 | 0.700 / 0.797 / 0.814 | 0.269 / 0.362 / 0.363 |
| stuck | 3 | 0.872 (0.851–0.895) | −0.037 |

### FD003 bias（3-seed）

| mag | R3 mean (min–max) | R5 mean |
|---:|---|---:|
| 2 | 0.798 (0.686–0.923) | 0.277 |
| 3 | 0.876 (0.780–0.967) | 0.153 |
| 5 | 0.952 (0.914–0.997) | 0.015 |

### FD003 drift（3-seed）

| mag | R3 mean (min–max) | R5 mean | R2 (s101) |
|---:|---|---:|---:|
| 2 | **0.640** (0.583–0.739) | 0.319 | 0.980 |
| 3 | **0.772** (0.736–0.840) | 0.494 | 0.987 |
| 5 | **0.841** (0.777–0.919) | 0.493 | 0.988 |

- mag2 3-seed mean **0.640 <0.65**，是 recovery 失败。
- mag3 0.772、mag5 0.841，均过门；drift 的失败是 **幅度依赖**，不是“drift 全失败”。
- R2 0.98+，瓶颈在检测/选择；小幅度 drift 是 FD003 最弱 cell。

### FD003 stuck / noise（3-seed）

| fault | mag | R3 mean (min–max) | R5 mean |
|---|---:|---|---:|
| stuck | 3 | **0.570** (0.338–0.932) | −0.244 |
| noise | 2 | **0.413** (0.025–0.796) | −2.618 |
| noise | 3 | 0.653 (0.513–0.801) | −2.260 |
| noise | 5 | 0.633 (0.573–0.723) | −1.837 |

- FD003 stuck 0.570（高方差 0.338–0.932）：15 步 run-length 延迟 + 短窗口，均值 <0.65。
- FD003 noise 与 FD001 不同：R1>R0（噪声伤害 B0），R3 0.41–0.65，R5 为负；
  noise 在 FD003 是 damaging，但仍低于 0.65（mag3 0.653 贴边）。

## 3. 结论

1. **FD001 单工况条件性成立**：bias/drift/stuck 全部 R3≥0.65，R3 显著优于 R5，
   clean 三 seed 全过，EOL per-cycle 0.24%。
2. **FD003 不能写成成立**：三个独立失败——drift mag2 3-seed **0.640 <0.65**、
   stuck **0.570 <0.65**、s303 clean cost **0.254 >0.15**；bias 与大漂移虽过门，也救不了。
   这是“故障类型 × 幅度 × seed”的边界，不能靠剔除 seed 或调 gamma 解决。
3. **noise 不是统一目标**：FD001 多数 seed no_harm，R7 滤波 12.16–12.25；
   FD003 noise 是 damaging，R3 0.41–0.65。论文单列，不写成主贡献。
4. **Phase 2 不启动**：Gate 1 未过；端到端不会修复 clean cost 或小漂移/stuck。
5. 路线取舍（RHSE 主贡献 vs LTAG 主线）仍 **PENDING-L2**；Codex 2026-11-21 后裁决。

## 4. 下一步

- 队列已 62/62 完成，0 failed；终版 3-seed 表见 `docs/PHASE1C_RESULTS.md`。
- 论文边界：FD001 条件性 + FD003 边界 + noise 单列；LTAG 主线。
- 不调 gamma、不剔除 seed、不按子集特判；test 封存。
