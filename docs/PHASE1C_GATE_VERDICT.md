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
| FD003 recovery | 过 0.65（bias m2 0.798、m3 0.876），但 clean 失败 |
| noise | mixed/no_harm，单列，不计入 recovery |
| 完整性 | FD003 drift/stuck/noise 仍在跑；clean 结论不受影响 |

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

### FD003 bias（partial）

| mag | R3 mean (min–max) | R5 mean |
|---:|---|---:|
| 2 | 0.798 (0.686–0.923) | 0.277 |
| 3 | 0.876 (0.780–0.967) | 0.153 |
| 5 | 0.952 (0.914–0.997) | 0.015 |

### FD003 drift mag2（s101，partial）

R3 **0.583** <0.65，R2 0.980，R5 0.327。小幅度 drift 即使 oracle 重构很好，
部署式 R3 仍低于门；3-seed 待补。这是 **clean 失败之外的第二个 FD003 边界**。

## 3. 结论

1. **FD001 单工况条件性成立**：bias/drift/stuck 全部 R3≥0.65，R3 显著优于 R5，
   clean 三 seed 全过，EOL per-cycle 0.24%。
2. **FD003 不能写成成立**：bias recovery 过门，但 drift mag2 s101 R3 0.583 <0.65；
   且 s303 clean cost 0.254 >0.15。两个独立失败（recovery + clean），不能靠剔除 seed 或调 gamma 解决。
3. **noise 不是 recovery 目标**：FD001 多数 seed no_harm；R7 滤波绝对 RMSE 12.16–12.25，
   作为探索性观察，不写成主贡献。
4. **Phase 2 不启动**：Gate 1 clean 门失败；端到端不会修复 clean cost。
5. 路线取舍（RHSE 主贡献 vs LTAG 主线）仍 **PENDING-L2**；Codex 2026-11-21 后裁决。

## 4. 下一步

- 等 FD003 drift/stuck/noise 收齐，补完整 3-seed 表（证据完整，不改判定）。
- 论文边界：FD001 条件性 + FD003 边界 + noise 单列；LTAG 主线。
- 不调 gamma、不剔除 seed、不按子集特判；test 封存。
