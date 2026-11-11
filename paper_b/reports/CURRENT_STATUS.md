# Paper B current status

Updated 2026-11-11 (takeover day). Authoritative numbers are added only after
the corresponding `metrics.json` / `test_metrics.json` exist.

## 1. Where the project stands

| item | state |
|---|---|
| branch | `punet-interval`, remote HEAD = local HEAD = `95e729e` |
| Paper A | untouched |
| Paper B working tree | `lab3090:/home/user/rulb` (no git); archived snapshot in `paper_b/archive/legacy_2026_11/` |
| active code | `paper_b/models`, `paper_b/training`, `paper_b/evaluation` |
| historical evidence | mostly INVALID or untraceable; see `ARCHIVE_INDEX.md` and `PROTOCOL_AUDIT.md` |
| current compute | protocol-fixed DSPT 8-config ablation + mechanism grids (2 x RTX 3090) |

## 2. What is trustworthy today

Only the following statements are supported by code that can be run end to end:

1. The repository copy of the old DSPT runner is non-runnable (`NameError`),
   and the committed test evaluator cannot load the old checkpoints (span 100
   vs 70). **However**, the archived prediction CSVs are arithmetically exact
   and are reproducible from the checkpoints with a reconstructed evaluator
   (corr 0.9935-0.9995; `results/legacy_repro.csv`, `legacy_all_repro.csv`).
   The historical table is protocol-confounded, not fabricated.
2. The OOF 60-cell grid used truncation points sampled from the official test
   RUL distribution. Those numbers are **INVALID**.
3. Recomputing the archived base/FULL checkpoints with the P0-fixed evaluator
   moves the absolute values (FD001 base 10.60 -> 10.92, FULL 12.70 -> 12.33)
   but does **not** flip the ranking (FULL worse on FD001, tie FD002, better
   FD003/FD004). The historical negative ablation is therefore not explained
   by the evaluation fixes alone.
4. The HI target is a decreasing health index (per-engine Spearman vs cycle
   -0.79 to -0.92). It is globally monotone but locally noisy: ~48% of adjacent
   steps move against the trend, and R^2(HI~RUL) is 0.61-0.78. This makes a hard
   monotone penalty questionable and makes HI largely redundant with RUL.
5. Sensor-correlation heterogeneity across operating conditions is 10-19x
   larger on FD002/FD004 than on FD001/FD003 (hetero/within = 0.47/0.26 vs
   0.025/0.032). On FD002 the top sensor pairs change correlation by 0.51-0.56
   between conditions (s8-s14, s13-s14, s12-s13).
6. The degradation signal is smooth on a ~25-cycle scale (first-PC
   autocorrelation 0.76-0.82 at lag 7, crossing 0.5 at lag 23-27), so the MS
   kernels k=3/5/7 all sit inside the correlation length and see nearly the
   same low-frequency content.
7. The archived FULL under-predicts high-RUL engines (FD001 bias -12.4 at
   RUL>=125 vs base -6.8), which is where its test deficit is concentrated.

## 3. Historical numbers (do not cite without the caveat)

| number | source | status |
|---|---|---|
| DSPT FD001 base 10.60 / FULL 12.70 | `archive/.../results/abl_test2.csv` | untraceable evaluator, 1 seed |
| DSPT FD002/3/4 base 11.40/11.95/12.36 | same | same |
| Vanilla FD001 BASE 15.91 | `archive/.../results/vb_test.csv` | 3 seeds, window-level selection, different evaluator |
| OOF FD001 10.12 | `archive/.../results/oof_summary.csv` | INVALID (test-derived truncation) |

## 4. Running now

| grid | runs | purpose |
|---|---:|---|
| `ablation8` | 96 | DSPT base / MS / SA / HI / pairs / FULL, 4 subsets x 3 seeds, L=70 |
| `vanilla` | 24 | independent Vanilla baseline and feature-matched control |
| `nocycle` | 12 | absolute-cycle control |
| `ms_mech` | 18 | projection vs multi-scale vs capacity |
| `sa_mech` | 24 | sensor identity vs operating-condition context |
| `hi_mech` | 18 | HI source and monotone sign |

## 5. Next decisions (evidence-gated)

1. After the 8-config ablation: is FULL ever better than base with 3 seeds and
   engine-level selection? If not, DSPT-Full is not the paper model.
2. MS: if `multi3` (parameter-matched single scale) matches `full`, the gain is
   capacity/projection, not multi-scale.
3. SA: if `cond` beats `basic` on FD002/FD004, the mechanism is condition-aware
   sensor interaction; if `identity` beats `basic`, the mechanism is sensor
   identity. If neither, SA is not supported.
4. HI: if `hB_correct` or `hf_nomono` beats `hf` with the legacy sign, the
   problem is representation/target noise, not the auxiliary task itself.
5. Window search only after the model is frozen; selection on validation only.