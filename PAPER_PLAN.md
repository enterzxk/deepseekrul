# Paper Plan

**Title (working)**: Cut-Dependent Inference-Time Aggregation for C-MAPSS RUL:
When Learned Gating Helps and When It Does Not

**One-sentence contribution**: On a frozen C-MAPSS backbone, the benefit of
multi-window RUL aggregation is cut-dependent — a learned gate (LTAG) improves
end-of-life test RMSE over K1, while simple context-aware calibration helps early
cuts but not end-of-life, and fair temporal baselines remove the apparent advantage
of measurement-model sensor reconstruction.

**Venue**: IEEE Transactions on Instrumentation and Measurement (TIM), journal;
fallback: IEEE PHM / IEEE Access.
**Type**: empirical / diagnostic (not a new-SOTA method paper).
**Date**: 2026-11-21.
**Page budget**: 12 pages including references (IEEE journal style); main body to
Conclusion ≈ 9 pages, references ≈ 2, appendix ≈ 1.
**Section count**: 8.

## Claims-Evidence Matrix

| # | Claim | Evidence | Status | Section |
|---|---|---|---|---|
| C1 | LTAG improves B0 test RMSE over K1 on all four subsets (15/15). | Table 1; `LTAG_test_eval.json`: FD001 +0.426, FD002 +0.162, FD003 +0.656, FD004 +0.321; mean +0.391. | Supported | §5.1 |
| C2 | The LTAG advantage is end-of-life/cut-dependent. | Table 3; cut=0.95: LTAG best on FD002/FD004, tied FD001/FD003; LS advantage at 0.40–0.55. | Supported | §6 |
| C3 | Simple context-aware calibration (K1-Calib, LS-ctx) beats LTAG on validation mean but not at end-of-life. | Table 2/3; LS−LTAG at 0.95: +0.267/+0.592/−0.276/+0.851. | Supported | §6.2 |
| C4 | Multi-window consistency is not necessary for the gain. | LTAG-noMW vs LTAG: differences ≤0.17; noMW better on FD001/FD002. | Supported (negative) | §6.3 |
| C5 | RHSE-TSH reconstruction is not better than fair backtracked hold/median on bias/drift; only stuck. | Table 4; R5m ≥ R3 on bias/drift; R3 only wins stuck. | Supported (negative) | §7 |
| C6 | LTAG is not a universal/SOTA aggregator. | Validation mean: K1-Calib/LS 4/4 beat LTAG; LS wins only 50–56% engines. | Supported (boundary) | §6.4 |

## Structure

### §0 Abstract (200–250 words)
- What: cut-dependent aggregation for C-MAPSS RUL on a frozen backbone.
- Why: multi-window gates are often reported as universal, but their benefit may
  depend on where in the engine life the prediction is made.
- How: frozen B0 + LTAG; 5-cut validation; 15/15 test; strong controls
  (K1, Median, Global-g, K1-Calib, Linear-Stack, noMW/noCtx).
- Evidence: test +0.391 (15/15); cut=0.95 LTAG best/tied; calibration wins early cuts.
- Most remarkable: the same method can look better or worse depending on cut
  composition — report cut-wise, not only mean.

### §1 Introduction (1.5 pages)
- Hook: RUL models are evaluated near end-of-life, but multi-window aggregation is
  usually tuned on a mix of early/late cuts.
- Gap: no study separates “aggregation helps early” from “aggregation helps at
  end-of-life”; strong post-hoc calibration controls are missing.
- One-sentence contribution (above).
- Contributions: C1–C6, each falsifiable.
- Hero figure: cut-wise RMSE curves for K1/LTAG/K1-Calib/LS (Fig. 1).

### §2 Related Work (1 page)
- C-MAPSS RUL backbones and evaluation (Saxena et al.; Heimes; Zheng et al.;
  Li et al.; and recent transformer/CNN RUL work) — `[VERIFY]` each entry.
- Multi-window / temporal aggregation and calibration (stacking, isotonic,
  quantile calibration) — `[VERIFY]`.
- Sensor-fault / measurement-model reconstruction in RUL — `[VERIFY]`.
- Positioning: this paper is a **cut-controlled diagnostic**, not a new backbone.

### §3 Problem Setup and Frozen Protocol (1.5 pages)
- C-MAPSS FD001–FD004; cap=125; engine-level split; cut fractions 0.4/0.55/0.7/0.85/0.95.
- Frozen B0 backbone; K1/Median definitions; LTAG gate.
- Estimators: engine_mean_rmse and pooled_rmse; never mixed.
- Test sealed until freeze; one-shot confirmatory.
- Code-lock deviation: current worktree + `code_lock_phase2.json`, not `ee39d7e2`.

### §4 LTAG Method (1.5 pages)
- Age-aligned windows: `p_j + (c − (t_j+1))`.
- Features: p_k1, p_med, p_mean, p_std, MAD, shift, regime_dist, cap, cut_norm, hist_norm.
- Gate: 2-layer MLP, sigmoid, zero-init last layer; censored hinge loss.
- 5-fold GroupKFold by (subset, engine); validation engines never used for fitting.
- Complexity: negligible inference overhead.

### §5 Experiments (2 pages)
- Table 1: B0 test (K1/LTAG/Median; 15/15, +0.391).
- Table 2: validation strong controls.
- Protocol: seeds 101/202/303 (+404/505/606 for FD001), engine-level paired bootstrap.
- Main text: test result; validation controls; no SOTA claim.

### §6 Cut-Dependent Analysis (1.5 pages)
- Fig. 1: cut-wise RMSE curves; crossover at 0.7–0.85.
- Table 3: LS−LTAG by cut; cut=0.95 bootstrap.
- C2/C3/C4: LTAG end-of-life; calibration early; noMW not necessary.
- C6: LS wins only 50–56% engines; FD001 median worse.

### §7 Negative Results: RHSE-TSH (1 page)
- Fair R5b/R5m baselines; Table 4.
- R3 only wins stuck; bias/drift fail; RHSE-TSH deleted.
- Implication: measurement-model reconstruction is not a general front-end.

### §8 Limitations and Conclusion (0.5 page)
- Test historically observed; one-shot confirmatory, not pristine blind.
- Only B0 backbone; no cross-backbone claim.
- Cut-dependent result may not transfer to other datasets.
- Conclusion: report cut-wise; learned gating is not universally better; fair
  baselines are mandatory.

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|---|---|---|---|---|
| Fig 1 | Hero line plot | Cut-wise engine-mean RMSE for K1/LTAG/K1-Calib/LS on 4 subsets; show crossover at 0.7–0.85 | `results/ltagv2_p1_*.preds.csv` | HIGH |
| Fig 2 | Bar chart | Test RMSE K1 vs LTAG per subset, with per-seed points | `LTAG_test_eval.json` | HIGH |
| Fig 3 | Bar/box | Per-engine ΔRMSE LS−LTAG; show 50–56% win rate and FD001 median worse | `results/ltagv2_p1_*.preds.csv` | MEDIUM |
| Table 1 | Main test | K1/LTAG/Median, Δ, wins | `LTAG_test_eval.json` | HIGH |
| Table 2 | Validation controls | K1/Median/LTAG/Global-g/K1-Calib/LS/noMW | `results/ltagv2_p1_*.json` | HIGH |
| Table 3 | Cut-wise + bootstrap | LS−LTAG by cut; cut0.95 CI | `results/ltagv2_p1_*.preds.csv` | HIGH |
| Table 4 | RHSE negative | R3/R5b/R5m by fault | `experiments/phase4/logs` | MEDIUM |
| Table 5 | Gate/decision | Gate 1/2, RHSE, route | docs | LOW (appendix) |

## Citation Plan

- §1/§2 C-MAPSS: Saxena et al. 2008 (PHM), Heimes 2008, Zheng et al. 2017,
  Li et al. 2018, and recent transformer/CNN RUL papers — **verify each in
  `docs/manuscripts/references.bib` before use; do not generate BibTeX from memory.**
- §2 aggregation/calibration: stacking (Breiman 1996), isotonic regression
  (Barlow et al.), conformal/quantile calibration — **verify**.
- §2 sensor-fault reconstruction: measurement-model / PCA reconstruction papers —
  **verify**.
- §4 LTAG: cite only the frozen protocol and the baseline backbone; no invented
  citation.
- Any unverifiable entry stays `[VERIFY]` and is excluded from the submission.

## Reviewer Feedback

- `mcp__codex__codex` returned `[WinError 2]`; `codex` CLI not on PATH.
- Reviewer unavailable → **REVIEW_UNAVAILABLE / PENDING-L2**. Do not claim external
  review. If Codex becomes available, send this outline for a 1–10 score on flow,
  claim-evidence alignment, missing experiments, positioning, page budget, and
  front-matter strength.

## Next Steps

- [ ] L2/venue decision: option B (this plan) vs A (LS-ctx main) vs C (stop).
- [ ] If B: write §1–§8; no new test; use frozen test + validation.
- [ ] If A: freeze LS-ctx and run one-shot test; main table must include win rate,
  median, and cut-wise trade-off.
- [ ] `/paper-figure` for Fig. 1–3.
- [ ] `/paper-claim-audit` against `docs/LTAGV2_TABLES.md`.
- [ ] Do **not** submit without L3 authorization.

## Generated Assets (2026-11-21)

- Draft: `paper/main.tex` + `paper/sections/0_abstract.tex` … `8_conclusion.tex`;
  `paper/references.bib` (5 verified keys); `paper/README.md`.
- Figures: `figures/fig1_cutwise.pdf`, `figures/fig2_test.pdf`,
  `figures/fig3_winrate.pdf`; LaTeX includes in `figures/latex_includes.tex`.
- Tables: `tables/tab1_test.tex`, `tab2_validation.tex`, `tab3_cutwise.tex`,
  `tab4_rhse.tex`, `tab5_gates.tex`.
- Reproducible scripts: `paper_b/eval/make_figures.py`,
  `paper_b/eval/make_tables.py`, `paper_b/eval/export_table_data.py`.
- Data: `results/ltagv2_p1_*.json`, `results/ltagv2_p1_*.preds.csv`,
  `results/cutwise.json`, `results/phase4.json`, `results/SHARED_R0.json`.
