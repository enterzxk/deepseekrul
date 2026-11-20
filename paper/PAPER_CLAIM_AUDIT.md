# Paper Claim Audit Report

**Date**: 2026-11-21
**Paper**: *Cut-Dependent Inference-Time Aggregation for C-MAPSS Remaining Useful
Life: When Learned Gating Helps and When It Does Not*
**Auditor**: local scripted audit (`tmp/local_audit.py`) against frozen JSONs.
**External zero-context reviewer**: unavailable (`mcp__codex__codex` returns
`[WinError 2]`; `codex` not on PATH) → **PENDING-L2**. This report does not claim
an independent cross-model audit.

## Overall Verdict: WARN

- Local numerical reconciliation: **PASS** (all checked claims exact or standard
  rounding).
- Independent reviewer: **not run**; the verdict is therefore WARN, not PASS.
- No claim was found to be inflated, cherry-picked, or config-mismatched in the
  local check.

## Claims Verified: 15 groups

| # | Location | Claim | Evidence | Status |
|---|---|---|---|---|
| 1 | Abstract / Intro | Test 15/15, mean +0.391 | `LTAG_test_eval.json`: 6/6, 3/3, 3/3, 3/3; deltas +0.426/+0.162/+0.656/+0.321; mean +0.391 | exact |
| 2 | Experiments | FD001 13.421→12.995 (+0.426) | same JSON | exact |
| 3 | Experiments | FD002 11.514→11.352 (+0.162) | same JSON | exact |
| 4 | Experiments | FD003 12.979→12.323 (+0.656) | same JSON | exact |
| 5 | Experiments | FD004 13.031→12.710 (+0.321) | same JSON | exact |
| 6 | Table 2 | K1 12.094/11.527/12.793/13.608 | `results/ltagv2_p1_FD00*.json` | exact |
| 7 | Table 2 | LTAG 11.334/11.210/12.372/13.522 | same | exact |
| 8 | Table 2 | K1-Calib 10.931/10.970/12.129/12.839 | same | exact |
| 9 | Table 2 | LS-ctx 10.150/10.604/12.033/13.080 | same | exact |
| 10 | §5.3 | Ens3-K1 11.005/11.231/11.828/13.342; Ens3-LTAG 10.292/10.937/11.410/13.228 | `results/ltagv2_p1b_FD00*.json` | exact |
| 11 | Table 3 | LS−LTAG by cut (−4.170…+0.851) | `results/cutwise.json` | exact |
| 12 | §6.2 | cut0.95 bootstrap: +0.252 [0.013,0.495], +0.411 [0.078,0.749], +0.592 [0.221,0.964], +0.851 [0.120,1.626], −0.115, −0.276 | `results/ltagv2_p1_*.json` | exact |
| 13 | §6.3 | per-engine wins 10/20, 28/52, 10/20, 28/50 | `ltagv2_winrate.py` on preds | exact |
| 14 | Table 4 | RHSE R3/R5b/R5m values | `results/phase4.json` | exact |
| 15 | Table 5 | Gate 1 FAIL; G1.1 0/4 mean −0.393; G1.2 0/4; G1.3 PASS; G1.4 2/4 | `docs/LTAGV2_P1_BASELINES.md`, `results/ltagv2_p1_*.json` | exact |

## Arithmetic Checks

- Mean test delta: (0.426 + 0.162 + 0.656 + 0.321) / 4 = **0.39125** → 0.391. OK.
- Test wins: 6 + 3 + 3 + 3 = **15/15**. OK.
- Gate G1.1 mean: (0.404 + 0.240 + 0.243 − 0.683) / 4 = **−0.393**. OK.
- RHSE stuck margins: FD001 0.737 − max(−0.695,−0.202) = **0.939** → 0.940 (rounding). OK.
- FD003 stuck: 0.605 − max(−0.179,−0.288) = **0.784** → 0.783 (rounding). OK.

## Issues / Caveats

1. **No independent reviewer.** The skill requires a fresh cross-model thread;
   Codex is unavailable. The audit is local and may share the executor's
   confirmation bias. Marked WARN / PENDING-L2.
2. **Test is frozen, not pristine blind.** The paper states this in
   Section~\ref{sec:conclusion}; the audit confirms the wording does not claim a
   new blind test.
3. **Estimator mixing.** The paper reports engine-mean and pooled separately and
   does not subtract one from the other. No violation found.
4. **LS-ctx validation numbers** come from per-seed rows (Phase 3) while the
   table/§5 uses the seed-ensemble/per-seed table; the paper consistently uses
   the `ltagv2_p1` numbers for Table 2 and does not mix 9.735 with 9.256. OK.

## Verdict

**WARN — local reconciliation passes; independent review pending.** Do not mark
submission-ready until a cross-model claim audit is run and the PDF is compiled.
