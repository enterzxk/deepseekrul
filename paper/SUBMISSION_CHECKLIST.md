# Submission Checklist — LTAG v2 Draft

**Date**: 2026-11-21
**Status**: draft, not submission-ready.

## Done

- [x] Paper plan (`PAPER_PLAN.md`) with claims-evidence matrix.
- [x] 8-section IEEE journal draft (`paper/main.tex` + `sections/`).
- [x] Appendix A: gate summary, frozen configuration, reproducibility paths.
- [x] 5 LaTeX tables (`tables/tab1`–`tab5`).
- [x] 3 figures (`figures/fig1_cutwise`, `fig2_test`, `fig3_winrate`) with
  `latex_includes.tex`.
- [x] Frozen test numbers audited against `LTAG_test_eval.json` (15/15, +0.391).
- [x] Validation numbers audited against `results/ltagv2_p1_*.json`.
- [x] Cut-wise numbers audited against `results/cutwise.json`.
- [x] RHSE numbers audited against `results/phase4.json`.
- [x] Citation keys checked against `paper/references.bib` (5 keys, no
  missing/uncited).
- [x] `\ref`/`\label` static check: no missing labels.
- [x] LaTeX environment balance check: `ENV_OK`.
- [x] Local claim audit (`paper/PAPER_CLAIM_AUDIT.md`, verdict WARN).

## Not done

- [ ] **PDF compile.** No `pdflatex`/`latexmk` on this machine. Need a TeX
  installation with `IEEEtran.cls`/`IEEEtran.bst`.
- [ ] **Page count.** IEEE journal limit; references count. Not measurable
  without a compile.
- [ ] **Independent claim audit.** `mcp__codex__codex` returns `[WinError 2]`;
  no cross-model reviewer. Marked `PENDING-L2`.
- [ ] **External scientific review.** Same blocker.
- [ ] **Figure quality review by reviewer model.** Same blocker.
- [ ] **Submission.** Requires L3 authorization; not requested.

## Evidence status

| Item | Status |
|---|---|
| Test | frozen, not rerun; 15/15, +0.391 |
| Validation | 5 cuts, seeds 101/202/303 (+404/505/606 FD001) |
| Gate 1 | FAIL (K1-Calib/LS 4/4 beat LTAG) |
| Gate 2 | not started (Gate 1 failed) |
| RHSE-TSH | removed (only stuck passes fair baselines) |
| Phase 2 end-to-end | not used in this paper |
| Disk | 98% used (96 GB free); no new large experiments |

## Recommended next actions

1. Provide a TeX toolchain (or a machine with `pdflatex`) and compile
   `paper/main.tex`; fix layout/page-count issues.
2. When Codex is available, run a fresh `paper-claim-audit` and a figure review.
3. Decide A/B/C (LS-ctx main vs LTAG test main vs stop); the draft implements B.
4. Do not submit without L3 authorization.
