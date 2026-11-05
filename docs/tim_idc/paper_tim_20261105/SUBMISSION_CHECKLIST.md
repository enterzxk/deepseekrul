# IEEE TIM submission checklist

**Manuscript:** *When Does a Causal Lifetime Prior Help Remaining Useful Life Prediction? A Budget-Controlled Study on C-MAPSS*

**Target:** IEEE Transactions on Instrumentation and Measurement (TIM), regular paper.

## 1. Files

- [ ] `main.tex` compiles with IEEEtran (pdflatex) and produces `main.pdf`.
- [ ] Figures: `figures/fig_schematic.pdf`, `fig_full_rmse.pdf`, `fig_forest.pdf`, `fig_measurement.pdf`, `fig_budget.pdf`.
- [ ] Tables: `tables/tab_full.tex`, `tab_arms.tex`, `tab_light.tex`, `tab_perseed_full.tex`, `tab_measurement.tex`, `tab_replication.tex`, `tab_context.tex`, `tab_metrics_full.tex`, `tab_metrics_light.tex` (supplementary).
- [ ] Bibliography: 21 `\bibitem` entries, all cited; no undefined citations.
- [ ] No undefined references; all `\ref` resolved.
- [ ] Author names, affiliations, funding, and corresponding author filled in.
- [ ] Check TIM's current anonymity policy before upload.

## 2. Compile checks (compiled on the server with TeX Live 2026)

- [x] Compile with pdfLaTeX; `main.pdf` is built (9 pages, 0 overfull boxes).
- [x] Fix overfull/underfull boxes; the only warnings are minor underfull boxes in the references.
- [x] Check figure placement and that `figure*` spans both columns; visually checked pages 3–7.
- [x] Check the abstract length (about 250 words) against the TIM limit.
- [x] Check page count against the TIM regular-paper limit (9 pages; the limit is 10).
- [x] Check that all five figures are referenced in the text (Figs. 1–5).

## 3. Claims and evidence

- [ ] The paper does **not** claim SOTA or a robust improvement over PE.
- [ ] The paper does **not** claim calibrated uncertainty (Gate 2 fails; σ_T is degenerate).
- [ ] The paper does **not** claim a conformal reliability contribution as the main result (intervals 33–83 cycles wide).
- [ ] The paper reports the accuracy-safety trade-off: the RMSE gain is significant only on FD003; the NASA score improves significantly on FD003 and FD004; MAE does not improve significantly; FD002 is worse than OPS on RMSE/MAE.
- [ ] The lightweight four-metric table is supplementary; Table VIII is the full-budget four-metric table.
- [ ] The adapted CET-MSA is labeled as an adaptation, not the authors' official protocol.
- [ ] The historical E22a/IDA numbers are labeled as a different caliber.
- [ ] The one-shot official test is described accurately: 96 predictions sealed, labels opened once, no test-driven tuning.
- [ ] The per-seed table uses the actual CIs from the scoring JSON.
- [ ] The text number audit passed (0 flagged 2+ decimal numbers outside the tables/whitelist).

## 4. Figures

- [ ] Vector PDF, fonts embedded, line widths readable at column width.
- [ ] Schematic (Fig. 1) labels match the method section.
- [ ] Full-budget RMSE (Fig. 2) and forest plot (Fig. 3) match Tables 1 and 3.
- [ ] Fixed-age MAE (Fig. 4) matches Table 5.
- [ ] Budget interaction (Fig. 5) matches Tables 1 and 3.
- [ ] Color-blind-friendly palette; grayscale check.

## 5. Tables

- [ ] Captions are self-contained; significance marks (bold) explained.
- [ ] `tab_context` is a `table*` (full width); the other wide tables are `table*`.
- [ ] `tab_arms` fits a single column; check column widths.
- [ ] No duplicated table numbers; the text refers to Tables I–VIII in order; Table IX is supplementary.

## 6. References

- [ ] Verify DOIs for CET-MSA (10.1007/s10845-026-02895-3) and BootBoGs-TS (10.1007/s44196-026-01344-6).
- [ ] Verify the recent TIM references (CRULP, Hou, Yang, Wang, Kim, Zhou).
- [ ] TIM reference style; check abbreviations and page numbers.
- [ ] All 21 entries are cited; no uncited entries.

## 7. Data and code availability

- [ ] C-MAPSS is public (NASA PCoE); state the source.
- [ ] The code, protocols, sealed predictions, and hashes are archived with the paper.
- [ ] The one-shot protocol and the 96-prediction seal are described in the appendix.
- [ ] The official test labels were opened once; no test-driven fitting.

## 8. Cover letter

- [ ] Use `COVER_LETTER.md`; fill author/affiliation placeholders.
- [ ] State the contribution honestly: a budget-controlled measurement study, not a new SOTA architecture.
- [ ] Explain the fit to TIM: measurement/instrumentation, lifetime prior as a measurement instrument, fixed-age analysis.
- [ ] Confirm originality and no concurrent submission.

## 9. Known gaps

- [x] TeX Live 2026 is installed at `/home/user/texlive` on the server; `main.pdf` is compiled. The local Windows machine still has no TeX; `OVERLEAF.md` is the fallback.
- [x] Text number audit passed; a final human read of the abstract/introduction is still recommended.
- [x] The paper is a measurement study; the RMSE result is negative, but the NASA safety score improves significantly on FD003/FD004. If the target requires a "new method + clearly better RMSE" contribution, this manuscript does not meet that bar.
