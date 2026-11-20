# LTAG v2 Paper Draft

Status: **draft, not submitted**. Test was not rerun; all test numbers are from the
frozen `LTAG_test_eval.json`. Validation numbers are from
`results/ltagv2_p1_*.json` and `results/ltagv2_p1_*.preds.csv`.

## Compile

Requires `IEEEtran.cls` and `IEEEtran.bst`.

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Figures are resolved via `\graphicspath{{../figures/}}`; tables are included from
`../tables/`.

## Scope

- Title: *Cut-Dependent Inference-Time Aggregation for C-MAPSS Remaining Useful
  Life: When Learned Gating Helps and When It Does Not*.
- 8 sections; IEEE journal style; no author information.
- Claims C1--C6 from `PAPER_PLAN.md`.
- Test: 15/15, mean +0.391 (frozen B0).
- Validation: K1-Calib/LS-ctx beat LTAG on the all-cut average; cut-wise analysis
  shows LTAG is competitive/better at 0.85--0.95.
- RHSE-TSH: negative result, removed from the method.

## Citations

Only entries present in `paper/references.bib` are cited: `saxena2008`,
`hou2025`, `frizzo2025`, `barry2026`, `srinivasan2026`. All were copied from
`docs/manuscripts/references.bib`; no BibTeX was generated from memory.

## Not done

- No external review (Codex unavailable, `WinError 2`).
- No page-count check (no local LaTeX toolchain).
- No submission (L3 authorization required).
