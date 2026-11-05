# Overleaf compile guide

**Status:** the 3090 server now has TeX Live 2026 at `/home/user/texlive`; `main.pdf` is compiled (9 pages, 0 overfull boxes), including the four-metric Table VIII. This guide remains a fallback for Overleaf or for a machine without TeX. The local Windows machine still has no TeX.

## 1. Create the project

1. New Overleaf project → Blank Project.
2. Delete the default `main.tex`.
3. Upload `main.tex`, the `tables/` folder (7 `.tex` files), and the `figures/` folder (5 `.pdf` files; the `.png` files are optional).
4. Keep the folder names exactly `tables/` and `figures/`; the manuscript uses `\input{tables/...}` and `\includegraphics{figures/...}`.

## 2. Settings

- Compiler: **pdfLaTeX**.
- TeX Live version: 2024 or newer.
- Main document: `main.tex`.
- The class is `IEEEtran` (available in Overleaf by default).

## 3. Compile

1. Click Compile. The first pass may report undefined references/citations; compile again.
2. Expected benign warnings: overfull hboxes from the wide `table*` tables, and `cite` range formatting.
3. Check the log for:
   - `File 'tables/tab_*.tex' not found` → the folder structure is wrong.
   - `File 'figures/fig_*.pdf' not found` → the figure files were not uploaded, or the names differ.
   - `Citation 'refN' undefined` → the `thebibliography` block is missing or the key is mistyped; all 17 keys are in `main.tex`.
   - `Reference 'fig:...' undefined` → compile twice.

## 4. Layout checks

- The five figures: schematic (Fig. 1), full-budget RMSE (Fig. 2), forest (Fig. 3), fixed-age MAE (Fig. 4), budget interaction (Fig. 5).
- The seven tables: full-budget official test (Table 1), arms (Table 2), lightweight (Table 3), per-seed (Table 4), measurement (Table 5), replication (Table 6), historical context (Table 7).
- If a `table*` overflows, add `\small` before `\begin{tabular}` or reduce the column count; do not change the numbers.
- Check that `tab_context` is full width (`table*`) and `tab_arms` is single column.
- Check page count against the TIM regular-paper limit.

## 5. Before submission

- Fill `\author{...}` with names, affiliations, funding, and the corresponding author.
- Add the TIM `\thanks` block if required by the template.
- Check the abstract length against the TIM limit (currently about 220 words).
- Check that all five figures are referenced (they are: Figs. 1–5).
- Download the PDF and run the checklist in `SUBMISSION_CHECKLIST.md`.

## 6. Known gaps

- The manuscript has not been compiled; the first compile may reveal minor overfull boxes.
- The author block is a placeholder.
- The paper is a measurement/negative study; do not add SOTA or calibrated-uncertainty claims during editing.
