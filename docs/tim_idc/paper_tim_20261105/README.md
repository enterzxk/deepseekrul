# TIM paper — budget-controlled lifetime prior

Status: **compiled 9-page manuscript** — `main.pdf` built with TeX Live 2026; figures, tables (including the four-metric Table VIII), citations, checklist, cover letter; all pages visually reviewed. The lightweight four-metric table is supplementary. The author block is the only user-supplied field.

## What the paper claims

The paper is a measurement/empirical study, not a new SOTA architecture:

- A strictly causal, nested-cross-fitted lifetime prior is injected into a PE backbone.
- Under a **lightweight** budget (24 epochs, stride 5), the sensor prior improves over PE on FD001/FD003/FD004; FULL−OPS is significant on all four subsets.
- Under a **full** budget (80 epochs, all windows), the RMSE gain is significant only on FD003; on FD002 the sensor prior is significantly worse than an age/condition-only prior (+0.590 RMSE, CI [+0.036, +1.148]).
- The asymmetric NASA score improves significantly on FD003 and FD004 (largest on FD004), while MAE does not improve significantly; the prior shifts predictions earlier, trading accuracy for safety. On FD002 the age/condition prior is better on RMSE, MAE, and NASA.
- A fixed-age measurement analysis shows sensor change/slope features carry real lifetime information (8/8 engine-paired CIs; 11.8–42.9 cycle MAE reductions), so the negative end-to-end result is about the interface/budget, not uninformative sensors.
- Uncertainty is not claimed: the prior’s σ_T is degenerate (max ~2.84e−14), Gate 2 Spearman is 0.034 on FD002, and the post-hoc conformal layer is over-conservative.

## Files

| File | Content |
|---|---|
| `main.tex` | IEEEtran manuscript (citations, 7 tables, 5 figures) |
| `MANUSCRIPT.md` | Markdown manuscript draft |
| `tables/tab_full.tex` | Full-budget official test (main table) |
| `tables/tab_light.tex` | Lightweight official test (controlled contrast) |
| `tables/tab_perseed_full.tex` | Per-seed full-budget contrasts (actual CIs from the scoring JSON) |
| `tables/tab_measurement.tex` | Fixed-age total-lifetime MAE |
| `tables/tab_replication.tex` | Three-seed outer-validation replication |
| `tables/tab_context.tex` | Historical IDA context (different caliber) |
| `tables/tab_arms.tex` | Arms and budgets |
| `tables/tab_metrics_full.tex` | Full-budget four-metric table (Table VIII) |
| `tables/tab_metrics_light.tex` | Lightweight four-metric table (supplementary, Table IX) |
| `data/multi_metric_ci.json` | Four-metric ensemble values and bootstrap CIs |
| `figures/` | Five publication figures (PDF + PNG) |
| `data/official_results.json` | Lightweight official test, one-shot |
| `data/official_fullbudget_results.json` | Full-budget official test, one-shot |
| `data/study_results.json` | First-round 80-model study |
| `data/replication_results.json` | Three-seed replication |
| `data/bridge_results.json` | Old-protocol bridge |
| `data/measurement_FD00X.json` | Fixed-age measurement results |
| `data/conformal_results.json` | Outer-val conformal calibration |
| `data/OFFICIAL_TEST_SUMMARY.md` | One-shot result summary |
| `data/PAPER_DECISION.md` | Decision memo after the test |
| `SUBMISSION_CHECKLIST.md` | TIM submission checklist |
| `OVERLEAF.md` | Overleaf upload and compile guide |
| `COVER_LETTER.md` | Draft cover letter |

## Evidence map

- **Main performance table**: `tab_full` / Table 1. Full-budget official test; the negative result is the paper’s core.
- **Controlled contrast**: `tab_light` / Table 3. Lightweight official test; shows the prior can help in a budget-limited regime.
- **Per-seed**: `tab_perseed_full` / Table 4. Exposes FD002 sign reversals and FD004 variability.
- **Mechanism**: `tab_measurement` / Table 5. Sensor change/slope features are informative at fixed ages.
- **Replication**: `tab_replication` / Table 6. Three-seed outer-val; lightweight regime.
- **Context**: `tab_context` / Table 7. Old IDA early-stopped numbers; not comparable.
- **Four-metric**: `tab_metrics_full` / Table VIII. RMSE/MAE/bias/NASA with CIs; the accuracy-safety trade-off.
- **Light four-metric**: `tab_metrics_light` / supplementary Table IX.
- **Baselines**: CET-MSA adapted and official reproduction, BootBoGs-TS; protocol caveats in the text.

## What not to claim

- SOTA or a robust improvement over PE.
- Calibrated uncertainty (Gate 2 fails; σ_T is degenerate).
- A conformal reliability contribution (intervals 33–83 cycles wide).
- CET as a competitive baseline under the full budget (adapted CET 16.1–18.7; official CET-MSA is a different protocol).
- Comparability with the historical E22a test numbers (different caliber).

## Checks performed

- Tables regenerated from the JSONs: 5/6 identical; `tab_context` differs only in the `table*` environment (intentional full width).
- Text number audit: 0 flagged 2+ decimal numbers outside the tables/whitelist.
- LaTeX structural lint: braces, environment stack, `$` balance, and table column counts all pass.
- Citations: 21 `\bibitem` entries, all cited; 5 figure `\ref`/`\label` pairs match; `tab:arms` label matches the new table reference.
- Factual fix during review: the intro's maximum `σ_T` was corrected to `5.68×10⁻¹⁴` (the largest subset value; FD003).
- Compiled with TeX Live 2026 on the server: **9 pages, 0 overfull boxes**; `main.pdf` is in the paper folder.
- Four-metric CIs: engine-paired bootstrap for MAE/bias/NASA (5,000 reps, seed 99); Table VIII; the lightweight four-metric table is supplementary to avoid splitting the references.
- Layout fixes after the first compile: title line break; `tab_arms` promoted to `table*`; table references converted to `\ref` so the text matches IEEEtran's Roman table numbers.

## Remaining production steps

1. **Page check**: the manuscript compiles to 9 pages with 0 overfull boxes; check the TIM page limit (9/10) and fill the author block.
2. **Text number audit**: passed — 0 flagged 2+ decimal numbers outside the tables/whitelist; the tables match the JSONs (5/6 identical; `tab_context` differs only in the `table*` environment).
3. **Internal review**: abstract/intro claims, table captions, baseline protocol labels, and the "do not claim" list.
4. **Submission package**: `SUBMISSION_CHECKLIST.md` and `COVER_LETTER.md` drafted; fill author/affiliation placeholders and compile.

## One-shot protocol

The 96 official predictions (four subsets × four arms × three seeds × two budgets) were sealed before labels were read. Labels were opened once by `score_both.py`; no model, recipe, arm, or budget was changed afterward. The full-budget table is the primary inference; the lightweight table is a controlled contrast. The four-metric MAE/bias/NASA analysis is a post-hoc re-analysis of the same sealed predictions, not a new model selection.
