# 5 Guidelines: ten checks before believing a C-MAPSS comparison

Each guideline below is derived from a measurement made in this paper, not from
general advice. The evidence column points to the section that supports it, and
the tool column names something a reader can run.

| # | Check | Evidence | Tool |
|---|---|---|---|
| G1 | **Report the engine count, not the window count.** Windows from one engine are not independent observations; a 30k-window test set is still 52 engines. | §4.1: engine sampling is 97.6% of the variance of a paired contrast | `scripts/mde.py` |
| G2 | **Compute the minimum detectable effect for your own engine count before claiming a win.** | §4.2: MDE 1.46 RMSE at 260 engines (FD002), 2.38 at 20 (FD001) | `scripts/mde.py --residuals …` |
| G3 | **Do not answer an underpowered comparison with more seeds.** Seeds address the 2.4% component. | §4.1: seed share 2.4%, engine share 97.6% | `scripts/dsh_e9_variance_decomp.py` |
| G4 | **Hold out every engine at least once (k-fold), not once in total (single 80/20).** | §4.3.3: 5-fold cross-validation turns a non-significant 52-engine contrast into a 260-engine one | `scripts/run_candidate.py --fold k` |
| G5 | **Score more than one cut per engine when the pipeline allows it.** Two cuts at fixed engine count cut the interval to two-thirds. | §4.5: width ratio 0.66–0.75 | `scripts/eval_resolution.py` |
| G6 | **State the cut policy explicitly.** "Matched per engine", "fixed absolute cycle" and "random deployment cut" are not interchangeable. | §4.5: the same time contrast is +2.69 under matched cuts and ≈0 under a fixed cut at cycle 100 | `scripts/eval_resolution.py` |
| G7 | **Ablate a time feature by destroying its alignment, not only by deleting it.** Deleting it confounds "no time" with "one fewer input"; scrambling its alignment isolates the information. | §4.3.2: scrambled-age control is +3.51 RMSE worse than the aligned encoding, and statistically identical to removing time | `--variant PS` |
| G8 | **Say which cap was applied to the target and where.** A cap change of 125→130 moves the reported RMSE by more than most method differences. | §4.5 | `run_candidate.py --max-rul` |
| G9 | **Define the late/severe metric by an explicit error threshold.** The legacy proxy under-states severe late reporting by roughly 5×. | §4.5 | `run_candidate.py` (`severe_e10`) |
| G10 | **Ship the per-engine predictions and the code hash with the table.** Without them a contrast cannot be re-derived, and a claim that looked multiplicative can turn out to have no artefact behind it at all. | §4.2 note on the withdrawn 21-pair table; §7 | `metrics.json` → `code_sha256`, `matched_cutoff_predictions.csv` |

## How to use the list

The checks are ordered by cost, not by importance. G1–G3 take minutes and
require no retraining; they only need the per-engine residuals that a
well-formed submission should already contain. G4–G5 cost GPU time but are
mechanical once the pipeline supports folds and multiple cuts. G6–G9 are
reporting conventions: they cost nothing and are the most common cause of two
papers disagreeing about the same dataset.

A worked example of the whole list applied to one contrast is section 4.3: the
same time-channel ablation is (i) large and significant on 52 FD002 engines,
(ii) directionally identical but below the MDE on 20 FD001 engines, (iii) zero
under a fixed absolute cut, and (iv) fully explained by alignment rather than
capacity once the scrambled control is run. Four different true statements about
one design change, separated only by evaluation choices.

## What the list does not do

It does not make a weak method strong, and it does not turn a null result into a
positive one. Its purpose is narrower: to stop a paper from reporting the seed
component as if it were the method, and to make the resolution of a claim
explicit before the claim is made.
