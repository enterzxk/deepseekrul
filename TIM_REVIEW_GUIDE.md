# TIM submission — review guide

**Read this file first.** The repository root and `README.md` belong to *other*
research lines (a CRGraphTransformer / PRC track). None of that is part of this
submission. Everything for the paper under review lives in `paper/`, `scripts/tim/`
and `experiments/tim/`.

## The paper

*Title:* What a Remaining-Useful-Life Interval Can Prove: Coverage, Width, and the
Calibration Object.

*Target:* IEEE Transactions on Instrumentation and Measurement.

*Thesis:* the coverage/width pair that RUL papers report is partly information and
partly reporting convention. Three conventions are priced on 653 C-MAPSS test engines
and 2 587 N-CMAPSS queries, and a construction is proposed that removes the largest
of them.

It is a measurement paper, not a point-accuracy paper. The backbone is deliberately
mid-tier (dev RMSE 11.1 / 13.9 / 10.5 / 13.9 on FD001–FD004) and is held fixed
throughout; only the interval construction varies.

## Reading order

1. `paper/TIM_MANUSCRIPT.pdf` — 10 pages, compiled. Start here.
2. `paper/TIM_MANUSCRIPT.tex` — the source, if you want to check a claim against
   the exact wording.
3. `paper/TIM_RESULTS.md` — the working ledger (Chinese): every table, every number,
   including results that did **not** make it into the paper and why.
4. `scripts/tim/` — the code that produced each table.
5. `experiments/tim/*.json` — the raw outputs.

## What has already been checked

Do not spend review effort re-deriving these; the numbers are in the JSON files.
Do attack the *reasoning* behind them.

- **Leave-one-out ablation** (`experiments/tim/ablation.json`). Variance
  normalisation is worth +5.0 Winkler points when removed (CI excludes zero on three
  fleets); none of the seven negative increments excludes zero.
- **Tail-allocation estimability** (`experiments/tim/gamma.json`). In simulation a
  cross-fitted five-fold choice recovers the selection cost; on the benchmark it does
  not (up to 0.078 of the admissible range of γ, 5.0 Winkler points worse). The paper
  therefore reports the free-tail gain as an upper bound and the ablation as a lower
  bound.
- **Second backbone** (`experiments/tim/backbones.json`, `matrix_B.json`). A 2-layer
  BiLSTM with attention, 119 490 parameters, 60 training runs. Width ordering
  replicates on all four fleets; the coverage level does not, and the paper says so.
- **Four nominal levels** (`experiments/tim/levels.json`, `matrix_lv80.json`,
  `matrix_lv95.json`). The convention cost is monotone in the level (32.9 % at 0.50
  down to 7.9 % at 0.95).
- **McNemar** (`experiments/tim/measure.json`). Paired test for free tail against
  symmetric band; significant in both directions on different fleets.
- **Detectability** (`measure.json`). 11–82 engines per RUL bin detect only 0.14–0.44
  coverage deviations at family-wise 5 % and 80 % power; 149 engines per bin are
  needed for 0.10.

## Known gaps, stated in the paper

- **Both benchmarks are simulations.** C-MAPSS and N-CMAPSS are run-to-failure
  simulators; there is no real sensor data in this study. §VI-C says so explicitly.
  The public run-to-failure bearing datasets (XJTU-SY, 15 units; PRONOSTIA, 17 units)
  cannot support a coverage claim at 149 units per bin, and were not reachable from
  the machine this was written on.
- **The author block is a placeholder.** `\author{Author names withheld in this
  draft}` is deliberate; nothing else needs to change before submission.
- **The proposed construction does not win everywhere.** It is 8 % wider than CV+ on
  FD004 at the 90 % level, and the parametric layer is narrower on three of four
  fleets at the 95 % level. Both reversals are in the paper.

## Questions worth attacking

1. Is Proposition 1 correct as stated, and is it more than a repackaging of known
   conformal results on asymmetric quantile splits? Appendix A has the proof.
2. The matched-object construction calibrates on the average of *S* clean models and
   deploys the average of *K·S*. How much of the remaining gap to CV+ is the
   *S*-versus-*K·S* mismatch rather than the object mismatch it claims to fix?
3. Is the detectability argument (δ⋆) a fair criticism of published conditional-coverage
   tables, or does it assume a binning the original authors did not use?
4. The ablation scores Winkler rather than width. Does that choice favour the proposed
   construction, given that Winkler penalises misses and the construction is tuned for
   width?
5. Is 100 calibration engines enough for a 90 % interval whose tail is set by
   order statistics, and does the paper's own δ⋆ argument apply to its calibration set
   as well as to its test set?
6. Does the level sweep (Table III) weaken the headline claim, since the convention
   cost shrinks to 7.9 % at 0.95 while the paper's abstract quotes the 0.90 figure?
