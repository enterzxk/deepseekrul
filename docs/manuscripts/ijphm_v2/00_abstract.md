---
title: "What conformal RUL intervals deliver: an 80-cell audit of coverage, per-engine failure, and a conditional epistemic scale"
author: "Author name(s) 鈥?placeholder, to be completed before submission"
date: "Draft, 2026-11-09"
abstract: |
  Remaining-useful-life (RUL) intervals are used to schedule maintenance, yet they are usually
  evaluated by marginal coverage on a single training run. We audit a frozen conformalized quantile
  regression pipeline on C-MAPSS at the resolution at which it is used: four subsets x five folds x
  four seeds = 80 cells, with an engine-disjoint calibration split. CQR lowers the Winkler score
  against a constant-width conformal baseline in 75/80 cells and 20/20 folds (mean -10.22%) at
  matched marginal coverage (-0.42 pp). The same cells contain engines whose intervals cover 12-49%
  of their windows, and 18/20 folds contain an engine below 0.80; the failure is repeatable across
  seeds (ICC 0.780). We then test whether a per-sample epistemic scale (MC-Dropout) repairs this.
  Against a same-centre constant-width control, the scale improves 19/20 cells on FD002 but only
  1/20 on FD003; at the fold level, 9/10 multi-operating-condition folds improve versus 2/10
  single-condition folds. A pre-registered calibration-side diagnostic fails its support criteria
  (-0.039 [-0.431, 0.317]), because with six calibration engines its own bootstrap standard
  deviation is 0.41-0.49. For practice, the results argue for reporting per-engine coverage and
  replicating across seeds before an interval is used in a maintenance decision.
keywords: "conformal prediction; remaining useful life; conditional coverage; C-MAPSS; uncertainty quantification; maintenance decision making"
---


