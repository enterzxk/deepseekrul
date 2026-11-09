# 3. Experimental protocol

<!-- draft note: numbers must match CLAIM_LEDGER.md (C-ids are stripped at typesetting);
     all values are recomputed by scripts/freeze_audit.py (50/50) and paper_evidence.py (17/17). -->

## 3.1 Data

We use the four C-MAPSS turbofan subsets FD001–FD004 [@saxena2008]. They differ in the number of
operating conditions (one for FD001/FD003, six for FD002/FD004) and in the number of fault modes
(one for FD001/FD002, two for FD003/FD004). Training targets are the piecewise-linear remaining
useful life clipped at 125 cycles. Windows of 50 cycles are built per engine; FD002/FD004 are
condition-normalised with six operating-condition clusters fitted on training engines only.

The official C-MAPSS test files are **not** used anywhere in this paper. All results come from
held-out engines inside the training data. This is a deliberate protocol choice: the question is how
a conformal interval behaves on unseen engines drawn from the same population, not how it scores on
a fixed test file. <!-- C-ledger: protocol note -->

## 3.2 Splits and seeds

Engines are partitioned into five folds by `KFold(n_splits=5, shuffle=True, random_state=20261104)`.
Within each fold, 30% of the validation engines are reserved for conformal calibration and the rest
form the test set; the split is engine-disjoint and fixed at seed 42 for every cell. The number of
calibration engines is therefore 6 (FD001, FD003) and 16 / 15 (FD002, FD004); the corresponding test
sets contain 14 / 36 / 14 / 35 engines. <!-- C-ledger: 6/16/6/15 -->

Each fold is trained with four seeds (42, 7, 2024, 13), giving 4 subsets × 5 folds × 4 seeds = **80
cells**. Seeds change only the initialisation and training trajectory; the engine partition is shared
by all four seeds.

## 3.3 Frozen predictor and interval

The predictor is an IDA-PE backbone with a zero-initialised projection of two prior features and two
output heads producing a centre $c_i$ and a non-negative half-width $s_i=\mathrm{softplus}(g_\phi(z_i))$:

$$
q^{lo}_i = c_i - s_i, \qquad q^{hi}_i = c_i + s_i .
$$

It is trained end-to-end with the pinball loss at 0.05/0.95 plus $0.1\,(c_i-y_i)^2$. Calibration is
split conformal on the CQR score $E_i=\max(q^{lo}_i-y_i,\ y_i-q^{hi}_i)$:

$$
\hat q = \mathrm{Quantile}_{(1-\delta)(1+1/n)}\big(\{E_i\}\big), \qquad
C(x) = \big[q^{lo}-\hat q,\ q^{hi}+\hat q\big],
$$

with $\delta=0.1$ and $n$ the number of calibration windows. The method is frozen before any of the
analyses below; it is treated as a baseline, not as a contribution. <!-- C-ledger: frozen baseline -->

## 3.4 Four arms

| Arm | Centre | Half-width | Purpose |
|---|---|---|---|
| **Constant width** | deterministic $c_i$ | $\hat q$ (absolute-residual conformal) | the frozen baseline the literature compares against |
| **CQR** | $c_i$ | $s_i+\hat q$ | the method under audit |
| **MC-Dropout scale** | mean of 30 stochastic forward passes | $\hat q\cdot \mathrm{sd}_i$ | a per-sample epistemic-uncertainty scale |
| **Same-centre constant** | *same* MC mean | $\hat q$ | **control**: isolates the scale from the centre shift |

The fourth arm is the key design element. MC-Dropout changes both the centre (to the MC mean) and the
scale; comparing it with the deterministic-centre constant width confounds the two. Comparing it with
the same-centre constant width attributes the difference to the scale alone.
<!-- C6 same-centre control -->

## 3.5 Metrics

The primary metric is the Winkler interval score at a fixed $\alpha=0.1$,
$\mathrm{mean}\big[(u-l)+\tfrac{2}{\alpha}((l-y)_+ + (y-u)_+)\big]$, which penalises width and
miscoverage jointly. We also report PICP (marginal coverage), PINAW, and the **per-engine coverage
distribution** (minimum, 5th percentile, median). The engine, not the window, is the unit for all
coverage statements: thousands of windows from a few engines are not independent samples.

## 3.6 Statistical discipline

The reporting unit is the **fold** (4 subsets × 5 folds = 20), with seeds averaged inside a fold.
Three caveats are stated rather than corrected away:

1. The 80 cells share one five-fold partition, so they are not 80 independent observations.
2. The 20 folds share training data through cross-validation overlap, so we report descriptive
   statistics and bootstrap intervals, not independent-sample significance tests.
3. Coverage-aligned comparisons interpolate each method's frontier using test labels and therefore
   cannot represent a deployable selection rule; they appear in Appendix A as a post-hoc diagnostic
   only, clearly separated from the nominal-level results.

## 3.7 Reproducibility and integrity gates

Three executable gates precede every number in this paper:

- **Frozen baseline** (`scripts/freeze_audit.py`, 50 assertions): re-derives the 80-cell table,
  the fold-level summaries, the seed-variance decomposition, the per-engine coverage tails, the
  permutation ablation and the cross-domain results from the stored predictions.
- **MC-Dropout integrity** (`scripts/verify_mcd80.py`): the 40 cells that required retraining were
  re-run from scratch, and their point predictions reproduce the frozen ones **bit-for-bit**
  (max $|\Delta y_{lo}| = 0.0$).
- **Sampling noise floor** (`scripts/compare_mcd_draws.py`): two independent MC-Dropout draws on the
  same checkpoints give a median absolute change of 0.63 pp, a 90th percentile of 3.20 pp, and a
  maximum of 14.70 pp on the primary metric, with **0/80 direction flips**. Subset-level differences
  smaller than 3.20 pp are therefore reported as within noise.

The protocol is frozen before the analyses in Section 4; the pre-registered diagnostic test of
Section 4.4 is registered in the project's hypothesis log with its thresholds fixed in advance.
<!-- C9 -->
