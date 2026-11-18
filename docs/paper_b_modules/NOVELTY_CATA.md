# Novelty assessment: CATA / RGTA (causal age-aligned temporal aggregation with a relative-dispersion gate)

> Scope: literature/novelty review only. No code, results or configurations were modified.
> Date: 2026-11-18. Author: DSH literature reviewer (subagent).
> Target venue assumed: IEEE TIM (method + mechanism + evidence).
> Local evidence base for the method description: `docs/paper_b_modules/TEMPORAL_AGG_PILOT_20261118.md` §2–§3.
> All bibliographic records below were resolved against Crossref (`api.crossref.org`) or arXiv `abs` pages;
> no citation in this file was taken from a search-result snippet alone unless explicitly flagged.

## 0. Blunt verdict first

**The specific combination is not, as far as this search reaches, published.** No retrieved work does
*inference-time, label-free, per-sample* aggregation of predictions from K **age-offset-removed** causal
windows of **one** model, gated by a within-sample dispersion statistic. That is the defensible core.

**But almost every constituent is prior art, and one 2026 paper owns the motivation verbatim.**
Li, Xia & Liu (Applied Sciences 2026, `10.3390/app16178865`) open by naming
"inconsistent predictions across overlapping time windows" as the problem to solve, build exactly the
earlier/current/later shifted-view construction, and add a consistency penalty plus a monotonicity
(ordering) penalty. Their answer is **retraining**; CATA's answer is **post-hoc arithmetic**. That is a real
difference, but it is a difference of *where* the fix lives, not of *what* is being fixed. Reviewers at TIM
will read that as "inference-time version of an existing training-time regulariser" unless the paper
supplies a mechanism-level argument for why the offset must be removed before dispersion is measured —
and an experiment showing the in-training route cannot do what the gate does.

**Recommendation up front:** do not sell CATA as a new ensemble, a new gate, or a new accuracy result.
Sell it as (i) the correct invariant for cross-window RUL consistency (offset removal, not variance
suppression) and (ii) a label-free per-sample rule that decides *when* that invariant is trustworthy —
with the honest headline that the rule's job is as much to **refuse** aggregation as to perform it.

---

## 1. Prior work table

Risk = probability that a knowledgeable TIM reviewer cites this work against the CATA/RGTA claim.
"Difference" is stated mechanically, not rhetorically.

| # | Work (authors, year, venue, DOI) | Topic | Mechanism — what it actually does | Difference from CATA/RGTA | Risk |
|---|---|---|---|---|---|
| 1 | **Li, B.; Xia, X.; Liu, Y.** (2026). *Multi-View Temporal Structure-Aware Learning for RUL Prediction.* Applied Sciences 16(17):8865. `10.3390/app16178865` | T1 | Builds three **temporally shifted observation views** (earlier / current / later window on the same degradation trajectory) by window displacement. Trains a dual-branch Transformer–GRN with (a) a **variance-based multi-view consistency loss** on the raw per-view predictions and (b) a **margin ranking loss** forcing earlier views to predict larger RUL. Reports C-MAPSS FD001–FD004, RMSE 13.44 / 15.41 / 13.80 / 16.63 at k=3, plus ablations. | Same *problem statement* ("inconsistent predictions across overlapping time windows"), same *view construction* (shifted causal windows). Different *intervention point and operator*: they suppress variance of **unaligned** predictions via a soft training penalty and recover the age trend only through an inequality margin; CATA removes the **known deterministic offset** `Δ` exactly, so its statistic is dimensionless, and it acts at inference with no retraining and no access to training data. They have no per-sample accept/reject, hence no way to abstain when smoothing hurts. | **High** |
| 2 | **Luo, F.; Si, G.; Fan, Y.; Wu, X.; Wang, Y.; Xia, T.; Xi, L.** (2026). *A Dual-Attention-Based Selective Ensemble Prediction Method for RUL.* IEEE Sensors Journal 26(9):13424–13433. `10.1109/JSEN.2026.3671579` | T2, T4 | Trains DATCN base learners over **multiple sliding-window lengths**; a PSO-driven **selective ensemble** prunes and weights base learners. Claims up to −14.2 % RMSE / −24.7 % Score vs a DCNN baseline on C-MAPSS. | Uses the phrase "selective ensemble" and window diversity, but selection is a **global, label-driven, one-off** decision (PSO optimised on data), fixed after training. CATA's selection is **per-sample at test time**, label-free, and chooses between *two estimators of the same quantity* rather than among models. Requires K trained models; CATA needs one. | **Medium-High** (terminology + window-diversity framing) |
| 3 | **Bharti, A.; Kim, N.H.; Kim, H.-C.** (2026). *Robust Ensemble Framework for Extrapolation and Uncertainty Quantification across Limited Time-Series Data in Prognostics.* EWSHM 2026, eJNDT 2026-08. No DOI; <https://www.ndt.net/search/docs.php?id=34011> | T2 | Ensemble of six heterogeneous models (RF, XGBoost, SVR, LSTM, CNN, Transformer) on C-MAPSS with **(i) median-absolute-deviation-based outlier filtering** of member predictions for stability, (ii) WTA³ cycle-by-cycle adaptive weighting, (iii) legacy-representative splitting. Reports ≈3 cycles RMSE, >30 % over best single model. | Closest existing use of **MAD as an ensemble gate in prognostics** — this is the single most damaging precedent for claiming the gate as novel. But the axis is **across models at one cutoff**, the action is **discard outlying members** (not choose between "aggregate" and "newest"), and weighting is performance-driven over cycles. No temporal axis, no age offset, no per-sample accept/reject of the ensemble itself. | **Medium-High** (gate component) |
| 4 | **Xia, T.; Song, Y.; Zheng, Y.; Pan, E.; Xi, L.** (2020). *An ensemble framework based on convolutional bi-directional LSTM with multiple time windows for RUL estimation.* Computers in Industry 115:103182. `10.1016/j.compind.2019.103182` | T1, T4 | Trains **several CNN-BLSTM base models at different window sizes** (30/60/90/120), routes each test unit to the models whose window fits its observed run length, then **weighted-averages** their outputs. | Multi-window ensembling for turbofan RUL, but members are **different trained models**, routing is by **test-sequence length**, aggregation is a fixed weighted mean, and no alignment or consistency statistic exists. CATA's members are the *same* model at different ages — which is exactly why the offset is known and labels are unnecessary. | **Medium** |
| 5 | **Guo, J.; Li, D.; Du, B.** (2024). *A stacked ensemble method based on TCN and convolutional bi-directional GRU with multiple time windows for RUL estimation.* Applied Soft Computing 150:111071. `10.1016/j.asoc.2023.111071` | T1, T4 | TCN + CNN-BiGRU base learners over **multiple time windows**, combined by **stacking**; ~20 % average improvement on C-MAPSS, ~10 % on N-CMAPSS. | "Multiple time windows" again means *model diversity*, combined by a learned meta-learner that needs labels. No temporal alignment, no gate, no inference-only operation. | **Low-Medium** |
| 6 | **Laine, S.; Aila, T.** (2017). *Temporal Ensembling for Semi-Supervised Learning.* ICLR 2017. `10.48550/arXiv.1610.02242` | T1 | Forms a consensus prediction by exponentially averaging network outputs **across training epochs** under different regularisation/augmentation, used as a consistency target for unlabeled data. | Pure **name collision**. "Temporal ensembling" here ensembles over *training time*, not over *input windows*, and is not an inference-time estimator. Must be cited to pre-empt the term, and the paper must not reuse the phrase without qualification. | **Low** (mechanical) / **Medium** (terminology) |
| 7 | **Liang, J.; He, R.; Tan, T.** (2024). *A Comprehensive Survey on Test-Time Adaptation under Distribution Shifts.* IJCV 2024. `10.1007/s11263-024-02181-w`; arXiv:2303.15361 | T1 | Taxonomy of TTA: test-time domain adaptation, batch adaptation, online adaptation — all of which **update model parameters on unlabeled test data before predicting**. | CATA updates nothing: no gradients, no buffers, no pseudo-labels. It is test-time *aggregation*, not test-time *adaptation*. The distinction must be stated explicitly or the contribution will be read as a weak TTA method. | **Low-Medium** (framing) |
| 8 | **Geifman, Y.; El-Yaniv, R.** (2017). *Selective Classification for Deep Neural Networks.* NeurIPS 2017. `10.48550/arXiv.1705.08500` | T2 | Risk-controlled **rejection**: calibrate a confidence threshold on held-out data so the classifier abstains to guarantee a target risk at a chosen coverage. | Foundational "gate" mechanism, but (a) classification, (b) the action is **abstain / output nothing**, whereas CATA **always predicts** and switches estimator, (c) the statistic is model confidence, not agreement among temporally related estimates. Citing this shows the gate idea is 2017-old, which is honest but also caps the claim. | **Low-Medium** |
| 9 | **Athanasopoulos, G.; Hyndman, R.J.; Kourentzes, N.; Panagiotelis, A.** (2024). *Forecast reconciliation: A review.* International Journal of Forecasting 40(2):430–456. `10.1016/j.ijforecast.2023.10.010` | T3 | Surveys **temporal hierarchies**: forecasts made at different temporal aggregation levels are made *coherent* by projecting them onto a constraint subspace, typically least-squares with an estimated error covariance. | The conceptual parent of "age alignment". CATA's constraint `RUL(c) = RUL(c−Δ) − Δ` **is** a temporal-coherence constraint. Differences: CATA's constraint is deterministic and known (no covariance to estimate), the projection is replaced by a robust median, and — the useful part — CATA turns the *residual incoherence* into a **reliability signal** rather than something to be projected away. Reconciliation theory also predicts that gains depend on the error covariance structure, which is precisely the justification for a gate. | **Medium** (conceptual mapping; favourable if engaged with) |
| 10 | **Nieves Avendano, D.; Vandermoortele, N.; Soete, C.; Moens, P.; Ompusunggu, A.P.; Deschrijver, D.; Van Hoecke, S.** (2022). *A Semi-Supervised Approach with Monotonic Constraints for Improved RUL Estimation.* Sensors 22(4):1590. `10.3390/s22041590` | T3 | Enforces **monotonicity as a training constraint** in a semi-supervised RUL model. | Training-time constraint, monotonicity only (an inequality), not the exact `−Δ` equality; no aggregation across windows, no per-sample decision. | **Low-Medium** |
| 11 | **OSM-PINN** (2026). *One-Sided Monotonic Physics-Informed Neural Network for Physically Consistent RUL Prediction.* IEEE conference, Xi'an, Sep 2026. No DOI; Xplore doc 11736982. | T3 | Asymmetric ReLU penalty that punishes **under-degradation** only, so RUL cannot increase between consecutive steps while rapid health drops remain allowed. C-MAPSS FD001–FD004, 6 seeds; violation rate 3.62 % → 2.06 %, 0.00 % on FD004. | Loss-level, requires retraining, and enforces only an ordering constraint — it cannot detect or exploit the magnitude of the age offset. Useful as evidence that "RUL increases between steps" is a recognised, still-open problem in 2026, which supports CATA's motivation without threatening its mechanism. | **Low** |

**Secondary (lower-risk) items worth one sentence each in related work, not rows:** deep ensembles
(Lakshminarayanan et al., NeurIPS 2017) and snapshot/SWA averaging — all ensemble estimates of the *same*
target and therefore need weights or variance estimates, whereas CATA's members estimate targets that
differ by a known constant; classical particle-filter / Kalman RUL smoothers — these smooth a latent state
with a motion model, i.e. they need a process model and are not model-agnostic wrappers; test-time
augmentation in vision/time series — TTA aggregates over *label-preserving* perturbations, and a temporal
shift is **not** label-preserving for RUL, which is precisely why TTA logic cannot be transplanted without
the offset correction.

**Unverified lead (do not cite as-is):** an EWSHM 2018 paper (Todd, `ndt.net/article/ewshm2018/papers/0142-Todd.pdf`)
appears, from indexed text, to report that for overlapping regression windows the **median is the least
volatile** predictor versus the mean. The PDF was not retrievable in this session, so this is recorded as a
lead only. If it holds, it is a direct precedent for "median beats mean over windows" and should be cited
before someone else finds it.

---

## 2. Top three novelty risks

### Risk 1 — The problem statement is already published, and the published fix is *in-training* (highest)

Li, Xia & Liu (2026) state CATA's motivation almost word for word: RUL models "typically model different
time windows as independent samples, lacking explicit constraints on the temporal structure relationships
across windows", so "when the input window experiences slight time shifts … the model output is prone to
significant fluctuations, even producing non-monotonic predictions". They then build the same
earlier/current/later shifted views and add a consistency penalty plus an ordering penalty.

Why this hurts: the natural reviewer question is **"why not just retrain with their loss?"** — and the
answer cannot be "because ours is training-free" alone, since training-free is a convenience, not a
mechanism. The answer has to be that their objective is *mis-specified for the quantity at hand*: a variance
penalty on **raw** predictions from views separated by `k` cycles pushes `RUL(c)` toward `RUL(c−k)`, which is
false by exactly `k`; the ordering margin only restores the sign of the trend, not its magnitude. CATA
removes the known offset first, so its dispersion is a genuine noise statistic rather than a mixture of
noise and true ageing. **This argument must be made explicitly and, ideally, demonstrated** (see §3, C4) —
otherwise the paper reads as a weaker, post-hoc variant of a training-time method that also reports better
absolute RMSE.

### Risk 2 — The gate is textbook robust statistics, and MAD gating already exists in prognostics

`MAD(a)/|median(a)| ≤ τ` is a robust coefficient of variation with a fixed cut-off — textbook robust
statistics, not a contribution. Worse, Bharti, Kim & Kim (2026) already use **MAD-based filtering as the
stability mechanism inside a C-MAPSS ensemble**, and Geifman & El-Yaniv established calibrated threshold
gating in 2017. A reviewer who knows either will not accept "we propose a dispersion gate".

Why this hurts: any sentence of the form "we propose a novel gating criterion" is a rejection risk. The only
defensible claim is about **what the statistic is computed on and what it decides**: dispersion of
*offset-removed* estimates of the *same* physical quantity, used to choose between two estimators per
sample, with no labels at any point. Note also that τ = 0.05 is currently a magic number; without a
principled selection story the gate looks tuned.

### Risk 3 — The effect is subset-conditional, so the gate may be doing nothing but "reproduce K=1"

From the pilot: ungated K=3 median helps FD001 on all six seeds (mean +0.533 RMSE, t ≈ 5.6) but **hurts
FD003 and FD004 clearly and is mixed on FD002**. The gate then "works" on FD004 by **rejecting every
aggregation** (τ = 0.05 leaves FD004 s101 bit-identical to K=1). So on three of four subsets the honest
description of CATA is "equals the baseline". A reviewer will ask the killer question: *is the gate
predicting anything, or is it a subset detector in disguise?* If the accept/reject decision can be predicted
from subset identity as well as from the dispersion statistic, the mechanism claim collapses and what
remains is "aggregation helps on FD001" — a dataset-specific observation, not a module.

Compounding this: "multiple time windows for C-MAPSS RUL" is a **six-year-old line from one group**
(Xia & Xi, SJTU: Computers in Industry 2020 → IEEE Sensors Journal 2026). A PHM-community reviewer will
pattern-match "K windows" to that line before reading the mechanism. The paper must state within the first
page that it uses **one** model with **K shifted inputs**, not K trained models — and that because the
members' targets differ by a *known* constant, the correct operation is offset removal plus a dispersion
test, not weighting.

---

## 3. Minimum experiments and controls needed to differentiate CATA/RGTA

Ordered by discriminating power per unit of compute. Items A–C are cheap and are the minimum; D–F are what
turn a module report into a mechanism paper.

**A. Identity and sanity controls (must exist, trivial cost)**
1. `K=1` parity with the frozen checkpoint's `preds.csv` — already demonstrated (max diff 3e-5); report it
   as the definition of "no change when the module is inert".
2. **Unaligned aggregation**: median of raw `p_t` with no `−Δ` correction. Must be clearly worse than
   aligned. This is the single experiment that proves *age alignment is the mechanism* and not decoration.
3. **Median of aligned vs mean / trimmed mean / recency-weighted** under the same gate — shows the result
   is not an artefact of picking median post hoc.

**B. Gate-mechanism controls (cheap, decisive)**
4. **Random gate at matched accept rate**: for each subset, replace the MAD test with a Bernoulli draw whose
   rate equals RGTA's observed accept rate (multiple draws, report the distribution). If RGTA does not beat
   the matched-rate random gate, the *statistic* is doing nothing and only the *rate* matters.
5. **Gate on the wrong dispersion**: apply the same τ to dispersion of the **unaligned** predictions. Should
   degrade toward the ungated result; separates "dispersion" from "aligned dispersion".
6. **Oracle gate**: accept aggregation exactly where it would have helped (uses labels, analysis only).
   Report **fraction of oracle gain achieved**. This is the cleanest single number for "the gate works".
7. **Always-accept and never-accept** endpoints, to locate RGTA on the accept-rate/gain curve.
8. **Cross-engine placebo**: aggregate predictions from *different* engines at matched K. Any apparent gain
   there falsifies the mechanism.

**C. Baselines that could subsume CATA (these decide whether the module is needed at all)**
9. **Post-hoc monotonic smoothing of the per-engine prediction sequence** — isotonic regression and a
   centred rolling median over the model's own cutoff sequence. Both are label-free, cheaper than K×
   inference, and are the obvious "why not just smooth?" objections. If isotonic matches CATA, CATA is dead.
10. **EMA / Kalman-style smoothing** of the per-engine RUL track (process model = constant RUL decrement).
    This is the classical-prognostics competitor and a reviewer *will* raise it.
11. **A small learned gate/aggregator**: freeze the backbone, fit a 1–2 layer gate on **training engines
    only** with the RUL loss, and compare against the fixed median + τ rule. Establishes whether the
    hand-designed rule is competitive with the obvious learned alternative. (Pilot §4.4 already lists this.)
12. **In-training consistency baseline (one subset is enough)**: retrain the backbone with a
    Li-et-al.-style multi-view variance + ordering loss on FD001, then evaluate with and without CATA on top.
    Two outcomes are both informative: if the retrained model removes the need for CATA, the contribution
    narrows to "post-hoc, no-retraining deployments"; if CATA still adds on top, the offset-removal argument
    is empirically supported. This is the direct answer to Risk 1 and is worth the GPU time.

**D. Hyperparameter honesty (guards Risk 2)**
13. **τ selected without touching validation**: choose τ on a holdout of **training** engines (or by
    cross-fitting), then apply the frozen τ to validation. Report both the "τ from training" and the
    "τ from validation" numbers and the gap. The label-free claim only covers deployment; τ selection must
    not silently consume validation labels.
14. **τ sweep → operating curve** of (gain vs accept rate), per subset. Report the whole curve, not the
    chosen point. If a single τ lands near-optimally on all four subsets, that is a genuine and reportable
    finding; if the optima differ per subset, say so.
15. **K and Δ sensitivity** (K ∈ {3,5,7}, Δ ∈ {1,5,10}) with the gate active — establishes that the result
    is not an artefact of Δ=5, K=5.

**E. Statistics and reporting discipline**
16. ≥5 seeds per subset (FD001 already has 6); **engine-level bootstrap CI** on the paired ΔRMSE
    (CATA − K=1) per subset; report the CI, not only the mean. Note that the pilot's FD001 evidence is
    already a paired, all-seeds-positive result (t ≈ 5.6) — that is the strongest asset and should be
    presented as a paired test, not as a mean difference.
17. **Report the accept rate per subset** alongside every accuracy number. A gain without the accept rate
    is uninterpretable, and the accept rate is what makes the FD004 result meaningful.
18. **Compute cost**: K× forward passes, latency, memory. Frame as a deployment trade-off, not a footnote.

**F. The decisive discriminability test (guards Risk 3)**
19. **Predict "does aggregation help this engine?" from the dispersion statistic alone.** Per engine, label
    `help = 1` if aligned aggregation beats K=1, then compute the AUC of `MAD/|median|` as a predictor.
    Compare against a baseline predictor that knows only the **subset identity**. If the statistic beats
    subset identity, the mechanism claim survives. If not, the gate is a subset detector and the paper must
    be reframed as an empirical characterisation of *when* temporal aggregation helps, with FD001 as the
    positive case — which is still publishable, but as a different paper.
20. **Conditional analysis on the harm set**: among engines where ungated aggregation is worse, what
    fraction does RGTA reject? This is the module's actual job description, and it should be the headline
    table rather than a supplement.

---

## 4. Recommended framing (one paragraph)

What is genuinely new here is narrow and should be stated narrowly: **the observation that cross-window RUL
predictions are estimates of quantities that differ by a known constant, so the correct consistency operator
is exact offset removal followed by a robust dispersion test — not variance suppression, not weighting, and
not projection — and that this turns residual incoherence into a usable, label-free, per-sample reliability
signal.** Everything around that is prior art and should be cited as such: ensembling predictions is
standard; the median as a robust aggregator is standard; `MAD/|median|` is a robust coefficient of variation
with a threshold; selective/gated prediction dates to 2017; MAD-based filtering inside a prognostics ensemble
already exists; temporal coherence across aggregation levels is forecast reconciliation; and the specific
motivation — unstable predictions across overlapping windows — was published in 2026 with an in-training fix.
The defensible contribution is therefore a *characterisation plus a wrapper*: an inference-time,
model-agnostic operator that (a) restores the physically correct invariant that training-time consistency
penalties only approximate, (b) decides per sample whether the invariant is trustworthy, and (c) is
demonstrated to earn its keep mainly by **refusing** to aggregate in regimes where aggregation is harmful
(FD004), while delivering a paired, all-seeds improvement where it is not (FD001). The paper should claim
"safe, label-free post-processing with a diagnostic for when temporal aggregation is admissible", explicitly
**not** a new architecture, **not** a state-of-the-art accuracy result, and **not** a novel statistical test.
If §3-F shows the dispersion statistic predicts benefit better than subset identity, that is the paper's
scientific core; if it does not, the honest move is to publish the negative characterisation and drop the
module claim.
