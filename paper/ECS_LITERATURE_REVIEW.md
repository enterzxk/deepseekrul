# ECS-RUL: literature review and novelty audit

Date: 2026-11-01. Scope: five parallel searches (architecture, partial observation,
survival/Bayesian, condition normalisation, repository mechanism). Verification tags:
[V] = publisher page or DOI resolved and metadata matched; [P] = metadata from index
page; [U] = unverified. Nothing here is invented; [U] items must be checked before
citation.

## 1. The question this review has to answer

> Has anyone already built a mechanism that (a) treats *evidence credibility* as an
> explicit, estimable quantity — driven by observation coverage and cross-sensor
> consistency — and (b) uses it to modulate the relative weight of a lifetime/age
> prior versus the sensor likelihood, producing a calibrated RUL distribution that
> degrades gracefully to the prior when evidence is thin?

If yes, the proposed direction is a re-naming exercise and must be dropped.

## 2. Nearest neighbours, and exactly how they differ

### 2.1 Statistical machinery that already exists (and must be cited)

| mechanism | canonical reference | what it does | what it does not do |
|---|---|---|---|
| **Power prior** — likelihood raised to α ∈ [0,1] | Ibrahim & Chen, *Statist. Sci.* 15(1):46–60, 2000 [V] | the rigorous form of "weight how much historical data counts"; α=0 discards history, α=1 pools it | α is static, one-shot, no sensors, no time evolution |
| **Commensurate prior** — borrowing strength driven by *conflict* | Hobbs, Sargent & Carlin, *Bayesian Analysis* 7(3):639–674, 2012 [V] | τ estimated from prior-data conflict; more disagreement → less borrowing | same: no sensor dimension, no online recursion |
| **Precision-weighted fusion** — w ∝ 1/σ² | Sun & Deng, *Automatica* 40(6):1017–1023, 2004 [V] | optimal multi-sensor Kalman weighting | weights depend only on noise variance, not on consistency or coverage |
| **Survival measurements as a likelihood constraint** | Si et al., *IEEE TIE* 67(4):3202–3215, 2020 [V] | "device survived to t" written into the likelihood, removing impossible paths | hard constraint, single dimension (length), no consistency term |
| **Evidence-theoretic discounting** | Baraldi et al., *ESWA* 83:115–126, 2017 [V] | each reference trajectory is an item of evidence, discounted by similarity before Dempster combination | discount is a monotone function of a *mixed* similarity; cannot separate coverage from shape from sensor mismatch |
| **Sample-entropy evidence reliability** | Xu Xiaobin et al., *RESS* 250:110250, 2024 [V] | online sample entropy of activated evidence sets its reliability weight, then evidential-reasoning fusion | reliability tracks signal complexity, not observation coverage; weight sits at the fusion layer, not in the distribution |

**Consequence:** any claim of the form "weight the prior by how much you trust the
data" is, mathematically, a power prior. The paper must say so in the first paragraph
of Related Work. The contribution, if any, is in *how the weight is produced* and
*what it is attached to*, not in the existence of a weight.

### 2.2 RUL-specific work that fuses age and sensors

| work | venue | mechanism | gap relative to the proposal |
|---|---|---|---|
| Srinivasan et al., "Integrating Survival-Based Aging Models with Data-Driven RUL Prognostics" | PHME 2026, DOI 10.36001/phme.2026.v9i1.4939 [V] | Weibull wear model and 1D-CNN sensor model each produce a failure-time distribution; combined as P(F\|W,S) ∝ P(F\|W)P(F\|S)/P(F), weights adaptive to each component's variance; N-CMAPSS | reliability = inverse predictive variance only; symmetric and passive; **no head-to-head against fixed weights**; conference paper |
| Zhou et al., UCD-OCMN | RESS 2026/2027, 10.1016/j.ress.2026.113138 [V] | physics-constrained Bayesian dynamic graph + Nyström Kriging + learnable Kalman fusion balanced by estimated reliability | fuses two *prediction branches*, not prior vs likelihood; reliability is implicit in the Kalman gain, not an interpretable scale |
| Chen et al., Bayes-LDDP | IEEE TR 73(2):990–1004, 2024 [V] | weak informative prior from similar systems + field inspection likelihood, MCMC | static prior, equally-trusted likelihood, no credibility term |
| Lin et al. | IEEE TII 21(3), 2025 [V] | sparse variational Bayes auto-weights a library of degradation models | weights candidate *models*, not evidence sources |
| Wang et al. | MSSP 236:112970, 2025 [V] | empirical Bayes updates the prior itself via EM | prior is learned, but fusion weight is not credibility-driven |
| Liu et al. | *Microelectronics Reliability* 55, 2015 [V] | information entropy selects sensor subset offline | selection only, not online weighting of RUL |
| Ta et al. | RESS 231:109033, 2023 [V] | information contribution rate weights multi-sensor HI fusion | weights stop at the HI layer, never reach the Bayesian update |
| Domingues et al. | JAIMLNN 6(2):106–119, 2026 [P] | shows marginal 90% coverage collapses to 37–50% near failure under regime shift, with 95% of miscoverage on the dangerous (over-prediction) side | diagnostic only; low-tier venue, use as corroboration not as authority |

### 2.3 Time-shortcut evidence (this is the empirical motivation, and it is partly taken)

| work | finding | status for us |
|---|---|---|
| Zhang et al., arXiv:2607.24809, 2026 [V] | deliberately avoids linear flight-time features because they create a "linear flight-time-to-RUL shortcut"; uses bounded periodic encoding plus per-unit period recalibration | **the design principle is already stated.** Cannot claim "we discovered time shortcuts" |
| Karthick, *WEVJ* 15(2):60, 2024 [V] | RUL–cycle-index correlation is −1; removes cycle index from the feature set | engineering awareness exists |
| Imbert, ICTAI 2025 / arXiv:2605.02507 [V] | window padding injects repeated noise and weakens temporal convolution; proposes whole-trajectory training | qualitative; no controlled measurement of padding's effect |
| Gupta, engrXiv 10.31224/8145, 2026 [P] | 1,080-cell audit: condition normalisation helps in every cell (−4.39 / −7.40), but the *fit protocol* alone moves RMSE by 1.12 | protocol sensitivity is being audited by others; do not re-litigate |
| Xiao et al., CDIG, RESS 253:110534, 2025 [P] | domain generalisation under unseen conditions; keeps conditions and time as explicit inputs | confirms "you cannot normalise everything away" |

## 3. Repository mechanism audit (evidence for the problem statement)

From `src/candidates/model.py` and `scripts/run_candidate.py`:

1. The age term in arm A is an additive bias `b + c·w`, **magnitude ∝ c**. Training
   windows only ever see c ∈ [0.2, 1.0] (`run_candidate.py:358`: windows start at
   `end=49`, c = cycle/250). The 24 short-trajectory FD002 engines have
   c ∈ [0.084, 0.196] — **all outside the support**.
2. Arm PE is a bounded 4-period encoding: every short c aliases onto a code value seen
   during training (e.g. L=21 → 21+50=71 → inside the support). It cannot extrapolate.
3. Back-solving the published RMSEs (E21 retained 235 vs E22a full 259) gives
   short-trajectory RMSE: **A 39.9, BE 26.8, PE 12.0** — and the constant-125 oracle
   scores 12.41. PE is behaving like "always predict the cap" on those engines.
4. Inside the support, A and PE are indistinguishable on the cap segment
   (mean prediction 116.02 vs 115.87). **The 4.7 RMSE gap is a boundary phenomenon,
   not a trend.**
5. The only "reliability" code in the repository receives 6 evidence features, of
   which e1, e2, e3 are **constants** (1.0, 1.0, 0.9048) across the entire dataset, and
   `ref_count` counts reference frames (≤5), not the observation coverage of the
   prediction window. The mechanism is structurally blind to "this engine was observed
   for 21 cycles".
6. First-frame padding makes 58% of a 50-cycle window a copy of frame 1 for L=21 — a
   pattern never seen in training.

## 4. Novelty verdict

| claim | verdict |
|---|---|
| "Time/age is legal information but over-relied upon" | **NOT novel** (Zhang 2026, Karthick 2024, Imbert 2025, Gupta 2026). Cite, do not claim |
| "Sensor evidence should be discounted when unreliable" | **NOT novel as a principle** (power prior, commensurate prior, Baraldi, Xu). Cite the lineage |
| "RUL should degrade to an age prior when evidence is thin" | **novel as an explicit design rule for a deep RUL model**, closest to Srinivasan 2026 (variance-weighted, no coverage term) |
| "Coverage length and cross-sensor consistency as two separately interpretable credibility factors" | **gap**: no work found that decouples them from similarity/entropy/variance |
| "Credibility estimated by cross-fitted predictive information gain" | **gap**: the estimation method itself is not in the RUL literature surveyed |
| "Calibrated by bin-wise conditional coverage under regime shift, with the prior-only and sensor-only endpoints as mandatory ablations" | **gap**: conditional coverage under shift is a known problem (Domingues 2026) but nobody has used it as the acceptance criterion for a fusion mechanism |

**Positioning sentence the paper can defend:**

> Existing work establishes that age is over-used (diagnosis), and that age and sensor
> models can be combined by inverse-variance weighting (fusion). It has not defined an
> evidence-credibility variable that is estimated from coverage and consistency,
> attached to a power-prior-style likelihood tempering, and validated by bin-wise
> conditional coverage with prior-only and sensor-only endpoints. That is the gap this
> paper fills — a gap in *how the weight is produced and validated*, explicitly built
> on the power-prior lineage rather than presented as new mathematics.

## 5. Risks and mandatory responses

| risk | response (must appear as an experiment, not a paragraph) |
|---|---|
| "This is a time-varying power prior" | **yes, structurally**; show the α=0 and α=1 endpoints and that the method interpolates between them on the power-prior family (ESS reported) |
| "Srinivasan 2026 already did it" | head-to-head: variance-weighted fusion vs credibility-weighted fusion, with the coverage/consistency ablations that variance weighting cannot express |
| "Gains come from extra parameters" | parameter-matched gate (same width, constant input) and identical training budget |
| "Gains come from padding artefacts" | the three controls C1/C2/C3 from the mechanism audit (coverage-coordinate fix, padding-shape fix, training-time padding augmentation) |
| "Conditional coverage is cherry-picked" | pre-register the bins (RUL quartiles × observation-length terciles) and report all of them |

## 6. Full reference list (36 entries, deduplicated across the five searches)

Architecture / UQ: EviAdapt (TIM 2025, 10.1109/TIM.2025.3551977) [V]; CRULP conformal
(TIM 2024, 10.1109/TIM.2024.3522678) [V]; GPVAE uncertainty decomposition (TIM 2024,
10.1109/TIM.2024.3485392) [V]; semi-supervised UQ (TIM 2023, 10.1109/TIM.2023.3334339)
[V]; neural-ODE + feature-time attention (TII 2024, 10.1109/TII.2023.3333933) [V];
CLformer + generalised Wiener (TII 2024, 10.1109/TII.2024.3450003) [V]; sparse
variational Bayes model library (TII 2024, 10.1109/TII.2024.3495776) [V]; KEST-GCN
(RESS 2025, 10.1016/j.ress.2025.110928) [V]; UCD-OCMN (RESS 2026,
10.1016/j.ress.2026.113138) [V]; empirical-Bayes exponential degradation (MSSP 2025,
10.1016/j.ymssp.2025.112970) [V]; adaptive-trajectory stochastic degradation (MSSP 2025,
10.1016/j.ymssp.2024.112171) [V]; MTCN with dual UQ (EAAI 2025,
10.1016/j.engappai.2025.111303) [V]; PMBCT Bayesian Transformer (IEEE TR 2025,
10.1109/TR.2024.3427797) [V]; multi-view graph + joint interval (JMS 2025,
10.1016/j.jmsy.2025.05.009) [V].

Partial observation / early life: DR-MC (J. Power Sources 2024,
10.1016/j.jpowsour.2023.233760) [V]; pattern-RVM Bayesian ensemble (ISA Trans 2025,
10.1016/j.isatra.2024.11.023) [V]; WSF-ML-R untimed short trajectories (RESS 2024,
10.1016/j.ress.2024.110517) [V]; semi-observable self-correcting RUL (RESS 2026,
10.1016/j.ress.2026.112909) [V]; RobustUQ imputation + conformal (RESS 2025,
10.1016/j.ress.2025.111177) [V]; PF vs Gibbs under sensor degradation (RESS 2026,
10.1016/j.ress.2026.112912) [V]; MAML-GRU early RUL (J. Power Sources 2025,
10.1016/j.jpowsour.2025.238569) [V]; single-trajectory augmentation + KDE (EiN 2024,
10.17531/ein/175585) [V]; Daigle & Goebel observability (IEEE Aerospace 2010,
10.1109/AERO.2010.5446822) [V]; entropy-based sensor selection (Microelectron. Reliab.
2015, 10.1016/j.microrel.2015.06.076) [V]; evidential-reasoning reliability weights
(Xu, RESS 2024, 10.1016/j.ress.2024.110250) [V]; fuzzy-similarity belief discounting
(Baraldi, ESWA 2017, 10.1016/j.eswa.2017.04.035) [V]; credibility of RUL prediction
(Wu, JRSE 2026, 10.1088/3050-2454/ae9ec9) [V]; actuarial conditional survival
(Bechhoefer, PHM 2015, 10.36001/phmconf.2015.v7i1.2758) [V].

Survival / Bayesian / calibration: Si et al. review (EJOR 2011,
10.1016/j.ejor.2010.11.018) [V]; survival models for PdM review (Sensors 2026,
10.3390/s26061915) [V]; DeepSurv (BMC Med Res Methodol 2018, 10.1186/s12874-018-0482-1)
[V]; DeepHit (AAAI 2018, 10.1609/aaai.v32i1.11842) [V]; SurvLoss censored regression
(PHME 2024, 10.36001/phme.2024.v8i1.4052) [V]; Wiener-process RUL review (EJOR 2018,
10.1016/j.ejor.2018.02.033) [V]; recursive-filter Wiener RUL (MSSP 2013,
10.1016/j.ymssp.2012.08.016) [V]; survival measurements (IEEE TIE 2020,
10.1109/TIE.2019.2908617) [V]; Prognostics 101 particle filter (RESS 2013,
10.1016/j.ress.2013.02.019) [V]; residual life without prior knowledge (IEEE TR 2009,
10.1109/TR.2008.2011659) [V]; power priors (Statist. Sci. 2000, 10.1214/ss/1009212673)
[V]; commensurate priors (Bayesian Anal. 2012, 10.1214/12-BA722) [V]; precision-weighted
fusion (Automatica 2004, 10.1016/j.automatica.2004.01.014) [V]; CRPS/coverage metrics
(de Pater & Mitici, PHME 2022, 10.36001/phme.2022.v7i1.3320) [V]; prognostic metrics
(Saxena, IJPHM 2010, 10.36001/ijphm.2010.v1i1.1336) [V]; UQ + isotonic + conformal
(RESS 2026, 10.1016/j.ress.2026.112763) [V]; conformal RUL intervals (IJPHM 2023,
10.36001/ijphm.2023.v14i2.3417) [V].

Condition / shortcut / padding: Gupta protocol audit (engrXiv 2026, 10.31224/8145)
[P]; functional-data condition normalisation (ICPHM 2019, arXiv:1904.06442) [V];
K-means condition normalisation (Sensors 2023, 10.3390/s23125669) [P]; physics-guided
domain adaptation with bounded periodic time encoding (arXiv:2607.24809, 2026) [V];
RioT right-reason loss (ECML PKDD 2025, arXiv:2402.12921) [V]; battery leakage from
unblinded training (ChemRxiv 2023, 10.26434/chemrxiv-2023-0zhhd) [P]; cycle index
removal (WEVJ 2024, 10.3390/wevj15020060) [V]; CDIG (RESS 2025,
10.1016/j.ress.2024.110534) [P]; CITDG (MSSP 2025, 10.1016/j.ymssp.2025.113527) [P];
multi-domain orthogonal DG (MSSP 2025, 10.1016/j.ymssp.2024.111924) [V];
whole-trajectory TCN (ICTAI 2025, 10.1109/ICTAI66417.2025.00160) [V]; scarce-series
indirect supervision (arXiv:2504.09206) [V]; Srinivasan age×sensor fusion (PHME 2026,
10.36001/phme.2026.v9i1.4939) [V].

## 7. What this changes about the plan

1. **The framing must be "fusion done right", not "we found a shortcut".** The shortcut
   literature is populated; the fusion literature is populated; the intersection is not.
2. **The mechanism's name should acknowledge the power prior.** Proposed:
   *credibility-tempered survival fusion* — a power prior whose exponent is a
   data-estimated credibility, attached to a conditional survival model.
3. **Two gaps survive scrutiny** and are worth the experiments: (i) decoupling coverage
   from consistency in the credibility estimate; (ii) validating the mechanism by
   *conditional* coverage under regime shift, with the α=0/α=1 endpoints as ablations.
4. **Nothing in the literature review supports claiming a new estimator class.** The
   GO/NO-GO will rest on the experiments in the method design, not on this document.
