# Independent source and DEVELOPMENT_LOEO audit

Date: 2026-11-13. Status: **PARTIAL — source preservation and a development-only collector are implemented; completion of four-paper unified training is not established.**

## Verified primary sources and exact author snapshots

| Paper | Verified DOI | Author source and fixed commit |
|---|---|---|
| A Benchmark on Uncertainty Quantification for Deep Learning Prognostics, RESS253 (2025) 110513 | [10.1016/j.ress.2024.110513](https://www.sciencedirect.com/science/article/pii/S0951832024005854) | [lbasora/bayesrul](https://github.com/lbasora/bayesrul/tree/2943f4d85a1bb71910c100c956b32c3f6b595ceb) — 2943f4d85a1bb71910c100c956b32c3f6b595ceb |
| Aging-Informed Dynamic Graph Network for Aero-Engine RUL Prediction, IEEE TII (2026) | [10.1109/TII.2026.3694202](https://doi.org/10.1109/TII.2026.3694202) | [TCCofWANG/AIDGN-RUL](https://github.com/TCCofWANG/AIDGN-RUL/tree/d1ac173b13659af1854cf500a9b1cba8367353fe) — d1ac173b13659af1854cf500a9b1cba8367353fe |
| Multi-objective Optimization of Extreme Learning Machine for Remaining Useful Life Prediction, EvoApplications2022, 191–206 | [10.1007/978-3-031-02462-7_13](https://link.springer.com/chapter/10.1007/978-3-031-02462-7_13) | [mohyunho/MOO_ELM](https://github.com/mohyunho/MOO_ELM/tree/b19b2bf47bdfc3b7d4ae2a540b605caf4fd087e3) — b19b2bf47bdfc3b7d4ae2a540b605caf4fd087e3 |
| Interpretable Prognostics with Concept Bottleneck Models, Information Fusion124 (2025) 103427 | [10.1016/j.inffus.2025.103427](https://doi.org/10.1016/j.inffus.2025.103427) | [EPFL-IMOS/concept-prognostics](https://github.com/EPFL-IMOS/concept-prognostics/tree/25ec477c834e81440a288e161ec1f4461737ee49) — 25ec477c834e81440a288e161ec1f4461737ee49 |

NASA dataset provenance: [official NTRS record 20205001125](https://ntrs.nasa.gov/citations/20205001125), [Data6(1):5, DOI10.3390/data6010005](https://www.mdpi.com/2306-5729/6/1/5). N-CMAPSS contains simulated run-to-failure trajectories driven by real flight operating profiles; it is not a fleet of measured field failures. RUL is in flight cycles, while 1Hz describes within-flight sampling.

The prior source audit actually preserved 309 original files: BayesRUL120, AIDGN90, MOO_ELM51 and Concept48. All passed immutable-commit Git blob SHA-1, SHA-256 and local read-only checks. Every upstream blob is listed as included or excluded in its SOURCE_MANIFEST.json. AIDGN1199 and MOO37 generated/data/checkpoint/figure/IDE blobs were excluded. The preserved BayesRUL notebook was not stripped or executed. Evidence: paper_b/ncmapss_response/upstream/PRESERVATION_ACCEPTANCE.json and the four SOURCE_MANIFEST.json files.

BayesRUL and AIDGN have no repository-wide license file in these snapshots. AIDGN N_CMAPSS_train.py has a file-specific MIT statement. Concept cem/setup.py contains an MIT package classifier, which does not establish a blanket license for every repository file. These258 source files were confirmed Git-ignored; only audit metadata is eligible for ordinary tracking. MOO_ELM has a root Apache-2.0 license, with separate GPL/LGPL notices in vendored hypervolume files. All notices were preserved. This is evidence recording, not redistribution approval.

## Reproduction boundaries

- BayesRUL has complete Inception-HNN source, dual softplus output and Gaussian NLL, plus training/prediction/configuration code. Its published setup jointly trains DS01–05, randomly splits windows and shapes labels using hs. DS02-only engine-held-out training is a separately named adaptation.
- AIDGN includes complete aging-conditioned graph code and NC preprocessing. Original NC inputs include T40/P30 virtual variables; the original cap is derived after combining development and test labels. Uniform observed-input and train-only preprocessing must be disclosed separately. The preserved package initializer imports missing Model.AIGCN; bypassing unrelated imports in an external adapter must not rewrite the original model.
- MOO_ELM includes complete NSGA-II search, HPELM and supporting source. The original20 inputs contain T40/P30, and validation is by random samples. Retain the search/objective and report its actual budget; an arbitrary small fixed ELM is not this method.
- Concept models use concept supervision from simulator health parameters, and hs-defined label plateaus. Published experiments cover DS01/04/05/07 and DS01/03/07, not DS02. A plain-CNN DS02 adaptation and privileged-concept conditions must be distinguished.

**No uniform-protocol training, checkpoint, RMSE or NASA result for these four papers is certified here. No public-paper metric is promoted to a matched result.** Source preservation and mathematical model availability are different evidence levels.

## Novelty boundary for the response candidate

[R2-D2, ICLR2019, arXiv1805.08136](https://arxiv.org/abs/1805.08136) already differentiates through a ridge solver. [Function Encoders, ICML2025, arXiv2501.18373](https://proceedings.mlr.press/v267/ingebrand25a.html) already learns shared neural bases with function-specific least-squares coefficients, discusses ridge-induced basis growth and input-distribution-dependent inner products. [Basis-to-basis operator learning, DOI10.1016/j.cma.2024.117646](https://doi.org/10.1016/j.cma.2024.117646) already maps learned function coefficients downstream.

[Operation-profile alignment, DOI10.1016/j.ress.2023.109718](https://arxiv.org/html/2302.01704), [physics/deep fusion, DOI10.1016/j.ress.2021.107961](https://doi.org/10.1016/j.ress.2021.107961), and [neural-guided particle filtering, DOI10.1016/j.ymssp.2026.115026](https://doi.org/10.1016/j.ymssp.2026.115026) establish prior work on operating conditions, response/health representations and estimation–prediction separation.

The current candidate's possible contribution is restricted to a **testable common-support, shared-flight response contrast and trace-normalized directional excitation weighting**. A differentiable inverse, function encoder, parallel sum or Transformer alone is not a new invention. The latent coordinates are not physical health parameters; excitation geometry is not calibrated uncertainty; normalization deliberately removes total excitation magnitude. A PASS requires comparison with FE, unrestricted delta and supported-unweighted contrast plus actual observed rank/weight effects. It does not prove causality, novelty priority, confidence calibration or a universally correct physical model.

## Collector contract and frozen interpretations

CLI: `python -B -m paper_b.ncmapss_response.summarize --cache <development-cache> --runs <sealed-runs> --output <new-output>`.

The collector imports neither run.py, prepare.py nor models.py. It decodes only raw/y/unit/cycle from the four explicitly named development NPZs. No HDF5, TEST arrays, model forward or optimizer step is used. If PyTorch is available, checkpoints are loaded with map_location=cpu and weights_only=True solely to verify tensor-state digests, finiteness/FP32, epoch counts and optimizer-step evidence. Unavailable checkpoint verification blocks full PASS.

Each SUCCESS must inventory every cell file; every recorded SHA, config self-hash, execution identity, cache/source identity, fold scaler, prediction row, raw/capped metric, drift and training-budget field is independently checked. Missing cells produce PARTIAL. Modified or inconsistent sealed cells produce INVALID. Existing output directories are refused. Output JSON/CSV/Markdown are sealed with their own COLLECTION_COMPLETE.json file inventory.

For each neural seed, the primary metric is the arithmetic mean of the six held-out-engine RMSEs, followed by the mean and sample SD across three seeds. Flight-pooled RMSE and NASA sum/mean are separate. The analytic age prior has24 deterministic folds: estimate each of five training engines' constant rawY+cycle lifetime, average the five lifetimes equally, predict max(0,mean_lifetime-current_cycle) for the held-out engine. No held-out lifetime is used for fitting. Each fold saves prediction.csv, config.json and checkpoint.json explicitly labelled analytic/no-training.

Paired deltas are retained for each engine and seed. Bootstrap first averages seeds within engine, then resamples the six engines10,000 times. Its percentile95 intervals are descriptive: LOEO training sets overlap and seed repeats are not additional independent engines. No p-values, significance labels or post-hoc subset selection are produced.

Before any result freeze, the parent accepted the following conservative interpretations:

| Gate | Rule |
|---|---|
| Primary | Same at least2 subsets jointly gain>=2% versus raw and FE; no subset/control worse>2%. |
| Age necessity | Strictly beat BOTH age MLP and analytic age prior on every subset. |
| Sensor necessity | Strictly beat W-only on every subset. W-only retains W4, coverage, age and the same70-flight head; Xs14 is zeroed and its auxiliary loss is off. It is a necessary reference, not a matched H-attribution arm. |
| Attribution | Same at least2 subsets jointly gain>=2% versus delta and supported; no subset/control worse>2%. |
| Mechanism | Same at least2 subsets jointly satisfy all FE/delta/supported controls: density drift reduction>=25%, control drift>=0.05cycles, positive clean gain, unpaired relative gain<=half clean relative gain, and common perturbed accuracy no>2% worse. |
| Diagnostic evidence | Actual rank/support and weighting-effect summaries are retained; missing diagnostics block full PASS. Exact zero support/weight effect cannot identify a spectral-weighting mechanism. |
| Incompleteness | Missing runs/caches, unavailable CPU checkpoint evidence or absent mechanism observations never count as a complete PASS. |

The controller stores actual tensor shape/mean/minimum/maximum summaries. The collector preserves and aggregates these; it cannot fabricate rank histograms, per-flight conditional effects or spectral eigenvalue distributions absent from those artifacts. All seven modes, four subsets, six folds and three seeds are retained:504 neural cells plus24 analytic folds. W-only was added before any actual training result. Full collection does not imply a hypothesis PASS.

Static checkpoint structure checks retain the full registered state schema and shapes, not just a positive reported parameter count:84075 total parameters for each response/head arm (including structurally inactive components) and4353 for age MLP. No model is constructed or forwarded to perform these checks.

W-only's encoder report inherits the raw branch's auxiliary-inclusive scope, whereas W-only's auxiliary loss is disabled. The collector preserves that report and separately accounts for4138 task-inactive encoder parameters:896 first-layer sensor-input weights,2730 response/background/basis/auxiliary-head parameters, and512 zero-latent embedding weights. W-only task-effective encoder/model counts are8704/79937, versus total12842/84075. These are structural counts, not a correction for sample-specific fallback or saturation, and do not turn W-only into a matched attribution arm.

## Implementation verification completed

AST parsing and CLI help passed. Purely synthetic temporary contracts passed for raw/NASA definitions, engine-before-seed aggregation, pooled-versus-engine distinction, six-engine bootstrap, same-subset gates, near-zero nonidentifiability, missing-evidence lock, equal-training-engine analytic prior, invariance of fitted prior to held-out labels, all24 analytic folds and existing-output refusal. A separate explicitly synthetic CPU-loader fixture passed cell identity/scaler/CSV/metric/diagnostic checks, full state-schema checks, and rejection of tampered seals, wrong scalers, wrong epoch counts and missing core tensors. No synthetic fixture is an experiment result or retained in formal experiment directories.

The local interpreter has NumPy2.3.5 but no PyTorch. Real torch.load deserialization and real trained-checkpoint acceptance were therefore not exercised locally. The production collector explicitly reports unavailable CPU checkpoint evidence as PARTIAL; it must perform real CPU checkpoint verification on the training environment before complete acceptance. No CUDA, model forward, optimization or TEST access was used in these checks.

The recorded successful-cell training-time sum excludes failed/incomplete attempts, evaluation and preparation overhead. It is not falsely presented as verified total GPU-process consumption.

