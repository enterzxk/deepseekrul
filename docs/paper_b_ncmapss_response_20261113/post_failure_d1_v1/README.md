# D1 evidence: exploratory training-side residual diagnostic

The fixed protocol was registered in `REGISTRATION.json` before this execution. Earlier common-route outcomes and the earlier descriptive condition audit were already known. This is post-failure exploration, not a prospectively independent confirmation of a new method. The closed common route and frozen experiments are unchanged; no new RUL model is authorized by this diagnostic.

Only `raw`, `unit`, and `cycle` are decoded from the four development NPZ caches in the existing ignored `paper_b/ncmapss_response/_data` directory. No raw NPZ is copied into this evidence directory. The input byte hashes and allowed decoded keys are retained in each result. Cache provenance mentions the earlier preparation of labels; this diagnostic never requests or decodes those label arrays.

For each subset and each excluded outer engine, each of the remaining five engines is predicted after a fresh fixed ridge fit on the other four. Both physical scalers and dynamic feature scalers use those four engines only. All120 crossfits are executed, with no hyperparameter search, optimizer, GPU, official TEST, or use of the excluded outer engine to choose the forms. The repeated overlapping fits are dependent and must not be counted as120 independent samples.

The correct sensor column order is T24, T30, T48, T50, P15, P2, P21, P24, Ps30, P40, P50, Nf, Nc, Wf. It intentionally differs from the mislabeled ordering in the older descriptive audit.

The dynamic input has37 original columns including constant1. After training-four standardization the constant is zero; a separate unpenalized intercept is prepended. Thus38 coefficients are stored, representing36 nonconstant features plus the intercept. Differences are between adjacent samples on a512-point normalized flight-position grid, never per-second rates.

Empirical residual scale is the training-four fitted static residual standard deviation, not noise, predictive uncertainty, or a health estimate. A fixed positive per-sensor division preserves within-engine Pearson age correlation exactly; that identity is explicitly checked. Total dynamic prediction gain cannot isolate differences from quadratic recalibration, the intercept, or normalized position because all are in the one frozen predictor.

Run with the local Python interpreter, without invoking a DSH agent:

```powershell
& 'C:/Users/20562/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/python.exe' -B `
  'C:/Users/20562/.codex/worktrees/paper-b-ncmapss-response-20261113/paper_b/ncmapss_response/diagnose_residuals.py' `
  --cache 'C:/Users/20562/.codex/worktrees/paper-b-ncmapss-response-20261113/paper_b/ncmapss_response/_data' `
  --output 'C:/Users/20562/.codex/worktrees/paper-b-ncmapss-response-20261113/docs/paper_b_ncmapss_response_20261113/post_failure_d1_v1/results'
```

The output directory is exclusive-create; the command refuses to overwrite a completed or partial attempt. `run.stdout.txt`, `run.stderr.txt`, and `EXIT.json` retain the actual run evidence. `SOFTWARE_CONTRACTS.json` contains synthetic software checks and is not an experiment result. Real fitted coefficients, scalers, per-engine/sensor values, and flight-mean arrays are under `results/`, with an exclusive completion seal only if all120 fixed fits finish.

The actual CPU run completed120/120 fits with exit0. The fixed scale-suppression gate was met; the fixed dynamic-omission gate was not met because only DS05 reached its threshold. The exact values and interpretation remain in `results/RESIDUAL_DIAGNOSTICS.json` and its companion report. No thresholds or diagnostic source were changed after these observations.

An independent read-only verifier reconstructs all120 held-inner predictions from the persisted coefficients, recomputes training-four physical scalers, all1680 sensor values and gates, checks every sealed file hash, and checks the unchanged frozen response execution hash:

```powershell
& 'C:/Users/20562/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/python.exe' -B `
  'C:/Users/20562/.codex/worktrees/paper-b-ncmapss-response-20261113/docs/paper_b_ncmapss_response_20261113/post_failure_d1_v1/verify_evidence.py'
```

The verifier refuses to overwrite `ACCEPTANCE.json`; its stdout, stderr and exit record are retained separately. Its first attempt exposed an evaluation-order assumption: the fixed diagnostic averages original FP32 raw observations before the FP64 Pearson calculation, while saved raw flight-mean arrays use FP64 averaging. `VERIFIER_ATTEMPT1.json` preserves that check failure. The corrected verifier reproduces the preserved metric order and separately quantifies FP64-mean sensitivity; it does not change the diagnostic, coefficients, or outcomes. Residual and dynamic quantities already use FP64.
