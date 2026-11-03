# C-MAPSS Remaining Useful Life Prediction

Current milestone: **IDA-PE (TIM IDC track) — Gate 0 PASSED 4/4 (2026-11-03),
status GO_TO_GATE1.** The gated pipeline is live: a strictly causal,
nested-cross-fitted lifetime prior is injected into the frozen PE backbone,
with pre-registered criteria and attribution controls. Branch: `tim-idc-rul`
(fast-forwarded into `main`).

## IDA-PE in one paragraph

**Causal lifetime prior -> individualized degradation clock -> conditioned PE
predictor.** A `HistGradientBoosting` ensemble predicts remaining engine
lifetime `mu_T` (and `sigma_T`) using only information available at each
cutoff (staircase construction: a window ending at cycle `t` consumes the
latest cutoff prediction `c <= t`, never a future interpolation). The
degradation clock `tau = t / mu_T` and the prior itself are injected into the
unchanged PE backbone via **zero-init** layers, so every arm starts exactly
equivalent to M0 and the delta is attributable to the information, not to
re-initialization.

## Gate 0 verdict (frozen criteria, engine-cluster bootstrap, 5000 reps)

| Criterion | Comparison | Delta RMSE | 95% CI | Verdict |
|---|---|---:|---|---|
| G0.1 | M1 (prior concat) vs M0 | **-0.279** | [-0.498, -0.053] | PASS |
| G0.2 | M1_shuf (permuted mu_T) vs M1 | **+0.312** | [+0.114, +0.517] | PASS |
| G0.3 | M1_const (median mu_T) vs M1 | **+0.279** | [+0.072, +0.485] | PASS |
| G0.4 | FD001 guardrail (M1 vs M0) | -0.103 | upper < +0.5 | PASS |

Attribution is closed: both controls have the same +192 parameters as M1, and
both reverse the gain — the benefit comes from **engine-individual lifetime
information**, not from capacity. Full report:
[`docs/tim_idc/gate0_report.md`](docs/tim_idc/gate0_report.md); pre-registered
criteria: [`docs/tim_idc/GATE0_PREREG.md`](docs/tim_idc/GATE0_PREREG.md) and
[`gates.json`](gates.json) (frozen; amendments require a new file, never
in-place edits).

Supporting diagnostic ladder (FD002, same causal features): HGB direct RUL
~31.3 vs prior-implied RUL 17.6 vs PE 12.0 vs M1 12.28 — lifetime *identity*
is far more learnable than trajectory regression, which motivates the clock
design rather than direct RUL transfer.

## Gate pipeline and discipline

- **Gate 0** (passed): does a causal prior add value at all (concat-only),
  with shuffled/constant controls and an FD001 no-harm guardrail.
- **Gate 1** (running): H1 (HGB direct RUL) / H2 (PE/HGB blend, inner-CV
  weights) / H3 (PE + OOF H1 features) vs M1/M0 — decides Module B/C.
- **Gate 2** (pending): uncertainty calibration; safety gate only if
  Spearman(u, |mu_T - T|) > 0.3.
- Rules: criteria frozen before results; a failed gate stops the line and is
  documented, never re-metriced; **engine-level bootstrap only** (window-level
  bootstrap is forbidden in any reported conclusion); the official test set is
  unsealed exactly once; ensemble and single-model results are reported
  separately.

Key code: `src/lifetime_prior/` (nested causal prior), `scripts/run_ida.py`
(five arms), `scripts/summarize_gate0.py` (verdicts),
`scripts/run_h1.py` / `run_h2inner.py` / `run_h3.py` / `run_h2finalize.py`
(Gate 1 arms), `tests/test_prior_causality.py` (leakage tests T1-T4, all
passing), `scripts/diag_prior_rul.py` (diagnostic ladder). Artifacts under
`experiments/ida/`.

## Prior milestones (historical, kept for integrity)

- **R3.1 closed (2026-10-29).** The deployable lifespan-scale feature (`c/T̂`)
  is a **negative result**: after fixing a validation-side leak that had fed
  the true lifespan into the estimator, `est` no longer beats the `cycle` arm
  (+0.08 RMSE, CI [-2.10, +2.27]); the earlier "38% of the oracle gain" is
  withdrawn. See [`docs/director/REVIEW_20261022_R3.md`](docs/director/REVIEW_20261022_R3.md)
  §J. The lsc line's final positive residue: the HGB arm significantly beats
  plain (12.98 vs 13.96, recovering 28.7% of the oracle gap) — this motivated
  the IDA-PE prior estimator choice. EAC/FiLM: NO-GO; Modern Backbone: failed;
  the PE backbone is retained.
- **P-line (P7-P10) is not citable** — withdrawn at book level on 2026-10-29;
  nothing from the P-line enters the manuscript (`paper/E5_CONSISTENCY.md` §4).
  Kept for the audit trail: [`P10_FINAL_VICTORY_REPORT.md`](P10_FINAL_VICTORY_REPORT.md),
  [`P9_DIRECTOR_REPORT.md`](P9_DIRECTOR_REPORT.md).
- Stage-1 literature/protocol table: [`docs/LITERATURE_BENCHMARK.md`](docs/LITERATURE_BENCHMARK.md).

## Baseline verification (2026)

- **CET-MSA** (Springer JIM 2026): official code verified
  (`suosimiaoqianli/Noise-Robust-RUL-CET-MSA`), cloned for reproduction.
- **BootBoGs-TS** (2026): official code verified (`Gyouness/BootBoGs-TS`).
- AIDGN / LVDACNN / MSMT-FDA: no official code available.

## Layout

~~~text
gates.json                     frozen gate criteria (pre-registered)
src/lifetime_prior/            nested causal lifetime prior (staircase)
src/dataset.py                 C-MAPSS, clip125, windows, metrics, FD002 scaler
src/model.py                   PE backbone (retained; EAC/FiLM/modern NO-GO)
scripts/run_ida.py             five-arm runner (M0/M1/M1p/shuffled/const)
scripts/summarize_gate0.py     engine-cluster bootstrap verdicts
scripts/run_h1.py              HGB direct-RUL arm (Gate 1)
scripts/run_h2inner.py         PE/HGB blend, inner-CV weight selection
scripts/run_h3.py              PE + OOF H1 features (zero-init)
scripts/diag_prior_rul.py      prior-implied-RUL diagnostic ladder
tests/test_prior_causality.py  leakage tests T1-T4
docs/tim_idc/                  prereg, gate0 report, paper skeleton
experiments/ida/               metrics/predictions (read-only aggregation)
~~~

## Quick start

~~~bash
python -m unittest tests.test_prior_causality -v   # leakage tests
python scripts/run_ida.py --subset FD002 --arm M1 --fold 0 --seed 42
python scripts/summarize_gate0.py                  # reproduce Gate 0 verdicts
~~~

## Integrity rule

No ablation number is hand-written or selected to fit an expected story. All
reported numbers come from metrics / predictions files generated by frozen
runners; aggregation scripts read those files, never human transcription.
Gate criteria are frozen before results exist and are not modified after.
Negative results are reported as negative.
