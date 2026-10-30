# E3 — Amendment 1: the non-inferiority verdict is withdrawn (bootstrap not converged)

Timestamp: 2026-10-29 23:20 UTC+8. Filed **before** any manuscript text used the
verdict. The pre-registration text is left untouched; this file and §12 of the
pre-registration record the change.

## 1. What was claimed

`paper/E3_RESULT.md` (first version) reported a one-sided 97.5% upper bound of
**+0.291** for ΔRMSE(PE − A) from **4,000** bootstrap resamples (RNG 20261029),
and declared the pre-registered non-inferiority rule satisfied (`< +0.30`).

## 2. What the re-analysis shows

Same estimator (Δ = RMSE(PE) − RMSE(A), paired by `unit_id`, averaged over the
three seeds, engines resampled within seed, 97.5th percentile):

| design | 97.5% upper bound |
|---|---|
| 4,000 reps, RNG 20261029 (the reported number) | **+0.291** |
| 100,000 reps × RNG 9000 | +0.309 |
| 100,000 reps × RNG 9001 | +0.321 |
| 100,000 reps × RNG 9002 | +0.324 |
| 100,000 reps × RNG 9003 | +0.316 |
| 100,000 reps × RNG 9004 | +0.309 |
| **5 × 100k, mean** | **+0.317** |
| 4,000 reps × 20 RNG seeds | median **+0.315**; **70% ≥ 0.30** |

Converged bound: **+0.32 > +0.30**. The 4,000-resample value was a low draw of a
distribution whose median already exceeds the margin.

Artifact: `results/dsh/e3/bootstrap_stability.json` (+ `_big.csv`, `_small.csv`);
script `scripts/dsh_e3_bootstrap_stability.py` (read-only over the predictions).

## 3. Decision (pre-registered rule, applied as written)

**E3 = FAIL: non-inferiority not established.** The margin is not moved, the
point estimate is not removed, and the 0.291 run is kept on record as a
Monte-Carlo draw.

What does **not** change:

* point estimate ΔRMSE = **−0.897** (per-seed −0.73 / −0.87 / −1.13), 3/3 seeds
  favour PE;
* NASA (2.729 vs 3.537) and late (0.487 vs 0.538) still favour PE;
* BE − A = **+2.65 [+1.32, +4.05]** (a separate contrast, unaffected).

What changes: the paper may no longer call the design change *adoptable* or
*non-inferior*. It is a **promising but unresolved** change, and the episode is a
worked example of the paper's own thesis — a 0.30-RMSE margin sits below the
resolution of 52 engines, to the point where bootstrap Monte-Carlo error alone can
flip the verdict.

## 4. Hash-chain corrections (deterministic, not probabilistic)

The pre-registration §3 table quoted **local mirror** hashes taken before the
mirror was synchronised with the execution host. Corrected:

| file | §3 (wrong) | actual at freeze (server) | after the PE patch |
|---|---|---|---|
| `src/candidates/model.py` | f8f53e00 | f8f53e00 ✓ | 4c576971 (expected: `use_pe`) |
| `scripts/run_candidate.py` | 309e002c ✗ | **4e183544** | 89aa90aa |
| `src/dataset.py` | efdd3bfc ✗ | **28c9a3eb** | 28c9a3eb (unchanged) |
| `src/baselines.py` | 32d10b2f | 32d10b2f ✓ | 32d10b2f (unchanged) |

The runner change is **4 lines**, registered here in full:

1. `CONFIGS["PE"] = dict(use_baseline=True, use_ema_features=True, use_cycle=True, use_pe=True, use_graph=False, use_dual_head=False, use_model_ema=False, cls_weight=0.0, nasa_weight=0.0)`
2. `use_cycle=cfg["use_cycle"], use_pe=cfg.get("use_pe", False),` in the model constructor.

`use_pe` defaults to `False`, so A and BE take the identical code path; this was
verified by the three equality checks (§2 of the pre-registration, all diffs 0.0).
The dataset was never modified — the §3 value was simply stale.

## 5. Provenance gaps to close from E4 onward

`metrics.json` recorded `elapsed_seconds` but not wall-clock bounds or the
pre-registration hash, and the mirrored file mtimes (22:31) reflect a bulk `scp`,
not the runs. Server-side `stat` gives the real completion times:

| batch | completed (UTC+8) |
|---|---|
| A | 22:10–22:12 |
| BE | 22:21–22:22 |
| PE | 22:32–22:34 |

Launch was 21:59, so the campaign took **≈35 min**, not the "21:59–23:05" stated in
the first version of `E3_RESULT.md` (that end time was when the status file was
last polled). From E4 on, every `metrics.json` must carry `prereg_sha`,
`started_at`, `finished_at`.

## 6. Lesson recorded for §4.3

The useful sentence is not "PE passed" but:

> a pre-registered 0.30-RMSE margin on 52 engines is inside the Monte-Carlo error
> of its own bootstrap; the verdict flipped between 4k and 100k resamples.

That is the same failure mode as the R3.1 leak, one level up: the first time the
measurement was wrong, this time the *uncertainty* was under-resolved.
