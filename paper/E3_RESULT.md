# E3 — Result: a bounded time encoding is **promising but not established**

> **CORRECTION (2026-10-29 23:20, `paper/E3_AMENDMENT_1.md`).** The first version of
> this file declared the pre-registered non-inferiority rule satisfied on a
> 4,000-resample upper bound of **+0.291**. With 100,000 resamples × 5 RNG seeds the
> bound is **+0.309 … +0.324** (mean **+0.317**), and 4,000 × 20 RNG seeds gives a
> median of **+0.315** (70% ≥ 0.30). The margin **+0.30 is not met** → verdict is
> **FAIL (non-inferiority not established)**. The point estimate (−0.897, 3/3 seeds
> favouring PE) is unchanged; only the strength of the conclusion changes.

Run: 2026-10-29 21:59–22:34 UTC+8 (server `stat` on the nine `metrics.json`),
A40, 9 runs × FD002 (208/52, `--split-seed 7`),
`--epochs 80 --patience 20 --d-model 64 --num-layers 1 --ref-mode global`.
Design: `paper/E3_PREREGISTRATION.md` (sha `c8f6d653`, frozen before any code change).
Artifacts: `experiments/e3/*/FD002/*/{metrics.json,matched_cutoff_predictions.csv}`,
`results/dsh/e3/E3_contrasts.{json,csv}`, `results/dsh/e3/bootstrap_stability.json`.

## Verdict: **FAIL — non-inferiority not established** (Amendment 1)

| arm | RMSE (matched, 52 engines) | MAE | NASA mean | late | severe e>10 | params |
|---|---:|---:|---:|---:|---:|---:|
| A (cycle `c/250`) | 12.676 ± 0.363 | 9.178 | 3.537 | 0.538 | 0.218 | 41,858 |
| BE (no time) | 15.329 ± 1.117 | 11.713 | 4.439 | 0.519 | 0.289 | 41,730 |
| **PE (bounded sin/cos)** | **11.768 ± 0.510** | **8.529** | **2.729** | **0.487** | **0.212** | 42,306 |

Per-seed RMSE: A 13.086 / 12.393 / 12.550; BE 14.214 / 16.449 / 15.325;
PE 12.355 / 11.526 / 11.424.

Engine-clustered bootstrap (4000 reps, engines resampled within seed):

| contrast | mean ΔRMSE | 95% CI | excludes 0 |
|---|---:|---|---|
| PE − A | **−0.908** | [−2.028, +0.291] | no |
| BE − A | **+2.653** | [+1.322, +4.047] | **yes** |
| PE − BE | −3.561 | [−5.042, −2.200] | yes |

Pre-registered rule, evaluated with the **converged** bootstrap: upper bound
**+0.317 > +0.30** → **not non-inferior**. NASA (2.729 ≤ 1.05×3.537) and late
(0.487 ≤ 0.538 + 0.02) still pass, but the rule is conjunctive. "PE better" also
fails (CI crosses 0). The manuscript may write **"tends to PE, not established"**.

### Bootstrap stability

| design | 97.5% upper bound |
|---|---|
| 4,000 reps, RNG 20261029 (first version) | +0.291 |
| 100,000 reps × 5 RNG seeds | +0.309 / +0.321 / +0.324 / +0.316 / +0.309 |
| 4,000 reps × 20 RNG seeds | median +0.315, 70% ≥ 0.30 |

The 0.291 was a low draw of a distribution whose median already exceeds the margin.
Kept on record, not deleted.

## What the paper can now say (§4.3)

1. **The time channel is load-bearing at the confirmed budget.** Removing it costs
   **+2.65 RMSE [1.32, 4.05]** at 80 epochs — larger than the +1.26 measured at 30
   epochs (§G). The shortcut is not an artefact of under-training, and it is not an
   artefact of BE's early stopping either: rerunning BE with `--patience 80` (T1,
   `paper/T1_BE_SENSITIVITY.md`) reproduces the same best checkpoint and the same
   matched RMSE to four decimals.
2. **A bounded encoding is promising but unresolved.** Replacing `c/250` with
   sin/cos lowers the point estimate by 0.90 (3/3 seeds) and NASA by 23%, but the
   pre-registered margin is **not** met once the bootstrap is converged. It may not
   be called adoptable.
3. **It is not a win, and now it is also a measurement lesson.** A 0.30-RMSE margin
   on 52 engines sits inside the Monte-Carlo error of its own bootstrap; the verdict
   flipped between 4k and 100k resamples. This is the second worked example of the
   paper's thesis (the first being the R3.1 leak): first the estimate was wrong, now
   the uncertainty was under-resolved.

## Threats to validity (must appear in §4.3/§6)

* **Capacity is not matched**: PE adds +448 parameters (+1.070%). Since the claim is
  no longer a claim of success, this mostly matters for any follow-up that tries to
  establish superiority; a capacity control (same 8→64 projection fed with
  uninformative channels) was **not** run and is listed as follow-up (T5).
* **4.07% of FD002 training windows saturate** (`c>250`), where `ĉ = 250`; PE is exact
  for the other 95.9%.
* **Early stopping differs across arms**: BE stopped at 32–44 epochs (best 12–24),
  A at 43–64, PE at 37–46. This is the specified patience, not a tuned choice, but it
  means the arms saw different compute.
* **FD002 only, 3 seeds, 52 held-out engines.** The ±0.29 bound is a *paired* bound:
  pairing removes engine-to-engine variance, so it is much tighter than the unpaired
  MDE (~1.6 on 260 engines). It must not be quoted as a general resolution limit.

## Equivalence evidence (pre-registered §2/§3)

`use_pe=False` is bit-identical to the frozen model:

| check | result |
|---|---|
| state_dict key sets | identical |
| forward on the saved fixed batch | max abs diff **0.0** |
| 3 optimizer steps from the saved init state | max loss diff **0.0** |

Baseline captured **before** the patch (`results/dsh/e3/equality/`), hashes in
`src/FROZEN_SHA256.txt`; post-patch code sha256:
model `4c576971f32c915b…`, runner `89aa90aa7ebb046c…`.

Note: the equality harness builds with `num_layers=2` (a stricter code-path test),
while the arms run `num_layers=1`; parameter counts were verified separately for the
run configuration (A 41,858 / PE 42,306).

## Rejected alternative reading

The 30-epoch probe (§F) said "removing time is better (−1.36)"; the 80-epoch frozen
pipeline says the opposite (+2.65). E3 therefore also closes that contradiction in the
paper: the probe's sign was a sample-size artefact, and no 30-epoch number may appear
as a result.
