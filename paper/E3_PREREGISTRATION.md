# E3 — Pre-registration: does a bounded (sinusoidal) time encoding survive the frozen pipeline?

Status: **frozen before any code change and before any run** (2026-10-29, round 1).
Any post-hoc change is an amendment in §12 with a timestamp; the review rule is
"amendments are allowed before a number exists, never after".

Reviewer checklist (§ of the 2026-10-29 instruction) is mapped at the end.

---

## 0. Scope correction that must be accepted first

The instruction says `src/model.py`. The frozen pipeline does **not** load that
file: `scripts/run_candidate.py` imports `from candidates.model import
CandidateModel` (line 49), i.e. **`src/candidates/model.py`**. `src/model.py` is
the older STG-MambaNet and is not on this path.

If we patch `src/model.py`, nothing changes in the experiment and the
"equivalence" smoke test would pass trivially while the PE arm never runs.
All references below are therefore to **`src/candidates/model.py`**.

## 1. PE definition (item 1)

The cycle channel enters once, additively, at the pooled-token level:

```python
pooled = pooled + self.cycle_proj(cycle).unsqueeze(1)   # model.py:301
```

with `self.cycle_proj = nn.Linear(1, d_model)` (model.py:168).

**PE arm, exact definition:**

```
u    = cycle feature as produced by the frozen dataset (already min(c/250, 1))
ĉ    = 250 · u                       # exact recovery for c ≤ 250
PE_j = sin(2π ĉ / P_j), cos(2π ĉ / P_j),  j = 1..4
P    = [25, 50, 125, 250] cycles
self.pe_proj = nn.Linear(8, d_model)          # replaces cycle_proj
pooled = pooled + self.pe_proj(pe).unsqueeze(1)
```

* **Injection point**: identical to `cycle_proj` — additive on `pooled`, broadcast
  over the 50 positions. Nothing else in the forward pass changes.
* **Boundedness**: every PE channel ∈ [−1, 1] for all c; no term grows with c.
  This is the property `c/250` lacks above 250 cycles.
* **Parameters**: `Linear(1,64)` = 128 → `Linear(8,64)` = 576, **Δ = +448
  (+1.070% on 41,858)**. Declared, not hidden; §5 reports the exact count.
* **Recovery loss above 250 cycles**: `ĉ = 250` for c > 250. Measured on
  `train_FD002` only: **1,668 / 41,019 windows (4.07%)** have c > 250
  (16.9% of engines live > 250, but those windows are a small share).
  So PE is exact for 95.9% of training windows; the residual is a known,
  quantified limitation, not a silent one.

## 2. Default-off and bit-identity (item 2)

`use_pe: bool = False` added to `__init__`; when False **no module is created**
(`pe_proj` absent from `state_dict`, exactly as `cycle_proj` is absent in BE).

Three equality checks, all required:

| check | tolerance |
|---|---|
| E1: `set(state_dict.keys())` for `use_pe=False` equals the pre-change key set | exact |
| E2: forward on a saved fixed batch, same weights, `use_pe=False` | max abs diff ≤ 1e-6 |
| E3: 3 optimizer steps from the saved init state, same seed/batch order | loss diff ≤ 1e-6 |

E2/E3 use the artefact from §3, so "equivalent" is a measurement, not a promise.

## 3. Baseline captured **before** any edit (item 3)

```bash
python scripts/dsh_e3_capture_baseline.py --subset FD002 --seed 2027 \
       --variant A --out results/dsh/e3/baseline
```

writes `pred_baseline.npz` (fixed X/extras/y batch, model init state_dict seed
2027, forward output), `pred_baseline.sha256`, and the frozen-file hashes.
Recorded in `paper/ROUNDS.md` round 1 before the patch.

Frozen hashes at freeze time (sha256, full):

| file | sha256 |
|---|---|
| `src/candidates/model.py` | `f8f53e008ae81bbbb9b83c7bf0ebacc247a33b6dfa4f55b76c6a4f3b29cee403` |
| `scripts/run_candidate.py` | `4e183544e1ba934caec4d3094ea774a14e6cdd098569f3916b9e01537dc7ea9f` |
| `src/dataset.py` | `28c9a3ebe0297579b50ebca70ec4a2761d4be5d59c7babf88ab52f0f83ccaf81` |
| `src/baselines.py` | `32d10b2fd53ad6f1a3e65cf6d9e1fc48d501e53accbd446be3a163100ed0cbfa` |

> Corrected 2026-10-29 (Amendment 1 §4): the first version of this table quoted the
> **local mirror** hashes (`309e002c…` runner, `efdd3bfc…` dataset) taken before the
> mirror was synchronised with the execution host. The values above are the ones that
> actually ran.

Backup before edit: `src/candidates/model.py.bak_r3_1` + the same four hashes
written to `src/FROZEN_SHA256.txt`.

## 4. Arms, seeds, budget (item 4)

| | |
|---|---|
| data | FD002, frozen `prepare()` (208 train / 52 val engines, `--split-seed 7`) |
| arms | `A` (cycle), `BE` (no cycle), `PE` (this proposal) |
| seeds | 2027, 2028, 2029 (`--seed`, model+dataloader) |
| epochs / patience | **80 / 20** (`--epochs 80 --patience 20`) |
| eval | validation RMSE (selection) + matched pseudo-test (report only) |

Command template (one line per run, 9 total):

```bash
python scripts/run_candidate.py --subset FD002 --variant {A|BE|PE} --seed {2027|2028|2029} \
       --epochs 80 --patience 20 --split-seed 7 --out experiments/e3/{variant}_s{seed}
```

Artifacts land at `experiments/e3/<variant>_s<seed>/FD002/<seed>/metrics.json`
(glob pattern verified against the `r3_cycle` layout).

## 5. Decision rule (item 5)

Primary quantity: ΔRMSE = RMSE(PE) − RMSE(A) on the 52 held-out engines,
paired, averaged over the three seeds, with an **engine-clustered bootstrap**
(4000 resamples, engines resampled within seed) giving a one-sided 97.5% upper
bound.

| outcome | rule |
|---|---|
| **adoptable** | upper bound < **+0.30** AND NASA(PE) ≤ 1.05 × NASA(A) AND late(PE) ≤ late(A) + 0.02 |
| **PE better** | additionally, the two-sided 95% CI of ΔRMSE lies entirely below 0 |
| **reject** | lower bound > +0.30, or a risk condition violated |
| **tie** | adoptable but CI crosses 0 → manuscript may say "tends to PE", never "better" |

Parameters are reported for every arm; a PE arm that only wins with more
capacity is reported as such.

## 6. No 30-epoch screening (item 6)

We already paid for this lesson: the 30-epoch probe said −1.36 (remove time is
better) and the 80-epoch-class frozen pipeline said +1.26 (time helps). E3 runs
at 80 epochs directly; no intermediate-budget run may be used to select or
reject an arm, and any 30-epoch number that appears is labelled "not a result".

## 7. Reporting (item 7)

Per arm × seed: RMSE, MAE, NASA mean, late, severe e>10, params, best epoch.
Aggregated: mean ± sd across seeds; engine-clustered CI for every pairwise
contrast; lifespan bands (short/mid/long by tertile) with RMSE and bias.
Tables are produced by `scripts/paper_e3_table.py` from `metrics.json` — no
hand-entered numbers.

## 8. Failure handling (item 8)

* E1–E3 equality fails → restore `.bak_r3_1`, stop, report; no "small fix" loop.
* PE significantly worse → record as a rejected design change; **no rerun, no
  seed shopping, no capacity top-up**.
* Runtime error → keep the log, fix only the harness (never the arm), rerun the
  identical command; the rerun is logged in ROUNDS.

## 9. Cost (item 9)

Measured on the R3.1 runs: ~460 s per 30 epochs on ~32k windows → 15.3 s/epoch.
80 epochs ≈ **20.4 min/run**; 9 runs ≈ **3.1 GPU-hours**, ≈ 65 min wall at 3-way
concurrency on the A40. Recorded in the budget ledger after the run.

## 10. Artifacts (item 10)

For each run: `metrics.json` (with `code_sha256`), `matched_cutoff_predictions.csv`,
stdout/stderr log, `params`. For the round: this file's sha256, the baseline
`pred_baseline.npz` + sha, `src/FROZEN_SHA256.txt`, and
`results/dsh/e3/E3_contrasts.json` (the bootstrap output).

## 11. What would make this pre-registration wrong

If PE's advantage comes only from the +448 parameters, or if the 4.07% saturated
windows are exactly the long-lived engines that drive the gain, the correct
conclusion is "the encoding change is confounded", and the arm is dropped rather
than reported as a win. Both checks are in §7 (lifespan bands, parameter count).

## 12. Amendments

**Amendment 1 — 2026-10-29 23:20, after the run, before any manuscript text.**
The pre-registered rule was applied with a **non-converged** bootstrap: 4,000
resamples gave a 97.5% upper bound of +0.291 (< +0.30), but 100,000 resamples × 5
RNG seeds give +0.309…+0.324 (mean +0.317), and 4,000 × 20 RNG seeds give a median
of +0.315. **The rule therefore FAILS: non-inferiority is not established.** The
margin is not changed, the point estimate is kept, and the 0.291 draw is retained
as a Monte-Carlo artefact. Full record: `paper/E3_AMENDMENT_1.md`.
Consequences: `E3_RESULT.md` carries a correction banner and a "Bootstrap
stability" section; claim C14 is downgraded; §4.3 may only say "promising but not
established". Also corrected in §3: the runner/dataset hashes were stale local-mirror
values (see Amendment 1 §4).

---

### Reviewer checklist mapping

| required item | § |
|---|---|
| 1 PE definition (k, period, formula, bounded, injection, params) | §1 |
| 2 default off + bit identity | §2 |
| 3 pre-change baseline artefact | §3 |
| 4 three arms, seeds, 80/20, same split & pseudo-test | §4 |
| 5 decision rule | §5 |
| 6 no 30-epoch screening | §6 |
| 7 reporting (5 metrics + per-seed + clustered CI + bands) | §7 |
| 8 failure handling | §8 |
| 9 cost | §9 |
| 10 artefacts (code_sha256, pred, logs, PREREG sha) | §10 |
