# C-MAPSS baseline performance and evaluation-protocol audit

Status: **partial** — data-side sections (1–6) are complete and reproducible from
`paper_b/results/baseline_audit/dual_criteria_all_runs.csv`. The literature
comparability matrix (§7) and the novelty judgement (§8) are still being filled
by their owners and are marked accordingly.

Baseline commit: `4bfba0b` (branch `punet-interval`).
No historical artefact was modified: this audit only reads existing
`test_pred.csv` / `metrics.json` and writes new files under
`paper_b/results/baseline_audit/`.

## 0. Summary of what is now established

1. `paper_b/training/data.py::make_test_windows` clips the official test RUL to
   `min(rul_map[unit], 125.0)` at line 245. The `y_true` column stored in every
   `test_pred.csv` is exactly this clipped value (verified per engine, 113 runs,
   **0 mismatches**).
2. Because the labels are clipped but the predictions are not, the reported
   test metrics are *capped-label* metrics. Re-scored against the raw
   `RUL_FD00X.txt` values, FD002 and FD004 RMSE roughly **double** and NASA
   score rises **4–6x**. 113 runs have test predictions; 109 are eligible after
   excluding four cells evaluated outside their freeze manifest.
3. No model escapes this: the inflation is a property of the protocol, not of a
   particular architecture.
4. The clipped-label error is *not* dominated by the clipped engines under the
   capped criterion (11–17% of SSE), which is precisely why the artefact is
   invisible in the numbers that have been reported so far.
5. There is a hard arithmetic floor `B = sqrt(mean(max(rul-125,0)^2))` on any
   raw-label score from a capped-output model: 3.74 / 18.32 / 4.71 / 19.53 for
   FD001–FD004. Published FD002/FD004 values sit **below** that floor, so they
   cannot be raw-label readings. **Capped is the literature-consistent
   criterion; raw is this audit's own stricter addition.**

## 1. Where the clipping happens

```python
# paper_b/training/data.py:233-248
def make_test_windows(test_frame, values, clusters, pop_ref, span, rul_map, max_rul=125.0):
    ...
    us.append(int(uid)); ys.append(float(min(rul_map[int(uid)], max_rul)))
```

- `rul_map` is loaded from `data/CMAPSS/RUL_FD00X.txt` and holds the **raw**
  official RUL for each test engine.
- `prepare_test` (`data.py:313`) is the only caller, and it is reached only from
  `paper_b/evaluation/evaluate_test.py` and read-only probe scripts. `train_one`
  never calls it, so the test split cannot leak into training or selection.
- The scaler used at test time is rebuilt from the training fold
  (`_prepare_split`), not fitted on test.

## 2. Dual-criteria recomputation (113 runs, no retraining)

Every run with a `test_pred.csv` was re-scored twice: against the clipped label
(the value stored in `y_true`) and against the raw `RUL_FD00X.txt` label. MAE and
the standard asymmetric NASA score (`exp(-d/13)` early, `exp(d/10)` late) were
computed for both.

| subset | runs | engines with raw RUL>125 | capped RMSE | raw RMSE | ratio | capped NASA | raw NASA |
|---|---|---|---|---|---|---|---|
| FD001 | 37 | 11% | 11.353 | 12.410 | 1.09 | 220.6 | 245.6 |
| FD002 | 24 | 22% | 11.118 | **22.965** | **2.07** | 615.6 | **3722.7** |
| FD003 | 24 | 15% | 11.411 | 12.758 | 1.12 | 320.0 | 349.1 |
| FD004 | 24 | 27% | 12.423 | **23.881** | **1.92** | 839.4 | **3336.2** |

Counts are the **eligible** set (109 runs): four A8 cells evaluated outside
`FREEZE_ABLATION_SEED101.json` are flagged `OUT` in the `freeze` column of the
CSV and excluded from every mean. They are not re-declared after the fact —
retro-fitting them into the manifest would rewrite the evaluation order, which
is what the freeze exists to prevent. Including them would move FD001 from
11.353 to 11.284 capped and 12.410 to 12.338 raw; FD002–FD004 are unaffected
because none of the out-of-manifest cells are there.

Per-run values are in `dual_criteria_all_runs.csv`
(`rmse_capped`, `mae_capped`, `nasa_capped`, `rmse_raw`, `mae_raw`, `nasa_raw`).

**Coverage is narrower than the run count suggests.** The repository holds 265
runs, but only 113 have a `test_pred.csv`; the other 152 were evaluated on
validation only and cannot enter any capped/raw table. Of the 113, four are
outside their freeze manifest and are excluded, leaving **109 eligible**. The
authoritative copy is `lab3090:~/rulb/paper_b/results/raw/` — the local Windows
checkout mirrors 83 runs (31%) and `a40` holds no test predictions at all, so
the local tree must never be used to answer "does this run exist".
Full inventory: `RUN_INVENTORY.md`.

FD001 and FD003 barely move because their engines are mostly late-life at the
truncation point. FD002 and FD004 move enormously because 22% and 27% of their
test engines are still far from failure and their true RUL is above the cap.

### 2b. A hard lower bound on any raw-label score

If a model never predicts above 125, its raw-label RMSE cannot be below

```
B = sqrt( mean( max(rul - 125, 0)^2 ) )
```

because every engine whose true RUL exceeds 125 contributes at least
`(rul - 125)^2` no matter what the model does. `B` is therefore the floor for
*any* capped-output model scored on raw labels.

| subset | test engines | max raw RUL | fraction >125 | B |
|---|---|---|---|---|
| FD001 | 100 | 145 | 11.0% | 3.738 |
| FD002 | 259 | 194 | 22.0% | **18.317** |
| FD003 | 100 | 145 | 15.0% | 4.710 |
| FD004 | 248 | 195 | 27.0% | **19.529** |

This is the decisive number for §8. A published FD002 RMSE of about 12 is
**below** B = 18.317, which is arithmetically impossible for a model whose
outputs are capped at 125 and which is scored against raw labels. The reported
values must therefore come from one of three situations: the authors also
clipped the test labels (most likely, and usually not stated), the model
extrapolates above 125 (not claimed), or the evaluation covers a subset of
engines (not stated). None of these is a raw-label reading.

Consequence: **capped is the literature-consistent criterion; raw is an
additional, stricter criterion introduced here.** The raw numbers are a
contribution of this audit, not a fair stick to beat published numbers with.

## 3. How much do the clipped engines contribute?

Two different questions, two different answers:

- **Under the capped criterion** (the one previously reported), engines with raw
  RUL > 125 contribute only **11.8% (FD001), 13.8% (FD002), 7.1% (FD003),
  16.7% (FD004)** of the total squared error. This is why the clipping is
  invisible in the reported numbers: the clipped engines look *easy* because
  their target was moved to meet the model.
- **Under the raw criterion**, those same engines dominate: the models predict
  at most ~127–131 and saturate near 125, while the true label can be 150+.

Mean maximum prediction per subset: FD001 127.1, FD002 128.9, FD003 126.5,
FD004 130.8. The number of predictions above 125 per run is small (3.9 / 10.9 /
3.2 / 25.3), i.e. the models have learned the cap and stop there.

## 4. Does the ranking change?

**No, not within a subset.** Ranking models by capped RMSE and by raw RMSE
inside FD001, FD002, FD003 or FD004 gives essentially the same order; the
differences between architectures are small compared with the ~2x level shift
that the criterion applies to every model.

A caution on a number that is easy to get wrong: a first pass that averaged
RMSE over *all subsets present per tag* appeared to show the vanilla Transformer
overtaking every DSPT variant under the raw criterion. That was an artefact —
`VAN_*` only has FD001 runs, so it was being compared against four-subset means.
Per-subset comparison removes the apparent flip. Do not average across tags with
different subset coverage.

## 5. Data-leakage check

- Train/validation split is by engine id; no engine appears in both. This was
  verified independently in the earlier protocol audit and has not changed.
- Test engines are the official held-out engines; `prepare_test` rebuilds the
  training-fold scaler and never re-fits on test.
- The `cycle` input is the observed truncation index at inference time, not a
  label-derived quantity, so it is legitimate — but it is also a very strong
  feature (see §6).
- **No label leakage was found.** The protocol defect is a *label definition*
  problem (clipping), not a train/test contamination problem. These should not
  be conflated: clipping inflates reported scores, it does not make the model
  see the answer.

## 6. Why is the plain base so strong?

Three effects, all measurable:

1. **Saturation.** A vanilla Transformer with the cycle feature reaches
   12.60 / 15.05 (FD001, `VAN_dsp_cycle` / `VAN_raw`) while the best DSPT cell
   reaches 10.29. The whole method space spans ~2 RMSE on FD001–FD003 against a
   seed spread of 0.2–0.9.
2. **The cycle feature carries most of the signal.** 173 of 185 runs enable
   `use_cycle`. Turning it off costs 0.9–2.1 RMSE on FD001
   (`VAN_raw` 15.05 vs `VAN_dsp_cycle` 12.60).
3. **The cap compresses the target range.** With labels clipped at 125, the
   hardest engines have their target moved toward the model's own saturated
   output.

## 6b. Four different things are called "base"

This is the single most confusable point in the project. All four are
`FD001/s101`; 27 training and split fields are identical, so the difference is
entirely structure and input.

| | A8_base | Factorial base | VAN_raw | VAN_dsp_cycle |
|---|---|---|---|---|
| ms / sa / hi | off / off / none | single_proj / mean / hf + λ0 | full (inert) / off / none | full (inert) / off / none |
| `use_cycle` / input | True / raw | True / raw | **False** / raw | True / **dsp** |
| parameters | 77058 | 81218 | 72257 | 74625 |

- `A8_base` has all three switches off — a bare model.
- `Factorial base` keeps the single-scale convolution and mean pooling, i.e. the
  DSPT baseline structure with HI supervision disabled. The extra 4160
  parameters are the HI branch itself.
- The two therefore differ by **0.35 RMSE at engine level (11.0851 vs 11.4316)**
  and **0.22 pooled (11.8739 vs 12.0932)** on FD001 over three seeds — not by
  the ~1.0 that an engine-level-minus-pooled subtraction would suggest.

Any table that says "base" must say which of the four it means.

## 7. MS / SA / HI: is there independent gain?

A8 8-config grid, A8-only main effects (with minus without; negative = helps).
Test predictions exist for seed 101 only, so these are single-seed deltas and
should not be read as seed-robust.

| switch | criterion | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|---|
| ms | capped | −0.41 | +0.10 | −0.78 | −0.10 |
| ms | raw | −0.48 | +0.13 | −0.67 | −0.08 |
| sa | capped | −0.54 | −0.59 | −0.06 | **+0.26** |
| sa | raw | −0.31 | −0.51 | +0.03 | **−0.82** |
| hi | capped | −0.03 | +0.07 | −0.33 | +0.32 |
| hi | raw | −0.05 | +0.03 | −0.22 | +0.17 |

- `ms` is stable across criteria (helps FD001 and FD003, neutral elsewhere).
- `hi` is stable across criteria.
- **`sa` flips sign on FD004**: +0.26 under the capped criterion, −0.82 under
  the raw one. Since §8 establishes that capped is the literature-consistent
  criterion, the main table keeps capped and "sa hurts FD004" may be stated —
  but it must carry the sensitivity note, and it may not be presented as a
  label-independent mechanism.
- Nothing here is large relative to seed spread except `ms` on FD003 and `sa` on
  FD002, both of which survive both criteria.

Two bookkeeping facts that matter for any merged table:

- `A8_full_L70` and `F_full_basic_hi1` are **the same twelve runs under two
  tags** (configs identical except `tag`). Any deduplication key must map them
  explicitly or the cells are double-counted.
- `FREEZE_ABLATION_SEED101.json` existed only on the GPU host and was not in the
  repository, so the freeze order could not be audited from a clone. It is now
  version-controlled.

## 8. Which published numbers are directly comparable?

Five candidate papers were read in full and tabulated in
`PAPER_PROTOCOL_MATRIX.md`. Findings:

1. **No paper releases official code** (GitHub API search returned 0/5 on a
   second pass). Protocols could only be verified from the text. The Codex
   reproductions are *re-implementations*, not author code; they are more
   comparable than the papers in one sense (the protocol is known) and less in
   another (they are not the authors' numbers), and must be labelled as such.
2. **None of the five states whether the test labels or the predictions are
   clipped**, although all five state their training cap. One paper
   (DCCG-Mamba) uses caps of 125/130/125/135 instead of a uniform 125.
3. **No paper is directly comparable.** Four are conditionally comparable under
   the capped criterion; DCCG-Mamba is not comparable on FD002/FD004 because its
   training cap (130/135) differs from ours (125).
4. **The authors' reported values cannot be raw-label readings** — see §2b. The
   FD002 and FD004 values fall below the arithmetic floor `B` for any model
   capped at 125.
5. **The raw criterion is this work's own stricter addition.** It is reported
   separately and never placed side by side with published values.

A residual item that was not resolved: ST-KDFormer's Eq. (29) extracts as
`late -> /13, early -> /10`, which is the opposite of the asymmetry its own text
describes. This is likely a PDF layout artefact. It is recorded as
*page-layout pending* and was not used to change any comparison.

## 9. Is the current structure enough for an IEEE TIM method contribution?

**PENDING-L2 — deliberately not decided here.** A novelty judgement is an L2
decision under this project's routing rules and may not be self-answered by the
executing agent. The advisor channel was unavailable when this section was
written: `mcp__codex__codex` fails with `WinError 2` (the `codex` executable is
not on PATH) and the local proxy `127.0.0.1:15721` referenced by
`~/.codex/config.toml` is not listening. This is recorded as a blocker, not as a
verdict.

What the advisor will be asked, and with which evidence:

**Question.** Does the DSPT line carry enough methodological novelty for IEEE
TIM, and if not, what is the minimal change that makes it sufficient?

**Evidence supplied.**

| claim | number | source |
|---|---|---|
| method space on FD001–FD003 spans ~2 RMSE | 10.29 best vs 12.60 vanilla | §6 |
| seed spread is the same order as the effects | sd 0.2–0.9 | `SEED_ROBUSTNESS.md` |
| module effects are subset-specific and partly criterion-dependent | sa on FD004: +0.26 capped, −0.82 raw | §7 |
| no published number is directly comparable | 0/5 papers with code; 0/5 state test clipping | §8 |
| the benchmark is cap-saturated | 22%/27% of FD002/FD004 engines above cap | §2 |

**What would make the advisor's answer falsifiable.**

- If an engine-level re-selection (R2 `metrics.py`) changes the A8 ranking, the
  "effects are within noise" claim weakens.
- If a cap-free evaluation (predicting raw RUL directly, no clipping at training
  or test) reproduces the same module ordering, the saturation argument in §6
  does not explain the flatness and the method space is genuinely small.
- If any of the five papers is shown to release code that contradicts §8, the
  comparability conclusion changes.

Until the advisor rules, the reversible work continues: the dual-criteria
numbers, the A8 grid and the protocol audit are all retained as evidence, and no
new module search is started.

## 10. Reproduction

```bash
# on the GPU host, from the repository root
python /tmp/dual.py          # writes paper_b/results/baseline_audit/dual_criteria_all_runs.csv
python /tmp/rank.py          # per-subset ranking under both criteria
```

`/tmp/dual.py` reads only `paper_b/results/raw/*/*/s*/test_pred.csv` and
`data/CMAPSS/RUL_FD00X.txt`; it writes nothing outside `baseline_audit/`.
