# Independent audit trail

Date: 2026-11-11. Reviewers were separate Codex subagent contexts, explicitly
`gpt-6.1-sol` with `reasoning_effort=high`. All reviews were read-only. Training
and changes were executed directly by the primary Codex session, without DSH.

## Source review: priority methods

`audit_priority_sources` verified AIDGN/Two-Phase/Dual-Mixer DOI/source
identity, exact commits, original entrypoints, preprocessing and objectives.
Its source findings are persisted in SOURCE_AUDIT.json, with primary links.

- Confirmed Two-Phase paired-subset pretraining, GRL and frozen fine-phase
  encoder, hardcoded45-cycle feature attention, and transductive test-feature
  normalization. A commented test-to-train concatenation is **not** evidence
  of supervised training on official test labels.
- Confirmed Dual-Mixer paper metrics are normalized/all-window RMSE+MAPE,
  not official-engine cycles/NASA. Public-code defaults differ from preprint
  hyperparameters; FSGRI must be enabled explicitly.
- Confirmed AIDGN aging uses observed cycles / mean training lifetime, not
  true test lifetime. Window-level validation and pre-split statistics differ
  from the matched engine protocol.
- A statically suspected AIDGN scaler shape crash was **not reproduced**:
  the original scaler/assignment ran unchanged on real data. This is kept as
  a fragile surplus-column assignment, not falsely reported as a proven crash.

## Source review: remaining methods

`audit_secondary_sources` verified FFVE/FC-STGNN and the sixth CPT candidate.

- FFVE corrected public numbers follow the author's RMSE-bug disclosure.
  Real entrypoint uses Encoder+Decoder; the unused wrapper has an invalid
  constructor keyword. Per-subset blocks/long residual/LR follow the shell.
- FC-STGNN performs test-based checkpoint selection after overlapping-window
  validation. Exact patch and graph branch dimensions must be preserved.
- CPT pretraining calls its classifier twice, producing an incompatible
  scalar-to36-dimensional input. A real-data interface probe reproduced the
  error; no regression-only model was substituted.

## Training and export review

`audit_training_integrity` separately compared implementation paths with
the original source and read actual A40 logs and artifacts.

- Found a dead `ops=None` indexing error before unified training. Fixed;
 20 real-data train-only forward/backward checks then passed. FFVE was
  subsequently rechecked for all four final per-subset configurations.
- Confirmed full AIDGN aging graph; full Two-Phase GRL/pretraining/freeze;
  original Gaussian sampler, augmentation, weighted InfoNCE and regression;
  FFVE stochastic reparameterization and KL+RMSE; original FC-STGNN branches.
- Found initial author Dual exports used125 instead of the exact inverse126
  scale. Reconciliation uses raw normalized arrays, checks every true window
  against raw unit/cycle/RUL, preserves initial exports, and recomputes NASA.
- Confirmed the author Dual loader omits short test engines. Retained-engine
  endpoints stay separate from complete official-test evaluation.
- Verified first Two-Phase shuffled export reconciliation on the actual
  checkpoint: sorted(truth,prediction) multiset agreement, then correct IDs.
- Review of gather/report required explicit rejection of any upstream
  nonfinite-output replacement; author official-engine label/ID checks;
  freeze-to-config hash binding; trainable parameter counts; and collection
  of all failed attempts. These requirements are implemented in gather.py.

## Final evidence review

Pending until the finite training matrix completes. A successful import,
forward/backward check or launched batch is not a successful reproduction.
Final verdicts must use saved checkpoints/predictions and complete audit rows,
with source/coverage/numerical limitations retained in the generated report.

## Lower-priority compatibility repairs: independent audit

The independent `gpt-6.1-sol/high` secondary-source reviewer inspected the
actual A40 v4 runtime copies, imports and completed training evidence.
Verdict: PASS, no new P1. No unified official TEST inference was executed.

- FFVE imageio imports from the isolated optional_dependencies/imageio path,
  version2.34.2. Wheel SHA256 independently matched
  a0bb27ec9d5bab36a9f4835e51b21d2cb099e1f78451441f94687ff3404b79f8.
- Five created FFVE runtime copies retained byte-identical main.py/utils.py;
  model_FFVE.py differed only by task seed99 -> specified seed.
- Five created FC runtime copies retained byte-identical Model.py/Model_Base.py.
  Removing the added __iter__ stub yielded AST equality with upstream
  data_loader_RUL.py; the stub only raises NotImplementedError.
- Both methods FD001/002/003 seed101 completed, with actual FFVE train/val
  epochs and FC41-epoch evidence; remaining v4 runs were in progress, no v4
  failures at this snapshot. Both original v3 failures and patches survive.

This compatibility review is separate from the final full-matrix artifact
audit. It does not certify publication-number agreement.

## Priority72: completed-run independent audit

Reviewer: independent `gpt-6.1-sol/high` priority-source agent. Actual SSH
read-only commands exited0; stdlib metric/source checks and torch finite
weight checks. No new training, TEST inference or file changes. Verdict PASS.

Reviewer-reported checks:

```text
author: aidgn=12, two_phase=12, dual_mixer=12
unified: aidgn=12, two_phase=12, dual_mixer=12
ISSUES []
PINNED_GIT_BLOB_PASS 401
VERIFIED_SOURCE_HASHES 401
FINITE_CHECKPOINTS 72
AUTHOR_CORE_IDENTICAL 120
RAW_LABEL_CYCLE_CHECKS 72
DATA_HASH_AND_TRAIN_ONLY_SCALER_PASS 36
SPLIT_VARIANTS_PER_SUBSET FD001/002/003/004 = 1/1/1/1
```

Actual author epochs: AIDGN36–127, Two-Phase200/600, Dual14–29. Unified
epochs: AIDGN18–58, Two-Phase28–101, Dual16–33. Dual's100 is its maximum
budget, not its actual early-stopped epoch count. All36 unified cells
matched minimum validation history, selected epoch, freeze, split/scaler
training IDs and post-freeze official-file access. All four mathematical
core hashes matched local/remote/config across all36 cells.

The audit independently checked every stored truth against raw engine/cycle/
RUL. Dual's inverse126*x-1 and real missing short engines were confirmed.
The original AIDGN scaler ran unchanged; an earlier speculative static
shape concern was disproven and is not a reproduction failure.

## Final-verifier review and repair

The independent training-integrity reviewer found a P1 acceptance gap:
old verification checked stored predictions/metrics but did not bind the
report's runs.csv and summary to those metrics, or require all120 unique
registered cells. The stdlib verifier now has `--require-complete`, exact
protocol/model/subset/seed set checks, per-row metric/hash/count/time binding,
independent mean/sample-SD/t95%-CI reconstruction, and stricter nonempty
unlocked TEST-access/freeze/source checks. The existing45-run snapshot
actually passed the strengthened partial-mode verifier (606 hashes).
This is acceptance/evidence code only; no training mathematics changed.

A separate P2 review tracks missing exact historical harness versions;
original upstream snapshots and runtime patches survive. Final provenance
coverage will be disclosed explicitly rather than modifying old configs.

## Lower48 and final full-matrix acceptance

The independent secondary reviewer audited all48 lower-priority trained
cells: FFVE/FC each12 author(v4) and12 unified(v3). Verdict PASS, no newP1.
Actual author FFVE epochs18–41, unified12–33; all FC runs41epochs. All48
checkpoints/configs/finite tensors/stored prediction scores/raw labels and
all24 unified min-validation checkpoint/freeze/access checks passed.
The23 fixed upstream files passed hashes; author runtime fixes preserve
original model bytecode/source and preprocessing AST. Author/unified state
keys and tensor shapes match per subset. Raw train-engine scaler statistics
were independently reconstructed; tiny KMeans-center differences <=7.11e-15
are numerical rounding, not a different engine split.

Reviewer-reported output:

```text
COUNTS {'upstream_hashed_files': 23, 'author_ffve': 12,
        'author_fcstgnn': 12, 'unified_ffve': 12, 'unified_fcstgnn': 12}
FAILURES 0 []
```

The independent integrity reviewer then executed the complete stdlib gate:

```bash
python -B paper_b/baselines/open_source/verify_artifacts.py \
  --root paper_b/experiments/open_source_reproduction_20261111 \
  --require-complete
```

Actual output (also independently passed locally after transport):

```text
PASS: 1698 file hashes and 120 independently recomputed run metrics; complete=True; report rows and summary bound
```

Independent final cross-check found exactly120 completed remote run IDs,
no omissions, extra completions or RUNNING claims;40 summary groups and
60 rows in each protocol export. Report main36 indicator rows match summary.
All60 unified core hashes match retained exact code. The provenance audit
was independently recomputed:40 exact author launcher sources,16 historical
hashes without exact source,4 early configs without runner hash. These20
retain original upstream/runtime patches/checkpoints/predictions but limit
AIDGN/Two-Phase/Dual to PARTIAL. Git320a and recovered dbd scripts are saved
with exact SHA; an additional262144 initial-to-final text candidate search
found no further historical matches. No candidate script was executed.

Final verdict PASS with disclosed P2 provenance limitation; FFVE/FC SUCCESS,
AIDGN/Two-Phase/Dual PARTIAL, CPT original pretraining interface probe FAILED.
No reviewer trained a model or ran a new unified official TEST forward.
Three file-read events per unified run include provenance hashing and
must not be described as three test prediction passes. Checkpoint/prediction
consistency rests on original execution, freeze hashes and saved logs,
not a second official-test evaluation.

The reproducible adversarial acceptance check also passed: the actual120-run
artifact set is accepted; isolated copies with119 runs, a corrupt report-row
checkpoint hash, or a stale summary mean are rejected even after updating
the transport manifest. Source artifacts are unchanged. Exact commands and
stdout: audits/final_acceptance_checks.py and its .stdout.json.
