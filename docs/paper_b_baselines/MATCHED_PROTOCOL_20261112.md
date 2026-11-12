# DSPT-matched confirmation batch — 2026-11-12

This batch replaces the old pooled-selection/complete-window table only for
comparison to the current DSPT data and selection protocol. Historical
experiments, frozen DSPT/model source and Paper A remain untouched.

The target source is captured byte-exact in `reference/` (nine files), with
its observed source HEAD and file hashes in SNAPSHOT_MANIFEST.json. The live
desktop worktree may continue changing; this batch never imports it. The
original DSPT trainer is called in its own snapshot package. A documented
runtime context disables autocast for training/validation/TEST; TF32 is off
for every method. This is a new common FP32 condition, not the historical
BF16 training trajectory. No DSPT architecture or objective is rewritten.

Primary reference: DSPT-V0-Cycle-FP32 (`full/basic/hf`, cycle on), fixed from
the existing taskbook. Auxiliary reference: DSPT-TrainDefaults-FP32
(`full/off/none`, cycle on). Both always appear in results; no TEST selection
between them. Their difference changes SA and HI together, not an isolated
HI effect. Five public baselines use the complete pinned upstream models and
core objectives from the previous source audit, with author windows and
optimizer/LR/batch. Original author-protocol results remain a separate table.

Common supervised protocol: split_seed7/fold0,80/20 engine IDs; seeds101/202/303;
cap125; validation cuts40/55/70/85/95%; all-cycle training, including early
left-zero-padded prefixes; engine-mean validation RMSE selects checkpoint
and early stop with improvement >1e-4, max40 epochs/patience12. Pooled
validation RMSE is recorded separately. Single-endpoint official TEST uses
standard cross-engine RMSE and NASA sum. Each run saves its actual weights,
validation/TEST predictions, per-epoch per-engine SSE, chosen state-content
hash, raw logs, parameter counts, times, config/code/data hashes.

Each method retains its window and optimizer/batch settings; this is equal
supervised-epoch budget, not equal gradient steps, equal compute or a claim
of exhaustive best tuning. Window tuning budget is zero. Models without
native padding masks receive the same padded sample universe without adding
new mask modules. FFVE preserves causal smoothing and actual stochastic
reparameterization. Two-Phase retains operating inputs and the prior frozen
whole-pipeline seed policy, plus paired-subset pretraining20/100 epochs;
both validation sets are excluded. Its extra data, pretraining time and
sample counts are reported in a separate resource-condition table.

84 supervised cells are registered: seven fixed configurations x four
subsets x three seeds. All are trained before any new official TEST is
opened. Training Guard intercepts builtins/io opens. An immutable all-run
BATCH_FREEZE binds source/config/data/checkpoint identities after replaying
per-engine-SSE/min_delta proofs. A permanent TEST_ATTEMPT consumes one
attempt even if evaluation fails; no generic retry releases the quota.
Old official TEST exposure is declared; this is not a pristine blind test.

Budget: at most60 A40 GPU-hours, with measured times/peak memory reported.
Failures retain their attempt and traceback; at most two cause-specific
repairs, in fresh attempts, without model substitution or TEST tuning.
Existing other GPU jobs are not stopped. Remote isolated deployment:
`/media/sangfor/vdb/user/zxk/paper_b_matched_baselines_20261112`.

Execution source was independently audited before first training. Added
identity guards re-check train bytes including paired data before freeze
and TEST; official test bytes are checked against prior recorded canonical
fingerprints only after unlock. The immutable dataset hashes are metadata,
not labels used to choose validation cuts. All original failed/preflight
attempts are preserved. Main code is frozen after preflight and before the
first training process. Reporting-only modules may be added later.

Commands (from the isolated deployment/repository root):

```bash
python -m paper_b.baselines.open_source_matched_20261112.preflight \
 --data-dir /path/to/CMAPSS --out experiments/preflight.json
python -u -m paper_b.baselines.open_source_matched_20261112.orchestrate \
 --phase train --root experiments --data-dir /path/to/CMAPSS --workers 3
python -m paper_b.baselines.open_source_matched_20261112.orchestrate \
 --phase freeze --root experiments --data-dir /path/to/CMAPSS
python -u -m paper_b.baselines.open_source_matched_20261112.orchestrate \
 --phase test --root experiments --data-dir /path/to/CMAPSS --workers 2
```

Do not overwrite or relabel the old120-run batch as this new experiment.
Final artifacts: `paper_b/experiments/open_source_matched_20261112/`.


## Completed validation-only amendment and final scope

The initial40epoch registration above remains historical. Before any new
officialTEST, PB-OS-009 added author-ceiling primary comparison and retained
fixed40 sensitivity. AUTHOR_LIMIT_POLICY.json defines exactuniformeligibility;
10fresh fullretrainings qualify. All94physicaltrainings/168logicalcases were
jointlyfrozen before newTEST. Complete results and limitations are in
[MATCHED_COMPARISON_REPORT_20261112.md](MATCHED_COMPARISON_REPORT_20261112.md).
Do not describe the amendedbudget as preregistered before initialvalidation.
