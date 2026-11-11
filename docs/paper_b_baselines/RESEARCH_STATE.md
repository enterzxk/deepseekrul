# Paper B public baseline reproduction — recovery state

Date: 2026-11-11. This is a separate Paper B experiment, not a change to
Paper A's research stage. User explicitly authorized direct Codex execution
(no DSH), independent high-reasoning audits, local commits and GitHub push.

- Worktree: `C:/Users/20562/.codex/worktrees/paper-b-open-source-20261111`
- Branch: `codex/paper-b-open-source-reproduction`, base `8a79b2c`.
- Main workspace had unrelated modified `research/agent-state/{state.json,current_task.md}`;
  those files are untouched and are not included in commits.
- Remote authorized compute: `a40`, Python
  `/home/yndx/miniconda3/envs/a40_env/bin/python` (torch 2.5.1+cu121).
- Remote exclusive root: `/home/yndx/paper_b_public_baselines_20261111`.
- Existing `lab3090:/home/user/rulb` jobs are not stopped or changed.
- Dataset is read-only `/home/yndx/deepseek-rul/data/CMAPSS`.
- Frozen DSPT, `paper_b/configs/protocol.json` and old results remain unchanged.

## Completed

Six exact commits acquired; all 433 preserved original files passed byte/SHA
verification. Originals remain local/remote unmodified. Source acquisition
script and lock/manifests are publishable; unlicensed upstream code is not
re-uploaded into this GitHub repository.

Three independent `gpt-6.1-sol/high` source/training audits completed. Main
implementation imports original model classes and retains AIDGN aging graph,
both Two-Phase stages and GRL/frozen encoder, Dual-Mixer FSGRI, FFVE stochastic
KL+regression loss, and FC-STGNN graph patches/branches. Twenty train-only
real-data forward/backward checks passed. FFVE four-subset configuration was
subsequently corrected to the author's shell defaults (2 blocks on FD002/4,
long residual on FD004); a targeted four-subset recheck is in progress.

Author protocol training is genuinely running. `experiments/v1` contains
Two-Phase and Dual-Mixer+FSGRI; AIDGN v1 failed on eager import of unused
N-CMAPSS h5py. AIDGN v2 removes only optional task/model imports and has
completed FD001 three seeds, with FD002 in progress. All failures are kept.

## Evidence corrections that must remain visible

1. Early Two-Phase exports had shuffled order with row indexes, not engine IDs.
   `reconcile_author.py` reloads the same checkpoint, emits engine-ordered
   predictions and requires sorted `(truth,prediction)` multiset agreement.
   FD001/s101 passed. Corrected metrics are `RECONCILED_metrics.json` and take
   precedence over initial metrics; the initial files remain as evidence.
2. Initial Dual-Mixer author exports incorrectly multiplied normalized labels
   by125. Correct inverse is normally `126*x-1`. The original normalized
   arrays are preserved. Reconciliation checks every true window label against
   raw unit/cycle/RUL; FD001/s101 passed. Recompute NASA from corrected cycles.
   The author's all-window score and retained-engine endpoints are separate;
   the original loader omits 6 short FD002 and 11 short FD004 test engines.
3. FFVE author RMSE and NASA use two different stochastic draws. Keep both
   the reported original RMSE and same-prediction recomputed metrics explicit.
4. AIDGN source's scaler also fits the label column and assigns a DataFrame
   with a surplus column; this looked suspicious statically but its unchanged
   original loader has run successfully. Do not claim a proven shape crash.

## Remaining work (continue without asking)

Finish priority author batches on all four subsets and seeds101/202/303;
then unified runs, with new freeze manifests and no test-based selection.
Run FFVE/FC-STGNN original+unified after priority jobs. CPT is a sixth candidate:
source audit found a pretraining representation/classifier shape mismatch;
do not substitute a plain regression Transformer.

Collect every checkpoint, prediction/config/hash, history, runtime and stdout;
reconcile early exports; independently recompute metrics and coverage counts;
generate separate published/author/unified tables and the requested
`OPEN_SOURCE_REPRODUCTION_REPORT.md`; record SUCCESS/PARTIAL/FAILED honestly.
Commit only this Paper B scope and push the branch to `origin` (authorized).

Read-only monitor:

```bash
ssh a40 'cd /home/yndx/paper_b_public_baselines_20261111; /home/yndx/miniconda3/envs/a40_env/bin/python -m paper_b.baselines.open_source.monitor --root experiments'
```

Do not restart completed task IDs. `RUNNING.json` is an exclusive O_EXCL
claim. `FAILED.json` and attempt-specific roots preserve failed attempts.
Initial computation budget: 60 A40 GPU-hours; per-method original caps and
the singleton window search are registered in `PROTOCOL.md`.

## Live continuation update

The original per-model dispatchers were replaced with `matrix.py` (only
orchestrator PIDs were stopped; existing training children were adopted,
never interrupted). The finite four-stage batch is now `matrix.log` in the
remote root, with4 concurrent slots. It keeps the stated priority and will
run all120 author+unified cells before completion. `matrix_stage_*.json` and
`matrix_complete.json` are terminal evidence; a running job is not completion.

Latest collected snapshot: **23 genuine author trainings independently
audited**, no unified run yet. Training remains in priority-author stage.
The final report generator exists (`report.py`), but its remote output is
provisional and must be regenerated after the terminal matrix records. Do
not push or present provisional status as final. Current raw counts may have
advanced; use the compact monitor, not this stale snapshot.

Remote artifact target is
`paper_b/experiments/open_source_reproduction_20261111/`; gather using
`python -m paper_b.baselines.open_source.gather --remote-root experiments
--data-dir /home/yndx/deepseek-rul/data/CMAPSS --target <target>`, then generate
the report from SOURCE_AUDIT.json using `report.py`. Their independent audit
now checks author engine IDs/official labels, freeze-to-config binding,
trainable counts, all FAILED attempts, and rejects any upstream nonfinite
prediction replacement. The artifact manifest is regenerated after summary.

Final pre-push tasks: collect/reconcile all real runs; fix genuine failures
with retained attempts; three independent high-reasoning final audits;
download artifacts/report to this worktree; verify transport SHA manifest;
append Paper B wiki/results pointers; assert Paper A/DSPT files unchanged;
commit only this scope and push the authorized branch. A generated
`paper_b_source_deploy.tar.gz` at this worktree root is a temporary deployment
bundle, not a deliverable and must not be added to git.
