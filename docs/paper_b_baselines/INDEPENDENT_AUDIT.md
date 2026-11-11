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
