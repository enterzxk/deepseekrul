# Paper B public-source reproduction protocol

Registered 2026-11-11 before any new training/test result. Scope is this
baseline experiment only. Frozen DSPT, Paper A, previous runs and remote
3090 jobs are read-only. Execution is directly by Codex as explicitly
requested by the user; no DeepSeek Harness is invoked.

## Two distinct experiments

1. **Author protocol:** execute the upstream architecture, preprocessing,
   objective and training/checkpoint selection rules. Document necessary
   compatibility fixes and missing settings. Preserve test-based selection,
   cross-subset pretraining, normalized labels and window-level evaluation
   when upstream uses them, and flag them. Never use this table as a matched
   DSPT comparison. Failures get real tracebacks and a bounded repair attempt.
2. **Unified protocol:** official training engines only; `split_units` from
   `paper_b/training/data.py`, split_seed=7, fold=0, val_frac=0.2, identical
   engine IDs for every method and every seed. Seeds=101,202,303; RUL cap=125
   for training/validation/test; endpoint training RUL=0; predictions in
   cycles. Validation uses five cuts at 0.40/0.55/0.70/0.85/0.95 of each held
   out engine's lifetime, never derived from official test labels. All
   statistics are fitted on training engines. Use each method's complete
   source architecture, objectives and necessary pretraining. Any paired
   subset pretraining also excludes that subset's validation engines and is
   explicitly identified as extra cross-subset data, in a separate table.

NASA score is the **sum** over engines of exp(-d/13)-1 for d<0 and
exp(d/10)-1 otherwise, d=prediction-truth. Store its mean separately. Official
test has exactly one final prediction per engine; expected counts are
100/259/100/248. Do not mix it with all-window, OOF or validation metrics.

The registered unified implementation uses DSPT's KMeans(k=6) operating-setting
sensor z-score fitted on training engines, and left zero-padding for short
validation/test histories. Two-Phase retains standardized operating settings;
FFVE retains causal exponential smoothing. Training uses complete method-length
windows. These preprocessing choices differ from some author pipelines;
the unified table compares full architectures/objectives under this common
input protocol, not unchanged author end-to-end preprocessing. The singleton
window budget does not establish each method's optimal tuned performance.
The six clusters on FD001/FD003 do not imply six physical operating conditions.

Two-Phase unified execution uses a whole-pipeline seed: FT is initialized
before PRE, and RNG states are not reset at the phase transition. The author
scripts reseed each stage independently. This is a disclosed stochastic-stream
deviation, not a claim of seed-matched author trajectories. A read-only
initialization audit (SEED_POLICY_AUDIT.json) confirmed this before any choice
to change the policy; the frozen implementation is retained without rerunning
or selecting a different random stream from observed test results.

## Search, checkpoint selection and freeze

Use upstream learning rates, optimizers, losses and architectural widths.
Unified runs use training-only preprocessing; 14 selected sensors; Two-Phase
also retains its three operating inputs. Aging input must be the author's
causal observed-cycle aging embedding, never a function of true lifetime.
Keep FSGRI in Dual-Mixer+FSGRI and both variational and regression losses in
FFVE. Do not silently substitute MSE-only models.

Initial window search is deliberately bounded: one upstream default per
subset (AIDGN 50; Two-Phase 45, hardcoded by feature attention; Dual-Mixer 30;
FFVE 30; FC-STGNN 30/50/50/50). Search range and budget therefore equal the
singleton upstream choice, **zero window tuning trials**. More windows may
only be registered before inspecting unified official test. This avoids
spending a large budget on a search that was not requested as mandatory.

Unified checkpoints minimize pooled engine-cut validation RMSE, with
patience=12 (FFVE upstream patience=10, FC-STGNN fixed 41 epochs). Keep
upstream maximum epochs (AIDGN 300, Two-Phase pretrain 20/100 and fine-tune
200/600, Dual-Mixer 100, FFVE 100000 with stopping). No ensembling, test-based
selection or removing difficult engines. Save a run-specific freeze manifest
with config/source/checkpoint hashes before opening official test; evaluate
once. Preserve negative/raw predictions unless the upstream model itself
constrains its output; never clip predictions after seeing results.

## Gates, budget and audit

Before dispatch, require import/forward/backward fidelity checks and source
blob verification. Initial budget: at most 60 A40 GPU-hours, revised only
from measured training times. Runs use a new remote experiment root and
exclusive per-run claims; existing jobs are not stopped. Maximum two repair
attempts per failing cause, with different fixes and retained error evidence.

SUCCESS requires all four subsets and all three seeds under both applicable
protocols, complete checkpoints/predictions/config hashes/time/parameter
counts, independent architecture/data/metric audit, and a defensible match
to published metrics (relative RMSE gap <=10% per subset, without test tuning).
PARTIAL means genuine training/evaluation is available but any required
coverage, fidelity or numerical-match gate is incomplete. FAILED means no
valid trained result can be obtained from the pinned implementation after
bounded repair. A launch or synthetic forward check is never SUCCESS.

Report parameter count including frozen parameters and separately trainable
count; total training wall time includes pretraining. Aggregate all three
seeds as mean, sample SD and Student-t 95% CI; keep per-seed rows. Missing
published metrics remain missing, never transcribed from unrelated papers.

## Dated current-DSPT comparison clarification (2026-11-12)

The registered rules above describe the completed historical baseline batch.
The live desktop DSPT trainer now selects by mean per-engine validation RMSE
and trains all cycles with left-padded early prefixes. This batch selects by
pooled validation RMSE and uses complete training windows. It therefore does
not satisfy strict current-DSPT training/selection equivalence, despite the
same protocol constants, splits, dataset hashes, capped labels and official
test scoring. See PROTOCOL_COMPARISON_20261112.md/.json. No historical run,
checkpoint, prediction or configuration was changed by this clarification.
