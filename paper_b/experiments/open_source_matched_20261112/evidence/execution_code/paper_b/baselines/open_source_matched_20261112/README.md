# DSPT-matched public baseline experiment, 2026-11-12

The completed comparison is in
`docs/paper_b_baselines/MATCHED_COMPARISON_REPORT_20261112.md`.
Artifacts: `paper_b/experiments/open_source_matched_20261112/`.
This experiment preserves the historical author-protocol and earlier unified
results; those are different conditions and must not be merged into this one.

Five full upstream methods plus two fixed DSPT references, FD001–FD004,
seeds101/202/303. Primary condition: method-author epoch ceilings with common
engine-mean validation selection/patience12/min_delta1e-4; fixed40 is a
sensitivity condition. There are94 physical supervised trainings,168 logical
condition results,91 actual whole-TEST prediction passes and3 proven reuses.
Mini-batches within a single dataset prediction are not counted as separate
whole-TEST passes. Two-Phase paired-subset pretraining is an extra-data
condition and is reported separately.

## Frozen source and data

`PROTOCOL.json` is the initial40epoch registration. The explicit validation-only
amendment before any newTEST is `AUTHOR_LIMIT_POLICY.json`, with scheduler
equivalence proof in `AUTHOR_LIMIT_EQUIVALENCE.json`. Both conditions and all
94selected weights were frozen together before the first newTEST.
`ANALYSIS_PLAN.json` fixes both DSPT references and all paired comparisons.

Training source manifests:

- Initial462files: SHA256 `3bbb4c5a6e5d48551adc75089eb4c8de5582ee7ff2c9a7dc65beaa63fac3ef9a`.
- Author-ceiling460files: SHA256 `f58a0d523decb5626a8b3822ec6d9ccefc63161a21e41f221e027bbf50112c1a`.
- Actual evaluation7files: SHA256 `7eab55f3ed72f47d163cf502e3698d035690f8eae5d1d1ffb2d736f8e6405a9b`.

`reference/` contains the byte-exact9file DSPT snapshot; the running main
workspace is never imported. No frozen DSPT model, PaperA or old experiment
was modified. The common runtime is FP32/autocastoff/TF32off; do not combine
with historical BF16 results as the same precision condition.

Restore the exact upstream code (six pinned commits, unmodified trees):

```bash
python paper_b/baselines/open_source/acquire.py
python paper_b/baselines/open_source/acquire.py --verify
```

The fetcher and manifests avoid redistributing upstream code without a
license. Per-run `environment.json` records the actual Python3.9/A40/PyTorch
environment; upstream optional plotting imports are excluded by the recorded
adapters. Place official C-MAPSS train/test/RUL files in an external directory.
`DATASET_EXPECTED_SHA256.json` pins all12 files. Normalization fits training
engines only. The source guards reject training access to TEST/RUL files.

## Reproduction in fresh output directories

Use a new project checkout on Linux/A40, with the upstream code restored.
Keep all sealed source bytes unchanged. Set the library path for your own
CUDA/PyTorch environment if necessary. The actual environment and versions
are archived per run. These commands use new output roots, not the recorded
experiment directories.

```bash
mkdir -p /path/to/new40 /path/to/newauthor
python -u -m paper_b.baselines.open_source_matched_20261112.preflight \
  --data-dir /path/to/CMAPSS --out /path/to/new40/preflight.json
python -u -m paper_b.baselines.open_source_matched_20261112.combined_queue \
  --root /path/to/new40 --author-root /path/to/newauthor \
  --data-dir /path/to/CMAPSS --workers 3 --budget-mib 36000
python -u -m paper_b.baselines.open_source_matched_20261112.joint \
  --phase freeze --root /path/to/new40 --author-root /path/to/newauthor \
  --data-dir /path/to/CMAPSS
python -u -m paper_b.baselines.open_source_matched_20261112.joint \
  --phase test --root /path/to/new40 --author-root /path/to/newauthor \
  --data-dir /path/to/CMAPSS
```

The finite queue runs all84fixed40 cases and exactly the validation-qualified
fresh author-ceiling extensions. It preserves method loss/LR/batch/windows,
shared split and the frozen seed policy. It does not continue model-only
checkpoints or tune on officialTEST. The memory admission budget covers the
total shared GPU use; never signal another project's jobs to make room.

`joint --phase test` is the supported batch entrypoint and requires full
READY, both complete component batches, all94physical identity checks and an
exclusive controller. Permanent per-run `TEST_ATTEMPT` prevents a repeated
forward even after failure. There is no automatic quota reset.

The original sealed per-run `runner --phase test` and `test_instrumented`
CLIs are retained for execution provenance; they only enforce their component
batch and are **not supported standalone entrypoints** for a joint experiment.
Post-experiment hardening adds `guarded_test.py`: it first validates joint
READY and the requested physical record. For example, this validates a
completed case without a newforward:

```bash
python -m paper_b.baselines.open_source_matched_20261112.guarded_test \
  --root /path/to/new40 --author-root /path/to/newauthor \
  --namespace fixed40 --model aidgn --subset FD001 --seed 101 \
  --data-dir /path/to/CMAPSS --audit-only
```

The production batch actually executed the sealed joint code, after a valid
READY. The new wrapper is not retroactively described as its executed source.
Arbitrary Python access to a frozen legacy module is not a security boundary;
the supported orchestration and recorded access audits establish this run's
evaluation order.

## Collection and CPU-only acceptance

After the global `JOINT_TEST_COMPLETE.json` and every output seal exist:

```bash
python -m paper_b.baselines.open_source_matched_20261112.collect_joint \
  --root /path/to/new40 --author-root /path/to/newauthor \
  --data-dir /path/to/CMAPSS \
  --target paper_b/experiments/open_source_matched_20261112 \
  --docdir docs/paper_b_baselines
python -m paper_b.baselines.open_source_matched_20261112.verify_joint \
  --root paper_b/experiments/open_source_matched_20261112
```

The standalone verifier uses only the Python standard library and performs
no model inference. It binds allphysical/logical rows, source/config/checkpoint
file hashes, actual outputs/reuse proofs, official labels, engine-SSE selection
state machine, same splits/cuts/sample counts and all summary/paired numbers.
The collector additionally loads every checkpoint on CPU, checking tensor
finiteness and canonical state hashes, and compares labels to the raw official
files. `official_truth.csv` is a capped endpoint-label extract for offline
acceptance; its source fingerprints remain in the dataset manifest.

`SUCCESS` in this new artifact is execution/evidence acceptance under the
registered common rules. Historical original-author reproduction statuses
remain AIDGN/Two-Phase/Dual-Mixer PARTIAL, FFVE/FC-STGNN SUCCESS; candidate CPT
FAILED. No test-selected tuning, seed selection, ensembles or simplified
networks were used. Three-seed confidence intervals are descriptive and
conditional on this fixed split. OfficialTEST has historical exposure.
