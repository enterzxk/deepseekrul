# Public-source Paper B baseline reproduction

This package imports full upstream implementations. It does not edit DSPT,
Paper A, the frozen Paper B protocol, or previous experiment results.

**2026-11-12 comparison correction:** this recorded batch shares engine
splits, labels, validation cuts and official-test scoring with DSPT, but its
pooled validation selection and complete-only training windows differ from
the current desktop DSPT engine-mean selection/all-cycle prefix construction.
It is not a strict current-DSPT matched comparison. See
`docs/paper_b_baselines/PROTOCOL_COMPARISON_20261112.md` before citing results.

## Source preservation

Run on a networked computer from the repository root:

```bash
python paper_b/baselines/open_source/acquire.py
python paper_b/baselines/open_source/acquire.py --verify
```

`sources.lock.json` pins six exact commits. Each
`upstream/<method>/SOURCE_MANIFEST.json` records the entire Git tree and
SHA256/Git blob hashes of the original author code. Unmodified source is
downloaded into separate method directories. Datasets, images and original
large checkpoints are listed but not downloaded. This repository tracks
manifests and the fetcher rather than re-uploading unlicensed source copies.

The existing lock/manifest is reused on subsequent acquisition. Author scripts
run in separate copies with a recorded runtime patch. Source snapshots remain
byte-exact and can be checked without network access using `--verify`.

## Environment

Actual training: Linux A40, Python3.9, PyTorch2.5.1+cu121, NumPy/Pandas/
SciPy/scikit-learn/einops/matplotlib/seaborn/tqdm in the existing a40_env.
Per-run environment JSON and stdout are archived. All methods use float32.
The source downloader itself needs Python standard library and Git only.

The exact installed versions are recorded in the artifact directory at
`evidence/environment_packages.json`. FFVE also requires `imageio==2.34.2`.
The recorded run installed its SHA256-verified PyPI wheel with
`pip install --no-index --no-deps --target optional_dependencies <wheel>`
from the repository root, leaving the shared environment unchanged. The
author FFVE launcher adds that directory to its own import path. Its wheel
hash and import location are recorded in `evidence/optional_dependencies.json`.
An equivalent clean environment may install that version normally.

Place official `train_FD001.txt` through `train_FD004.txt`, their test files
and RUL files in an external data directory. The actual12 source-file
fingerprints are archived. Unified training cannot open official test files
before its checkpoint freeze.

## One run

Use a **new** output root; completed/running run IDs are refused.

```bash
python -u -m paper_b.baselines.open_source.author \
  --model aidgn --subset FD001 --seed 101 \
  --data-dir /path/to/CMAPSS --out-root /path/to/new-results

python -u -m paper_b.baselines.open_source.unified \
  --model aidgn --subset FD001 --seed 101 \
  --data-dir /path/to/CMAPSS --out-root /path/to/new-results
```

Models: `aidgn`, `two_phase`, `dual_mixer`, `ffve`, `fcstgnn`. Sixth candidate
`cpt` is source-audited and its released pretraining contract is exercised by
`probe_candidate.py`; no regression-only substitute is used.

`dual_mixer` includes FSGRI. `two_phase` includes paired-subset pretraining,
gradient reversal, and a frozen transferred encoder. Both paired validation
sets are excluded under the unified protocol. FFVE keeps stochastic
reparameterization and KL+regression, with the author's per-subset block,
residual and learning-rate settings. FC-STGNN keeps per-subset patch/graph/CNN
dimensions.

## Finite batch and checks

```bash
python -m paper_b.baselines.open_source.verify_runtime \
  --data-dir /path/to/CMAPSS --out runtime_checks.json

python -u -m paper_b.baselines.open_source.matrix \
  --root /path/to/new-results --data-dir /path/to/CMAPSS --workers 4

python -m paper_b.baselines.open_source.monitor --root /path/to/new-results
```

Four stages: priority author; priority unified; FFVE/FC-STGNN author; their
unified runs. Four subsets and seeds101/202/303 give120 planned trainings.
O_EXCL `RUNNING.json` claims implement run-ID idempotency. Completion/failure
records and separate attempts prevent overwriting old evidence. This is a
finite batch for the current request, not a periodic automation.

For this recorded run, `v1` retains the initial AIDGN optional-import failure
and Two-Phase/Dual-Mixer results; `v2` contains the isolated AIDGN retry;
`v3` contains unified results and the first FFVE/FC-STGNN author launch
failures; `v4` contains their repaired author runs. A fresh matrix creates
these namespaces itself; it never overwrites a failed run. FC-STGNN's graph
preprocessing container needs a concrete `__iter__` to instantiate on the
recorded PyTorch version: the runtime-only stub raises if iterated. The
author entrypoint reads its original arrays directly. Model files and array
preprocessing remain byte/AST-identical, with the exact patch archived.

## Author export corrections and reporting

```bash
python -m paper_b.baselines.open_source.gather \
  --remote-root /path/to/new-results --data-dir /path/to/CMAPSS \
  --target paper_b/experiments/open_source_reproduction_20261111

python -m paper_b.baselines.open_source.report \
  --artifacts paper_b/experiments/open_source_reproduction_20261111 \
  --source-audit docs/paper_b_baselines/SOURCE_AUDIT.json \
  --report-dir docs/paper_b_baselines
```

`gather` expects runtime checks and candidate evidence in the parent of the
batch root, as in the recorded deployment. Run the candidate probe explicitly:

```bash
python -m paper_b.baselines.open_source.probe_candidate \
  --data-dir /path/to/CMAPSS --out /path/to/candidate_execution.json
```

Early Two-Phase row indexes were not engine IDs. Reconciliation re-infers
from the same trained checkpoint and requires prediction/truth multiset
agreement. Early Dual exports incorrectly used a125 scale; raw normalized
arrays survive, and corrected `126*x-1` labels are checked against every
unit/cycle/RUL. `RECONCILED_metrics.json` supersedes initial exports without
deleting them. The author Dual loader omits short test engines; retained-engine
endpoints are not promoted to complete official-test results.

Final metrics are independently recomputed with `math.fsum`/`math.expm1`;
config/checkpoint hashes, finite weights, engine counts and official labels
are checked. Unified runs also have freeze and test-access audit records.
Published/author/unified tables stay separate. Missing sources and negative
gaps are visible; test results never change registered configurations.

Protocol, source audits and recovery state: `docs/paper_b_baselines/`.
Main deliverable: `docs/paper_b_baselines/OPEN_SOURCE_REPRODUCTION_REPORT.md`.

After generation or transfer, run the GPU-free final acceptance gate:

```bash
python paper_b/baselines/open_source/verify_artifacts.py \
  --root paper_b/experiments/open_source_reproduction_20261111 \
  --require-complete
```

This requires exactly120 unique registered cells and independently binds
report rows, summary mean/SD/95% CI, prediction metrics and artifact hashes.
Omit the flag only for explicitly incomplete progress snapshots.
