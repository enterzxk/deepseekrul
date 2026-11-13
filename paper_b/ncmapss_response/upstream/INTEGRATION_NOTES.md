# Pinned author source integration notes

This is audit metadata, not an alteration of author code. Original files are read-only and individually verified against fixed-commit Git blob SHA-1 and SHA-256. No author module was imported or executed and no environment, training, data or TEST inference was run.

## Acquisition and verification

From the worktree root:
- `python paper_b/ncmapss_response/upstream/ACQUIRE_SOURCES.py`
- `python paper_b/ncmapss_response/upstream/VERIFY_SOURCES.py`

The acquisition program uses the cached complete Git tree in PINNED_TREES.json, immutable raw GitHub URLs, explicit per-file inclusion/exclusion rules, path guards and original binary bytes. It never strips notebooks or rewrites original source. The verification result is PRESERVATION_ACCEPTANCE.json.

| Snapshot | Commit | Included | Excluded |
|---|---|---:|---:|
| bayesrul | 2943f4d85a1bb71910c100c956b32c3f6b595ceb | 120 | 0 |
| aidgn | d1ac173b13659af1854cf500a9b1cba8367353fe | 90 | 1199 |
| moo_elm | b19b2bf47bdfc3b7d4ae2a540b605caf4fd087e3 | 51 | 37 |
| concept | 25ec477c834e81440a288e161ec1f4461737ee49 | 48 | 0 |

Each SOURCE_MANIFEST.json inventories every upstream blob as included or excluded, with exact links and reasons. Author data, checkpoints, stored predictions, bytecode, IDE settings and non-source figures are excluded. The BayesRUL notebook is retained verbatim, not executed.

## BayesRUL complete Inception-HNN

- Source: `bayesrul/bayesrul/models/nets/inception.py:142`; class `Inception(win_length, n_features, activation="relu", bias="True", dropout=0, out_size=2)`.
- Original assertion requires 18 features. Forward input is [batch,time,18]; both output channels use softplus and a 1e-9 threshold. Keep the full Inception branches, reduction module and head.
- HNN source: `bayesrul/bayesrul/models/frequentist.py:9`; `HNN(net, optimizer, mc_samples, p_dropout)`.
- Original loss: Gaussian NLL of output loc and variance scale squared, not MSE or a replacement uncertainty head. Author initialization via weights_init must be retained.
- Source training configuration: `bayesrul/bayesrul/conf/experiment/ncmapss_hnn.yaml`: Adam lr0.001574, weight_decay0.001, batch250, max500, patience50, NLL validation selection. All original Hydra configs and Poetry lock are preserved.
- Importing the Lightning wrapper also requires its results/metrics.py and utils/miscellaneous.py imports. Requirements are declared by pyproject.toml and poetry.lock; torchinfo is an eager dependency of the Inception module.
- Original data defaults mix DS01-05 and use hs-shaped labels and random window validation; a unified adapter must be explicitly reported separately.

## Complete AIDGN

- Source: `aidgn/Model/AIDGN.py:7`; `AIDGN(args, supports=None)`.
- Constructor args referenced directly: AATE_dim, fc_dropout, fc_layer_dim, feature_fc_layer_dim, feature_num, hidden_dim, hop, nlayer, outlayer, reshape, sequence_len. Retained original configs and entry scripts supply author values.
- Forward: `forward(x, OCC=None)`; x is [batch,sequence_len,feature_num], OCC is required in practice despite the default None (immediately transposed), and return is `(None, rul_prediction)`. Keep aging-conditioned graph construction, graph convolutions, residuals, temporal aggregation and channel decoder.
- The mathematical core imports only NumPy and PyTorch; it is self-contained in AIDGN.py.
- Original `Model/__init__.py` imports missing Model.AIGCN. Do not repair the snapshot. An external adapter may load the exact AIDGN.py directly with importlib to avoid unrelated package imports, documenting that integration change.
- Original NC preprocessing includes two virtual sensors and a cap derived before dev/test separation. Original code is retained; unified train-only preprocessing is a distinct protocol.

## Complete MOO_ELM search

- Entrypoint `moo_elm/enas_elm_moo.py`; author README command `python3 enas_elm_moo.py -w 1 -s 1 --pop 50 --gen 50`.
- Preserve NSGA-II search in utils/ea_multi.py, task fitness in utils/elm_task.py, model construction in utils/elm_network.py and vendored utils/hpelm. A fixed arbitrary ELM or tiny replacement is not this method.
- Direct search dependencies include TensorFlow, NumPy/pandas/scikit-learn/SciPy, matplotlib/seaborn, pathos and DEAP. Vendored hpelm also imports six and PyTables; GPU backend requirements depend on selected accelerator. requirements.txt is incomplete for the observed imports and pins old NumPy/scikit-learn/TensorFlow-GPU.
- C/C++ hypervolume source and headers are retained as author source. They contain separate GPL/LGPL notices; root Apache-2.0 does not replace those notices.
- Original preprocessing includes T40/P30 virtual sensors, random-window validation and test-window pooled metrics. Unified input/split/scoring adaptation must be named.

## Redistribution boundary

BayesRUL and AIDGN have no explicit repository-wide license file or pyproject grant in these pinned trees. AIDGN N_CMAPSS_train.py has a file-specific MIT statement. Concept has a MIT classifier in cem/setup.py, but this does not establish a repository-wide license for every file. These three snapshots are local read-only sources and Git-ignored; manifests remain trackable. Do not force-add their sources.

MOO_ELM has a root Apache-2.0 LICENSE plus retained file-specific GPL/LGPL notices. This audit records licensing evidence; it does not grant redistribution rights or certify mixed-license compatibility. No commits, pushes or source publication were performed.

