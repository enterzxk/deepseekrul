# Exact historical execution-source recovery

These are our own historical launchers, retained as execution evidence.
Use the current top-level author module for new runs. Upstream code is not
bundled here. Each filename includes the exact SHA256 stored in historical
config files.

- 320a6c8f... is the exact Git blob from c50d2e0 (also7321d40/a64574b).
- dbd204e1... was reconstructed from the initial deployment and that Git
  blob using SequenceMatcher byte-line opcodes (autojunk=False). Of16
  non-equal opcodes, target edits0–12 and15 were selected (mask40959).
  Exact SHA256 equality to the original recorded runner hash is required;
  candidate text was never executed. An independent reviewer found this
  match using an in-memory exhaustive search.

Missing exact historical versions, if any, are explicitly indexed in the
artifact execution provenance audit. Original configs are never rewritten.
