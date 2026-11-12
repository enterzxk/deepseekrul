# N-CMAPSS DS02 reproducibility audit

Status: **path chosen, nothing executed yet**.

## Pinned facts

| item | value |
|---|---|
| base commit (branch point) | `ac94c29` "v11 code lock" |
| working branch | `ncmapss-ds02` |
| dataset archive | outer zip 15,760,443,389 bytes, SHA256 `d1271732…`, **full CRC pass** |
| dataset file | `N-CMAPSS_DS02-006.h5`, 2,450,472,504 bytes, SHA256 `47971a68…`, mode `444` |
| data location (never committed) | `/home/user/deepseek-rul/data/NCMAPSS/` |
| experiment output (never committed) | `/home/user/deepseek-rul/experiments/ncmapss/` |
| compute host | 3090 box, 2×RTX3090, 64 cores |

**The archive is nested.** The outer zip contains a second `data_set.zip`, and the
DS02 HDF5 file lives inside it. A single-level `unzip` will not find the file and
will look like a corrupt or incomplete download. Unpack in two steps. NASA
publishes no checksum for this archive, so the integrity evidence is the ZIP CRC
pass plus our own SHA256 — see "Open items".

Measured dataset shape (from `experiments/ncmapss/ds02_audit.json`, before any
run): dev units `[2,5,10,16,18,20]` hold **446 flights** in total, test units
`[11,14,15]` are the official held-out set. Longest lifetime is **89 flights**
(dev unit 5) and **76** (test), so no unit reaches the 125 cap. Unit 14 has only
1,912 rows per cycle (`Fc=1`) against 11k–13k for most others — flight-level
statistics from that unit rest on far fewer 1 Hz samples, which is expected and
is exactly what the slope guard is for.

## Protocol freeze

The protocol is **frozen at commit `922a2d1`** (branch `ncmapss-ds02`,
2026-11-12). **No §4 or §6 change after this commit until the LOEO development
results are in.** A change requires a recorded decision first — never edit while
runs are in flight, because an unrecorded edit leaves runs produced before and
after it in different protocols with no marker.

Verify the frozen document:

```bash
git rev-parse 922a2d1:docs/paper_b_ncmapss/DATASET_PROTOCOL.md
# ebfbfcb8d41c7f4dd0bfc3d0c35a50225b57d7d1
```

Every row in `registry/runs.csv` must cite **this blob hash**, not only the
commit: a commit can be amended or rebased, a blob hash cannot. A run whose
`protocol_blob` does not match is treated as produced under an unknown protocol
and leaves the main evidence table.

### Anchor history

| anchor | blob | status | why it moved |
|---|---|---|---|
| `5b228ad` | `6c184bbf…` | void | initial freeze |
| `7f01909` | `4458d9ea…` | void | §6.3 named the primary metric; §6.4 fixed the relative-segment denominator bug; HI scope narrowed |
| **`922a2d1`** | **`ebfbfcb8…`** | **current** | two settings decided from the measured file, before any run |

The last move needs spelling out, because it is the one that could be mistaken
for tuning after seeing results. It was not: both numbers come from the file's
row counts, not from any model output, and no run existed when they were set.

1. **The 125 cap is inactive on this dataset.** The measured longest lifetime is
   89 flights (dev unit 5) and 76 (test), so `min(raw, 125) == raw` everywhere.
   The capped criterion carries no information here and is reported as
   `inactive (no unit exceeds the cap)` — never as a stricter criterion, and
   never as a second column printed beside raw.
2. **40 epochs was 40 gradient steps.** The dev split holds 446 flights, so a
   fold trains on ~380 windows; with `batch = min(512, 380) = 380` each epoch is
   a single step. The budget was restated as **1000 optimizer steps**
   (`epochs = ceil(1000 / ceil(N_train / batch))`), identical for all four rungs.

The earlier move from `5b228ad` had three reasons, still worth keeping in the
record: §6.3 previously did not say which metric was primary; §6.4 normalised the
relative segment by the evaluated frame, which on the one-flight-per-engine test
split makes the fraction identically 1.0 and silently labels every row `early`;
and the HI scope was narrowed (`e7c2e41`, `026320c`) because the frozen N0–N3
ladder contains no HI arm and the earlier text pre-registered a comparison
against an arm that does not exist.

Note for anyone verifying on Windows: **blob hashes are sensitive to line
endings**, and this repository pins `eol=lf` in `.gitattributes`. Always verify
with `git rev-parse` **inside the repository**; never hash the working-tree file
with `sha256sum` and compare, because a CRLF re-save changes the blob while the
text looks identical.

## Commit path: **(a) patch from the compute host, applied and pushed from Windows**

Chosen over the alternative (pushing directly from the 3090 host) for three
reasons:

1. **No credentials on the compute host.** The 3090 box has git but no confirmed
   push credential, and this repository is public. Keeping the only push
   credential on the Windows workstation means a compromised or shared compute
   node cannot write to the repository.
2. **The artifacts are text.** Checkpoints (`*.pt`) and the 15.8 GB archive are
   gitignored, so everything that needs to travel — configs, metrics JSON, CSVs,
   logs — is diffable text and survives `git format-patch` / `git apply`
   unchanged.
3. **One direction only.** Code flows Windows → 3090 by clone/fetch; results
   flow 3090 → Windows by patch. Neither side ever writes the other's branch, so
   the "two writers" failure mode that bit the C-MAPSS ledger cannot recur here.

### Procedure

On the 3090 host, after a batch of runs:

```bash
cd /home/user/deepseek-rul
git add -A experiments/ncmapss docs/paper_b_ncmapss paper_b/ncmapss
git commit -m "<batch description>"
git format-patch -1 --stdout > /tmp/batch_N.patch
# copy /tmp/batch_N.patch to Windows, then:
```

On Windows, in the `ncmapss-ds02` worktree:

```bash
git am < batch_N.patch
git push origin ncmapss-ds02
```

If `git am` conflicts, the patch is **rejected and re-cut** on the host — it is
never hand-merged, because a hand-merged result is not the one that produced the
numbers.

### Worktree topology

Three places have held a copy of this branch and they have already drifted twice.
The rule, effective now:

- **`origin/ncmapss-ds02` is the single source of truth.** No local copy is
  authoritative, including the Windows worktree.
- **The 3090 host is a read-only consumer.** Before each batch:
  `git fetch && git reset --hard origin/ncmapss-ds02`. No commits are made on the
  compute host.
- **All code changes travel as patches**: author → patch → applied in the Windows
  worktree → pushed. A change that exists only on the compute host is not part of
  the project, no matter how good it is.
- **No long-lived untracked files on the compute host.** A file that matters goes
  into a patch; one that does not gets deleted. `features.py`, `train.py` and
  `env/` were the last offenders and are now committed.

Why this is a rule and not advice: twice already two writers touched this branch
and "which copy is real" had to be settled by hand. The failure mode is not lost
work — it is a run recorded against code nobody can reconstruct.

## Prediction-table contract (enforced in code, deliberately not restated in §6)

The frozen protocol §6 states that `evaluate()` requires `unit_lifetime`, but it
does **not** restate the prediction-table column contract. That contract is:

```
unit, cycle, y_true_raw, y_true_cap125, y_pred
```

- exactly these five columns, one row per `(unit, cycle)`;
- **no bare `y_true` column** — that name would silently mean the capped label,
  which is the same implicit-default trap that produced the C-MAPSS criterion
  confusion;
- `y_true_cap125 == min(y_true_raw, 125)` must hold, and `evaluate()` asserts it
  before computing any score.

**It is deliberately not added to §6 right now.** Any edit to §4/§6 moves the
protocol blob and voids the freeze anchor. The anchor has just moved once
(`922a2d1` / `ebfbfcb8…`) for reasons taken from the measured file, and it will
not move again until the first LOEO batch lands — a second move for a prose
clarification would make "which anchor is real" a question no reader can answer
from the document alone.

The contract is enforced in two places that cannot drift: the code (hard
rejection at the entry point) and `registry/SCHEMA.md` v5 (column 28). Once the
first LOEO batch lands, it gets folded into §6 as a recorded decision, with a new
anchor and an explicit note that the old one is void.

## Environment pinning

- Python environment and package versions must be recorded per batch
  (`pip freeze` output committed under `docs/paper_b_ncmapss/env/`).
- Every metrics file records the git SHA of the code that produced it. A metrics
  file without a SHA is treated as unattributed and cannot enter a results table.
- The dataset archive hash is recorded once, before any run, and re-checked when
  the extracted tables are rebuilt.

## What must never be committed

- `data/NCMAPSS/` (15.8 GB), `experiments/ncmapss/` (checkpoints and raw outputs)
- any `*.h5`, `*.pt`, `*.pth`, `*.ckpt`
- anything derived from the official test units before the freeze manifest is
  committed

## Open items

- [x] Record the **self-computed** SHA256 of the DS02 archive and a passing
      `unzip -t` (full ZIP CRC verification): outer zip `d1271732…`, CRC pass,
      15,760,443,389 bytes; inner HDF5 `47971a68…`, 2,450,472,504 bytes, mode
      `444`. **NASA publishes no official checksum**, so there is nothing to
      compare against — the audit must not claim "matches the official hash".
- [x] Record the Python environment path on the 3090 host
      (`/home/user/miniconda3/envs/ncmapss/bin/python`, Python 3.10.22,
      torch 2.5.1+cu121, same version set as the A40 environment).
- [ ] Record the first patch round-trip (patch id, `git am` result) as proof the
      path works end to end.
