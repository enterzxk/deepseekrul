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

The protocol is **frozen at commit `9ac1744`** (branch `ncmapss-ds02`,
2026-11-12). **No §4 or §6 change after this commit until the LOEO development
results are in.** A change requires a recorded decision first — never edit while
runs are in flight, because an unrecorded edit leaves runs produced before and
after it in different protocols with no marker.

Verify the frozen document:

```bash
git rev-parse 9ac1744:docs/paper_b_ncmapss/DATASET_PROTOCOL.md
# a3c40c634da7a3bb44ace788f74896b17e51c3fb
```

Every row in `registry/runs.csv` must cite **this blob hash**, not only the
commit: a commit can be amended or rebased, a blob hash cannot. A run whose
`protocol_blob` does not match is treated as produced under an unknown protocol
and leaves the main evidence table.

### Anchor history

| anchor | blob | status | why it moved |
|---|---|---|---|
| `c481289` | `2e4df8dd…` | void | branch skeleton: protocol discipline, baseline ladder, commit path |
| `4bfa5d8` | `386b13a5…` | void | DS02 data dictionary from the official tables + observability boundary |
| `7cc5824` | `1fa8ec4c…` | void | evaluation protocol §6.1–6.9: unit of analysis, LOEO discipline, dual segment rules, criterion hygiene, test gate |
| `e1de832` | `2e3f36b0…` | void | froze the 51-column flight feature contract and the `span=70` window (information matching across N1/N2/N3) |
| `c69db46` | `668c98a9…` | void | froze the training plan (`protocol.json` verbatim) and the k=6 W-cluster conditioning (no information advantage for the DSPT arms) |
| `d9e49af` | `dc347037…` | void | split schedule from label construction: `rul_cap`/`val_cuts` not inherited, no early stopping on the held-out engine |
| `5b228ad` | `6c184bbf…` | void | **first freeze** — bounded the "same core structure" claim (stopping rule differs by design, patience unused) |
| `7f01909` | `4458d9ea…` | void | §6.3 named the primary metric; §6.4 fixed the relative-segment denominator bug; HI scope narrowed |
| `922a2d1` | `ebfbfcb8…` | void | the 125 cap measured inactive; the epoch budget was restated as optimizer steps |
| `e7ad15f` | `35b40d8b…` | void | the budget's alignment quantity was wrong: 1000 steps on 380 windows is 1000 passes, 25x the C-MAPSS budget |
| `1c76b37` | `443462ca…` | void | fold-local target standardisation and a 200-pass budget: 40 passes left every rung undertrained |
| `7e9415e` | `21af549e…` | void | a fixed budget was replaced by inner-validation stopping |
| **`9ac1744`** | **`a3c40c63…`** | **current** | the median of per-split minima collapsed to 1–2 passes, so stopping moved to the argmin of the mean inner curve, floored at 20 |

**Correction (2026-11-12):** an earlier version of this table listed only five
moves, and that was wrong. `git log --follow -- docs/paper_b_ncmapss/DATASET_PROTOCOL.md`
returns **13 commits with 13 distinct blobs**. The five-row version silently
omitted the six drafting generations and `1c76b37` — the generation that
introduced fold-local target standardisation, i.e. exactly the kind of change
someone would later want to trace a run back to. A five-row table cannot answer
"which protocol did this run use", so the table is rebuilt in full above.

**These twelve moves are not equally defensible, and the record should say so.**
Six were drafting, taken before the text was frozen at all: nothing could be
invalidated by them. Three were setup moves made before the first grid run
(`7f01909`, `922a2d1`, `e7ad15f`), taken from the measured file with no run in
existence — unambiguous. The last three came **after diagnostic runs**
(`1c76b37` after the probe showing 40 passes left every rung undertrained,
`7e9415e` and `9ac1744` after smoke runs exposed the stopping rule failing),
which is a weaker position: they are setup decisions made in the presence of
diagnostic runs, not in their absence.

What makes them defensible is the shape of the failure. The median-of-minima
rule collapsed `epoch*` to 1–2 passes and produced visibly untrained models
(N2 went from 5.29 to 18.09); that is a broken instrument, not an unfavourable
result. And the response was not to pick the better number: **both evaluation
points are now reported side by side** (`inner_val` and `fixed_passes:200`), so
the choice is visible instead of hidden.

**The distinction that has to be written down, because it is the only line
between fixing an instrument and picking a result:** all three late moves were
*triggered* by numbers that looked wrong, but they were *justified* by a failure
of measurement — 40 passes leaving N0 at 15.23 against a 5.19 ridge on the same
features (the training loop could not reach the scale of its own labels),
`epoch*` collapsing to 1–2 passes, N2 dropping from 5.29 to 18.09. A trigger is "the number was not what we expected"; a criterion is
"the instrument could not measure". Only the second licenses a change. Any future
change whose only justification is that the numbers were disappointing crosses
the line, and after the full batch starts every change does.

What would not be defensible is another move after the full batch starts. At that
point a change stops being a repair and becomes a selection.

The narrative detail of the two stopping-rule moves is worth keeping, because
both were overturned by data rather than by preference:

- **Fixed budget → inner-validation stopping**: the four rungs converged at
  rates differing by up to 20x (`holdout_min`), so one fixed budget could not fit
  all of them.
- **Median of per-split minima → argmin of the mean curve**: with four training
  engines and one inner validation engine, the validation loss bottoms out at
  passes 1–12 and then rises as the model memorises the four. Per-split minima
  are therefore not an estimable stopping point; averaging the five curves and
  flooring at 20 is more stable, and the floor application is recorded explicitly
  rather than applied silently.

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

## Open defect: N1 may be missing an input path the DSPT rungs have

N1 (vanilla) scores **21.6–23.1** at `epoch*` = 20 and at 94, and **18–22** at a
fixed 200 passes. Its error barely responds to the training budget, which points
at the model rather than at the schedule.

The DSPT rungs consume a `cond` embedding (k-means over `W`, fitted on the fold's
training engines only) and a `dual_span` branch. N1 consumes neither. If N1 is
simply missing an input path that the DSPT rungs have, then the N1-vs-DSPT
contrast compares **different information**, not different architectures — the
same failure mode that invalidated the C-MAPSS cross-paper comparisons in
D041/D043.

**Until this is resolved, no N1 number enters the main results table.** The check
is not what the config says: it is to enumerate, at runtime, the input tensors
each rung actually consumes, and diff the two lists.

- If it is an implementation gap → N1's numbers are void and must be rerun.
- If it is a genuine architectural difference → it is a writable contrast, and
  the information-matching statement must say so explicitly rather than implying
  the rungs saw the same inputs.

**No silent overwrite.** If the gap is confirmed, the pre-fix N1 numbers are
recorded as void with the reason stated (which input path was missing, and what
was compared against what), not deleted and not quietly replaced by the rerun.
The rerun is a new row; the old rows keep their `void` marker. A reader must be
able to see that a contrast existed, why it was withdrawn, and what replaced it.

## Environment pinning

- Python environment and package versions must be recorded per batch
  (`pip freeze` output committed under `docs/paper_b_ncmapss/env/`).
- Every metrics file records the git SHA of the code that produced it. A metrics
  file without a SHA is treated as unattributed and cannot enter a results table.
- The dataset archive hash is recorded once, before any run, and re-checked when
  the extracted tables are rebuilt.

## Cross-run reproduction: one fit, two snapshots, two historical runs

The first cells of the full batch reproduce two independent historical runs
**exactly**, from a single training run each:

| cell | fixed-budget run (historical) | this run, `fixed_passes:200` | v4 smoke (historical) | this run, `inner_val` |
|---|---|---|---|---|
| N1 s101 | 18.05 | **18.05** | 21.58 | **21.58** |
| N1 s202 | 22.21 | **22.21** | 21.75 | **21.75** |
| N1 s303 | 21.41 | **21.41** | 23.11 | **23.11** |
| N2 s101 | 8.01 | **8.01** | 8.20 | **8.20** |

Two things follow, and both are load-bearing.

1. **No environment drift between the two historical runs.** Bit-level agreement
   across four cells is not something a change in torch, cuDNN, the HDF5 tables
   or the feature construction survives. The two earlier runs were made under
   different code, days apart, on the same host; they agree to the last printed
   digit.
2. **The two-point implementation introduces no numerical difference of its own.**
   Both snapshots come from the same weights of the same fit, so the gap between
   the two columns is **where training stopped**, not retraining variance.

This closes the obvious reviewer question — "are the two columns two independent
trainings, so the difference could be initialisation?" — with a measurement
rather than an assurance: one fit, two snapshots.

Scope, stated so the claim is not overread: this is 4 of 60 cells, N1/N2 only,
one host, one dataset. It is evidence of stability, not a proof that the batch is
reproducible end to end. The end-to-end check is the paired-column comparison
once all 120 rows exist.

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
