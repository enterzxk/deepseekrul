# N-CMAPSS DS02 reproducibility audit

Status: **path chosen, nothing executed yet**.

## Pinned facts

| item | value |
|---|---|
| base commit (branch point) | `ac94c29` "v11 code lock" |
| working branch | `ncmapss-ds02` |
| dataset | N-CMAPSS DS02, official archive 15,760,443,389 bytes |
| data location (never committed) | `/home/user/deepseek-rul/data/NCMAPSS/` |
| experiment output (never committed) | `/home/user/deepseek-rul/experiments/ncmapss/` |
| compute host | 3090 box, 2×RTX3090, 64 cores |

## Protocol freeze

The protocol is **frozen at commit `7f01909`** (branch `ncmapss-ds02`,
2026-11-12). **No §4 or §6 change after this commit until the LOEO development
results are in.** A change requires a recorded decision first — never edit while
runs are in flight, because an unrecorded edit leaves runs produced before and
after it in different protocols with no marker.

Verify the frozen document:

```bash
git rev-parse 7f01909:docs/paper_b_ncmapss/DATASET_PROTOCOL.md
# 4458d9ea05c965251aabb782bba9765304ebf686
```

Every row in `registry/runs.csv` must cite **this blob hash**, not only the
commit: a commit can be amended or rebased, a blob hash cannot. A run whose
`protocol_blob` does not match is treated as produced under an unknown protocol
and leaves the main evidence table.

**The anchor moved forward from `5b228ad` / `6c184bbf…`, and the old anchor is
void.** Three reasons, none of them cosmetic:

1. **§6.3 now names the primary metric.** Per-engine mean RMSE is primary and
   pooled RMSE is supplementary. The previous text did not say which was which —
   a criterion ambiguity, not a content change.
2. **§6.4 fixes a real bug in the relative segment rule.** The denominator must
   come from the engine's full lifetime, supplied by the data layer. The previous
   wording normalised by the evaluated frame, which on the test split (one
   flight per engine) makes the fraction identically 1.0 and labels every row
   `early` — silently.
3. **The HI scope was narrowed** (`e7c2e41`, `026320c`). The frozen N0–N3 ladder
   contains no HI arm, and the earlier text pre-registered a comparison against
   an arm that does not exist.

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
protocol blob and voids the freeze anchor (`7f01909` / `4458d9ea…`), and the
compute host is about to be synchronised to exactly that anchor for the first
LOEO batch. Moving the anchor for a prose clarification while a sync is in
flight is precisely how a batch ends up invalid.

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

- [ ] Record the **self-computed** SHA256 of the DS02 archive and a passing
      `unzip -t` (full ZIP CRC verification) on the 3090 host.
      **NASA publishes no official checksum for this archive**, so there is
      nothing to compare against — the audit must not claim "matches the
      official hash". The two pieces of integrity evidence we can actually offer
      are (a) the ZIP CRC check covering every member, and (b) our own SHA256
      recorded before any run and re-checked when the extracted tables are
      rebuilt.
- [ ] Record the Python environment path on the 3090 host
      (`/home/user/miniconda3/envs/ncmapss/bin/python`, Python 3.10.22,
      torch 2.5.1+cu121, same version set as the A40 environment).
- [ ] Record the first patch round-trip (patch id, `git am` result) as proof the
      path works end to end.
