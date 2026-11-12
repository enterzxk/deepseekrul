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
