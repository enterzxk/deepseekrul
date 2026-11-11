# Legacy Paper B snapshot (pre-cleanup, 2026-11-11)

This directory preserves every Paper B artefact that existed before the
2026-11-11 takeover. Nothing here is paper evidence unless explicitly marked
otherwise in `../../reports/ARCHIVE_INDEX.md`.

## Why it is archived

1. Commit `95e729e` claims to fix P0 reproducibility bugs, but `git show --stat`
   proves it changed **only** `docs/paper_b_code/P0_FIXES.md`. The code fixes
   existed only in the working tree of the lab3090 server (`/home/user/rulb`).
2. The repository copy and the server copy therefore diverged:

   | file | repository md5 | server md5 |
   |---|---|---|
   | `run_dspt.py` | `05bd73df14e44e9e9b9249130a82af17` | `fee0d9a83d4a2b05b989fa1a2628413c` |
   | `dspt.py` | `634eb8db4a3f093ee572d623499dc3d4` | `ade5c091e1aad746b7444309ec77b9ef` |
   | `dspt_vanilla.py` | `fcc80336eb6ef4925d6759fc104af8aa` | same |

3. The repository `run_dspt.py` cannot run: `windows_for` calls
   `legacy=legacy_rul` although the signature has no `legacy_rul` parameter
   (NameError), and `ev()` passes `cycle` unconditionally.
4. The official-test evaluator in this snapshot (`eval_dspt.py`) hard-codes
   `MAX_SPAN=100` and cannot load the archived 32-cell checkpoints, whose
   `temporal_pe` is `[1,70,64]`:

   ```
   RuntimeError: size mismatch for temporal_pe: copying a shape [1, 70, 64]
   from checkpoint, the shape in current model is [1, 100, 64].
   ```

   The archived test CSV (`results/ABLTEST/FD001_base.csv`, mtime 12:20) was
   therefore produced by a different, uncommitted evaluator than the one saved
   here (checkpoint mtime 00:39). Its exact protocol cannot be reconstructed.

## Server snapshot

`run_dspt.py`, `dspt.py`, `eval_dspt.py`, `eval_official.py` are the files
copied from `lab3090:/home/user/rulb` at takeover time. They contain the
working-tree P0 fixes but are still not in git history.

## Restore

The immutable git tag `archive/paper-b-pre-cleanup-95e729e` points at
`95e729e`. To restore any file from before the move:

```bash
git checkout archive/paper-b-pre-cleanup-95e729e -- docs/paper_b_code/run_dspt.py
```

The server working tree is preserved at `lab3090:/home/user/rulb` and was not
modified by the cleanup.
