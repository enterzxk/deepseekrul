#!/usr/bin/env python3
"""Recompute the code fingerprint from git objects, not from the manifest.

Why a second implementation
---------------------------
`provenance.json` records what the manifest *declared*. This script reads the
8 files straight out of a commit (`git show <rev>:<path>`), so it neither
touches the working tree nor trusts any recorded field. Run it at every
`git_head` a batch used; if all of them give the same h8, the batch really did
run one code identity.

Usage
-----
    python scripts/audit_fingerprint.py <repo> <rev> [<rev> ...]
    python scripts/audit_fingerprint.py <repo> --worktree

`--worktree` hashes the files as they are on disk (for a local sanity check).
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

FILES = (
    'paper_b/models/dspt.py',
    'paper_b/training/train.py',
    'paper_b/training/data.py',
    'paper_b/training/metrics.py',
    'paper_b/sensor_aug/augmentation.py',
    'paper_b/sensor_aug/noise_stats.py',
    'paper_b/sensor_aug/evaluation.py',
)
RUNNER = 'scripts/run_sensor_aug.py'


def from_git(repo: str, rev: str, paths) -> str:
    h = hashlib.sha256()
    for rel in paths:
        h.update(subprocess.check_output(['git', '-C', repo, 'show', f'{rev}:{rel}']))
    return h.hexdigest()[:12]


def from_disk(repo: str, paths) -> str:
    h = hashlib.sha256()
    for rel in paths:
        h.update((Path(repo) / rel).read_bytes())
    return h.hexdigest()[:12]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    repo, revs = sys.argv[1], sys.argv[2:]
    if revs == ['--worktree']:
        h8 = from_disk(repo, FILES + (RUNNER,))
        h7 = from_disk(repo, FILES)
        print(f'worktree  H8={h8} H7={h7}')
        return 0
    for rev in revs:
        h8 = from_git(repo, rev, FILES + (RUNNER,))
        h7 = from_git(repo, rev, FILES)
        print(f'{rev[:8]}  H8={h8} H7={h7}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
