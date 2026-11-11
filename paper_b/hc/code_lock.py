"""Committed-code lock for hosts without git (a40).

The 2026-11-11 ruling forbids launching a run from an uncommitted tree. a40 has
no git, so this tool records the md5 of every tracked code file *as committed*
(``git show HEAD:<path>`` -- the exact blob bytes, LF, no working-tree CRLF) and
``run_hc`` refuses to start unless every file on the host matches that lock.

Usage::

    python -m paper_b.hc.code_lock --write     # regenerate after a commit
    python -m paper_b.hc.code_lock --check     # verify this host against it
"""
import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

from .run_hc import FROZEN, OWN, REPO

DEFAULT = REPO / 'docs' / 'paper_b_hc' / 'registry' / 'code_lock.json'


def _blob(rel):
    try:
        return subprocess.check_output(['git', 'show', f'HEAD:{rel}'], cwd=REPO)
    except subprocess.CalledProcessError:
        raise SystemExit(f'REFUSED: {rel} is not in HEAD; commit it before generating the lock')


def build():
    code_md5 = {rel: hashlib.md5(_blob(rel)).hexdigest() for rel in FROZEN + OWN}
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip()
    return dict(schema_version=1, git_sha=sha,
                generated_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                definition='md5 of `git show HEAD:<path>` blob bytes',
                code_md5=code_md5)


def check(lock_path=DEFAULT):
    lock = json.loads(Path(lock_path).read_text())
    got = {rel: hashlib.md5((REPO / rel).read_bytes()).hexdigest()
           for rel in lock['code_md5']}
    return {k: (got[k], v) for k, v in lock['code_md5'].items() if got[k] != v}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--write', action='store_true', help='regenerate from HEAD')
    g.add_argument('--check', action='store_true', help='verify this host')
    ap.add_argument('--out', default=str(DEFAULT))
    a = ap.parse_args()
    if a.write:
        lock = build()
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(lock, indent=1) + '\n')
        print('wrote', a.out, 'git_sha', lock['git_sha'], 'files', len(lock['code_md5']))
        return 0
    bad = check(a.out)
    print('CODE LOCK OK' if not bad else json.dumps(bad, indent=1))
    return 0 if not bad else 1


if __name__ == '__main__':
    raise SystemExit(main())
