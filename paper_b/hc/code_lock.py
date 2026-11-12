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


def _blob(rel, from_index=False):
    ref = ':' + rel if from_index else 'HEAD:' + rel
    try:
        return subprocess.check_output(['git', 'cat-file', 'blob', ref], cwd=REPO)
    except subprocess.CalledProcessError:
        raise SystemExit(f'REFUSED: {rel} is not in the index/HEAD; add and commit it first')


def build(baseline_version='unregistered', from_index=False):
    code_md5 = {rel: hashlib.md5(_blob(rel, from_index)).hexdigest() for rel in FROZEN + OWN}
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip()
    except subprocess.CalledProcessError:
        sha = None
    return dict(schema_version=1, git_sha=(None if from_index else sha),
                baseline_version=baseline_version,
                source=('staged index (the commit that contains this file)' if from_index
                        else 'HEAD'),
                generated_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                definition='md5 of `git cat-file blob <ref>:<path>` bytes',
                code_md5=code_md5)


def check(lock_path=DEFAULT):
    """Verify this host against the lock.

    Byte comparison is CRLF-normalised: the repo stores LF blobs, a Windows
    working tree with ``core.autocrlf=true`` has CRLF, and a40 has LF again.
    Comparing raw bytes therefore reports two false mismatches on Windows (v9
    fix; the md5 in the lock always refers to the LF blob).

    A lock with no ``git_sha`` also fails. A ``--from-index`` lock cannot name
    the commit that contains it, so it records ``null``; the md5s are still
    correct, but ``md5(blob@git_sha)`` can no longer be replayed and the run's
    ``git_sha`` column is empty -- the same defect that made 24 cal rows
    unreplayable. Reporting CODE LOCK OK on such a lock would let the hole pass
    silently, which is why it is a failure here and not a warning.
    """
    lock = json.loads(Path(lock_path).read_text())
    if lock.get('git_sha') in (None, ''):
        raise SystemExit(
            'REFUSED: %s has no git_sha (built with --from-index). Regenerate it '
            'in HEAD mode as a separate commit: a lock cannot name the commit '
            'that contains it.' % lock_path)
    if 'index' in str(lock.get('source', '')):
        raise SystemExit('REFUSED: %s was built from the staged index (%s); '
                         'regenerate from HEAD' % (lock_path, lock.get('source')))
    got = {rel: hashlib.md5((REPO / rel).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
           for rel in lock['code_md5']}
    return {k: (got[k], v) for k, v in lock['code_md5'].items() if got[k] != v}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--write', action='store_true', help='regenerate from HEAD')
    g.add_argument('--check', action='store_true', help='verify this host')
    ap.add_argument('--out', default=str(DEFAULT))
    ap.add_argument('--version', default='unregistered',
                    help='registered baseline label written into the lock (e.g. v7)')
    ap.add_argument('--from-index', action='store_true',
                    help='hash the staged index so code and lock can be one commit')
    a = ap.parse_args()
    if a.write:
        lock = build(a.version, a.from_index)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(lock, indent=1) + '\n')
        print('wrote', a.out, 'git_sha', lock['git_sha'], 'files', len(lock['code_md5']))
        return 0
    bad = check(a.out)
    print('CODE LOCK OK' if not bad else json.dumps(bad, indent=1))
    return 0 if not bad else 1


if __name__ == '__main__':
    raise SystemExit(main())
