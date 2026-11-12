"""Generate code_lock.json on a host that has git but not torch.

``paper_b/hc/code_lock.py --write`` imports ``run_hc`` (for FROZEN/OWN/REPO),
which imports torch; the Windows working copy has git but no torch, and a40 has
torch but no git. This script closes that gap without duplicating the file list:
FROZEN and OWN are parsed straight out of ``run_hc.py`` with ``ast``, so there is
still exactly one source of truth for what the lock covers.

The definition is the same as ``code_lock.build``: md5 of
``git cat-file blob HEAD:<path>`` bytes (LF blob, never the working tree).

Usage (run from the repo root, after committing the code you want locked and
before committing the lock itself):

    python docs/paper_b_hc/probes/make_lock.py --version v10
"""
import argparse
import ast
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / 'paper_b/hc/run_hc.py'
OUT = REPO / 'docs/paper_b_hc/registry/code_lock.json'


def file_lists():
    tree = ast.parse(SRC.read_text(encoding='utf-8'))
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id in ('FROZEN', 'OWN'):
                found[tgt.id] = ast.literal_eval(node.value)
    for k in ('FROZEN', 'OWN'):
        if k not in found:
            raise SystemExit('could not parse %s out of %s' % (k, SRC))
    return found['FROZEN'], found['OWN']


def blob(rel, ref='HEAD'):
    return subprocess.check_output(['git', 'cat-file', 'blob', '%s:%s' % (ref, rel)],
                                   cwd=str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='unregistered')
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args()

    frozen, own = file_lists()
    files = frozen + own
    code_md5 = {rel: hashlib.md5(blob(rel)).hexdigest() for rel in files}
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO)).decode().strip()
    lock = dict(
        schema_version=1,
        git_sha=sha,
        baseline_version=a.version,
        source='HEAD',
        generated_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        definition='md5 of `git cat-file blob <ref>:<path>` bytes',
        code_md5=code_md5)
    Path(a.out).write_text(json.dumps(lock, indent=1) + '\n')
    print('wrote %s  version=%s  files=%d  git_sha=%s'
          % (a.out, a.version, len(files), sha))
    for rel in files:
        print('  %-44s %s' % (rel, code_md5[rel]))


if __name__ == '__main__':
    main()
