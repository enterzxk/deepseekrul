"""Which commits produced the three B0 baseline fingerprints, and did B0's path move?

The sprint's "B0 FD001 80ep three seeds" were trained on three different code
snapshots (files_sha256 c67df063 / e89b23b0 / 6f8f8423). The baseline is only
usable if B0's numeric path -- model construction, the B0 forward branch, the
loss, the optimiser and the evaluator -- is line-for-line unchanged across them.

This resolves each fingerprint to a commit and then lists every hunk that could
touch B0, so the answer is a diff rather than an inference.
"""
import hashlib
import subprocess

FILES = ('arrival.py', 'hi_d.py', 'data_d.py', 'train_d.py', 'config_hc.py')
TARGETS = {'c67df063': None, 'e89b23b0': None, '6f8f8423': None}
PATHS = ['paper_b/hc/' + f for f in FILES]


def blob(sha, path):
    try:
        return subprocess.check_output(['git', 'cat-file', 'blob', '%s:%s' % (sha, path)],
                                     stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return b''


def fp(sha):
    h = hashlib.sha256()
    for f in FILES:
        h.update(f.encode())
        h.update(blob(sha, 'paper_b/hc/' + f))
    return h.hexdigest()[:32]


commits = subprocess.check_output(
    ['git', 'log', '--format=%H', '--'] + PATHS).decode().split()
print('scanning %d commits that touched the fingerprinted files' % len(commits))
for c in commits:
    if not blob(c, 'paper_b/hc/train_d.py'):
        continue          # train_d.py did not exist yet at this commit
    v = fp(c)
    print('  %s %s' % (c[:9], v))
    hit = [k for k in TARGETS if v.startswith(k)]
    if hit and TARGETS[hit[0]] is None:
        TARGETS[hit[0]] = c
        subj = subprocess.check_output(
            ['git', 'log', '-1', '--format=%s', c]).decode().strip()
        print('  %s -> %s  %s' % (v, c[:9], subj[:80]))

missing = [k for k, v in TARGETS.items() if v is None]
if missing:
    print('NOT FOUND: %s (the snapshot may be from an uncommitted tree)' % missing)

known = sorted([v for v in TARGETS.values() if v])
if len(known) >= 2:
    lo, hi = known[0], known[-1]
    print('\n=== diff %s..%s over the fingerprinted files ===' % (lo[:9], hi[:9]))
    d = subprocess.check_output(['git', 'diff', '-U1', lo, hi, '--'] + PATHS).decode()
    print(d if d.strip() else '(no differences)')
else:
    print('\nnot enough resolved snapshots to diff')
