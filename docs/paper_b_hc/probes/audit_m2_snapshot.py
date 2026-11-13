"""Audit: is the M2 (CDE) arm's numerical path identical across the code
snapshots that produced its three seeds?

Background.  The manager's rule is that every seed of a final-table arm must sit
on one code snapshot, unless the diff between snapshots is provably off that
arm's numerical path.  Two candidate snapshots were named for M2: ``89947c2d``
(commit 66dd13e) for s101 and ``c010c9fb`` (commit 4ea5d5d) for s202/s303.
Separately, process start times show all three seeds actually imported
``a92ebaf1``, so this audit is the stronger, redundant argument: even across the
two named snapshots the arm's path does not move.

Checks
  A  ``cde.py`` -- the CDE module itself (evidence encoder, q_t, cde_head) --
     is byte-identical at both snapshot commits and at HEAD.
  B  every hunk of the ``train_d.py`` diff between the two snapshots lies inside
     ``term_grads()`` or its single call site: a post-optimiser diagnostic that
     only feeds ``history.json``.  It runs after ``sch.step()``, uses
     ``torch.autograd.grad(..., retain_graph=True)`` and never writes ``.grad``,
     a parameter, or the optimiser.
  C  ``SCR_ARMS`` does not contain M2, so ``scr=False`` on both sides and the
     new ``scr=``/``lam_a=`` arguments are inert for this arm.
  D  everything committed after the later snapshot touches only the fingerprint
     recorder, never the training path.

Exit code 0 only when all four pass.

Note on hashing.  This repo has ``core.autocrlf=true``: the working tree holds
CRLF while the git blob holds LF, so the same content has two different md5s
(``6e46d330`` in the tree vs ``c010c9fb`` in the blob).  Every hash below is
taken over the *blob*, which is what a Linux checkout materialises and what the
3090 records.

Usage:  python docs/paper_b_hc/probes/audit_m2_snapshot.py
"""
import hashlib
import re
import subprocess
import sys

TRAIN = 'paper_b/hc/train_d.py'
CDE = 'paper_b/hc/cde.py'
S101_MD5 = '89947c2d'   # train_d.py as named for M2 s101
LATER_MD5 = 'c010c9fb'  # train_d.py as named for M2 s202/s303


def git(*args):
    return subprocess.run(('git',) + args, capture_output=True, check=True).stdout


def blob(rev, path):
    return git('cat-file', 'blob', '%s:%s' % (rev, path))


def md5(b):
    return hashlib.md5(b).hexdigest()


def resolve(path, want):
    """Return the commit whose blob for ``path`` hashes to ``want``."""
    seen = {}
    for line in git('log', '--all', '--format=%H', '--', path).decode().split():
        h = md5(blob(line, path))[:8]
        seen.setdefault(h, line)
        if h == want:
            return line
    raise SystemExit('FAIL cannot resolve %s -> %s (saw %s)'
                     % (path, want, sorted(seen)))


def span(src, start_marker, end_marker):
    """1-based line numbers from the line holding start_marker up to (not
    including) the line holding end_marker."""
    i = src.index(start_marker)
    j = src.index(end_marker, i + 1)
    return set(range(src[:i].count('\n') + 1, src[:j].count('\n') + 1))


def call_site_lines(src, pattern):
    """Lines of a parenthesised statement that spans several lines."""
    out, lines = set(), src.splitlines()
    for i, ln in enumerate(lines):
        if not re.search(pattern, ln):
            continue
        depth, j = 0, i
        while j < len(lines):
            depth += lines[j].count('(') - lines[j].count(')')
            out.add(j + 1)
            if depth <= 0:
                break
            j += 1
    return out


def changed_lines(diff_text):
    """New-side line numbers of added/removed lines in a unified diff."""
    out, cur = [], None
    for ln in diff_text.splitlines():
        m = re.match(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@', ln)
        if m:
            cur = int(m.group(1))
            continue
        if cur is None or ln.startswith(('+++', '---')):
            continue
        if ln.startswith(('+', '-')):
            out.append(cur)
        if ln.startswith(('+', ' ')):
            cur += 1
    return out


def report(tag, ok, msg):
    print('%s %s %s' % ('PASS' if ok else 'FAIL', tag, msg))
    return ok


def main():
    a = resolve(TRAIN, S101_MD5)
    b = resolve(TRAIN, LATER_MD5)
    head = git('rev-parse', 'HEAD').decode().strip()
    print('snapshot named for s101      : %s  train_d=%s' % (a[:8], S101_MD5))
    print('snapshot named for s202/s303 : %s  train_d=%s' % (b[:8], LATER_MD5))
    print('HEAD                         : %s  train_d=%s'
          % (head[:8], md5(blob('HEAD', TRAIN))[:8]))
    ok = True

    hashes = {r[:8]: md5(blob(r, CDE))[:8] for r in (a, b, head)}
    ok &= report('A', len(set(hashes.values())) == 1,
                 'cde.py identical at s101/s202/HEAD: %s' % hashes)

    src = blob('HEAD', TRAIN).decode('utf-8')
    # line numbers in the a->b diff refer to b's file, so the span has to come
    # from b's source too -- reading it off HEAD breaks as soon as anything
    # above term_grads() changes, which is exactly what the fingerprint fix did
    src_b = blob(b, TRAIN).decode('utf-8')
    diag = span(src_b, 'def term_grads(', '\ndef ') | call_site_lines(
        src_b, r'term_grads\(net, arm, ebt')
    touched = changed_lines(git('diff', a, b, '--', TRAIN).decode('utf-8'))
    outside = [n for n in touched if n not in diag]
    ok &= report('B', not outside,
                 'a->b changes confined to term_grads(): %d changed lines, '
                 '%d outside %s' % (len(touched), len(outside), outside[:12]))

    m = re.search(r'^SCR_ARMS\s*=\s*(.+)$', src, re.M)
    scr = m.group(1).strip() if m else '<not found>'
    ok &= report('C', m is not None and "'M2'" not in scr,
                 'M2 not in SCR_ARMS (%s): scr=False on both sides' % scr)

    book = span(src, 'def code_fingerprint(', '\ndef _now(') | {
        i for i, ln in enumerate(src.splitlines(), 1)
        if 'code_fingerprint=_FP_AT_IMPORT' in ln}
    late = changed_lines(git('diff', b, head, '--', TRAIN).decode('utf-8'))
    bad = [n for n in late if n not in book]
    ok &= report('D', not bad,
                 'post-snapshot commits touch only the fingerprint recorder: '
                 '%d changed lines, %d outside %s'
                 % (len(late), len(bad), bad[:12]))

    print('AUDIT %s' % ('OK' if ok else 'NOT OK'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
