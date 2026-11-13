"""Cross-snapshot audit for B3 / M3B / M6: did the code that decides their
numbers change between the snapshots their seeds ran on?

Why this exists. The manager found that for each of B3, M3B and M6 the first seed
ran on one code fingerprint and the other two on another, so the reported
per-arm sd mixes seed variance with code variance. Re-running 15 cells to fix
that is ~3 h; deciding whether the sd is usable costs one static audit plus one
reproduction cell. This is the static audit.

What it can and cannot prove. It resolves each recorded fingerprint to a commit
by recomputing the same aggregate the trainer records, then classifies every
changed line of train_d.py by the function that encloses it. It therefore proves
"no changed line lies inside the functions this arm executes". It does NOT prove
bit-identical runs -- only the one-cell reproduction (M3B-s101, preds_md5
684345ff...) can. The two together are the claim; either alone is not.

Usage
  python docs/paper_b_hc/probes/audit_snapshot_stability.py
  python docs/paper_b_hc/probes/audit_snapshot_stability.py --verbose
"""
import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMES = ('arrival.py', 'hi_d.py', 'data_d.py', 'train_d.py', 'config_hc.py')
MOD = 'paper_b/hc/'

# arm -> (fingerprint of its first seed, fingerprint of the later seeds)
ARMS = {
    'B3': ('ce031984', 'c7df9d16'),
    'M3B': ('6f8f8423', '7f7f6fcf'),
    'M6': ('1883b283', 'c7df9d16'),
}

# Functions that provably cannot change the numbers a run produces:
#   term_grads                  -- runs after the last optimiser step, only feeds
#                                  history.json; autograd.grad writes no .grad
#   code_fingerprint            -- the recorder itself
#   _selfcheck_degenerate_guard -- pre-run guard on synthetic numpy arrays
OFF_PATH = {'term_grads', 'code_fingerprint', '_selfcheck_degenerate_guard'}

# The recorded fingerprint covers five files. These four carry the modules that
# M2/M4/M5/M6 actually add and are NOT in it, so a change there would never show
# up in files_sha256 -- checked separately here.
EXTRA = ('cde.py', 'csr_pool.py', 'efr.py', 'dpc.py')

RESULTS = []


def check(tag, ok, msg):
    RESULTS.append(bool(ok))
    print('%s %s %s' % ('PASS' if ok else 'FAIL', tag, msg))
    return ok


def git(*args):
    return subprocess.run(('git',) + args, capture_output=True, check=True).stdout


def blob(rev, path):
    r = subprocess.run(('git', 'cat-file', 'blob', '%s:%s' % (rev, path)),
                       capture_output=True)
    if r.returncode != 0:
        raise KeyError('%s has no %s' % (rev[:8], path))
    return r.stdout


def aggregate(rev):
    """The same aggregate code_fingerprint() records: sha256 over name+bytes,
    LF-normalised (the 3090 checkout is LF, so this equals the old raw-bytes
    value it recorded). None when a commit predates one of the five files."""
    h = hashlib.sha256()
    try:
        for n in NAMES:
            b = blob(rev, MOD + n).replace(b'\r\n', b'\n')
            h.update(n.encode())
            h.update(b)
    except KeyError:
        return None
    return h.hexdigest()[:32]


def resolve():
    """fingerprint[:8] -> commit, over every commit that touched one of the five
    files (the aggregate can only change when one of them does)."""
    commits = []
    seen = set()
    for n in NAMES:
        for line in git('log', '--all', '--format=%H', '--', MOD + n).decode().split():
            if line not in seen:
                seen.add(line)
                commits.append(line)
    out = {}
    for c in commits:
        agg = aggregate(c)
        if agg:
            out.setdefault(agg[:8], c)
    return out


def func_spans(src):
    """[(name, start_line, end_line)] for top-level defs, 1-based inclusive."""
    lines = src.splitlines()
    starts = [(i + 1, m.group(1)) for i, ln in enumerate(lines)
              for m in [re.match(r'^def (\w+)\(', ln)] if m]
    spans = []
    for k, (ln, name) in enumerate(starts):
        end = (starts[k + 1][0] - 1) if k + 1 < len(starts) else len(lines)
        spans.append((name, ln, end))
    return spans


def func_at(spans, lineno):
    for name, a, b in spans:
        if a <= lineno <= b:
            return name
    return '<module>'


def changed_lines(diff_text):
    """(new-side line number, sign, text) for every added/removed line."""
    out, cur = [], None
    for ln in diff_text.splitlines():
        m = re.match(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@', ln)
        if m:
            cur = int(m.group(1))
            continue
        if cur is None or ln.startswith(('+++', '---')):
            continue
        if ln.startswith(('+', '-')):
            out.append((cur, ln[0], ln[1:].rstrip()))
        if ln.startswith(('+', ' ')):
            cur += 1
    return out


def arm_tuples(src):
    """Module-level arm tuples, so a changed line can be asked 'is it gated on an
    arm set that excludes the arm under audit?'."""
    ns, out = {}, {}
    for m in re.finditer(r'^([A-Z_]+) = (\([^)\n]*\))(\s*\+\s*([A-Z_]+))?\s*$',
                         src, re.M):
        name, expr, _, extra = m.groups()
        try:
            val = eval(expr, {}, {})
            if extra:
                val = val + ns.get(extra, ())
            ns[name] = val
            out[name] = val
        except Exception:
            continue
    return out


def sets_in(text):
    return re.findall(r'arm (?:in|==|not in) ([A-Z_]+)', text)


def classify_line(lineno, text, src_lines, tuples, arm, fn_start):
    """'gated' | 'arm-specific' | 'ungated' for one changed line.

    gated        -- it cannot execute for this arm (inside a block whose header
                    tests an arm set that excludes it, or an inline conditional
                    that does)
    arm-specific -- it is about arms and this arm is one of them
    ungated      -- no arm test between it and its function start
    """
    names = sets_in(text)
    if names:
        return 'gated' if all(arm not in tuples.get(n, ()) for n in names) \
            else 'arm-specific'
    indent = len(text) - len(text.lstrip())
    for j in range(lineno - 1, fn_start - 1, -1):
        ln = src_lines[j - 1]
        if not ln.strip() or ln.lstrip().startswith('#'):
            continue
        if len(ln) - len(ln.lstrip()) < indent:
            if re.match(r'\s*(if|elif) ', ln) and sets_in(ln):
                return ('gated' if all(arm not in tuples.get(n, ()) for n in sets_in(ln))
                        else 'arm-specific')
            indent = len(ln) - len(ln.lstrip())
    return 'ungated'


def docstring_lines(src_lines, spans):
    """1-based line numbers inside each function's docstring. A docstring change
    cannot change a number, so it must not count against the audit."""
    out = set()
    for _name, a, b in spans:
        j = a                      # 0-based index of the first body line
        while j < b and not src_lines[j].strip():
            j += 1
        if j >= b:
            continue
        s = src_lines[j].strip()
        q = s[:3]
        if q not in ('"""', "'''"):
            continue
        out.add(j + 1)
        if s.count(q) >= 2 and len(s) > 3:
            continue
        j += 1
        while j < b and q not in src_lines[j]:
            out.add(j + 1)
            j += 1
        if j < b:
            out.add(j + 1)
    return out


def fn_start_of(spans, name):
    for n, a, _ in spans:
        if n == name:
            return a
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    print('--- resolving recorded fingerprints to commits ---')
    fmap = resolve()
    for arm, (f1, f2) in ARMS.items():
        print('  %-4s %s -> %s   %s -> %s'
              % (arm, f1, (fmap.get(f1) or '<UNRESOLVED>')[:8],
                 f2, (fmap.get(f2) or '<UNRESOLVED>')[:8]))
    unresolved = [f for arm in ARMS for f in ARMS[arm] if f not in fmap]
    # An unresolvable fingerprint is itself a finding: that run cannot be tied to
    # any commit, so it is not auditable at all. It does not abort the others.
    check('R1 every recorded fingerprint resolves to a commit', not unresolved,
          'unresolved=%s (of %d commits scanned)' % (unresolved, len(fmap)))

    for arm, (f1, f2) in ARMS.items():
        print('\n=== %s : %s -> %s ===' % (arm, f1, f2))
        if f1 not in fmap or f2 not in fmap:
            check('%s auditable' % arm, False,
                  'a snapshot of this arm cannot be tied to a commit, so nothing '
                  'about it can be verified')
            continue
        c1, c2 = fmap[f1], fmap[f2]

        mods = {n: ('same' if blob(c1, MOD + n) == blob(c2, MOD + n) else 'CHANGED')
                for n in NAMES if n != 'train_d.py'}
        check('%s module files identical' % arm,
              all(v == 'same' for v in mods.values()), str(mods))

        # the modules the fingerprint does not cover
        extra = {}
        for n in EXTRA:
            try:
                extra[n] = ('same' if blob(c1, MOD + n) == blob(c2, MOD + n)
                            else 'CHANGED')
            except KeyError:
                extra[n] = 'absent in one snapshot'
        check('%s fingerprint-external modules identical' % arm,
              all(v == 'same' for v in extra.values()), str(extra))

        diff = git('diff', c1, c2, '--', MOD + 'train_d.py').decode('utf-8')
        src2 = blob(c2, MOD + 'train_d.py').decode('utf-8')
        src_lines = src2.splitlines()
        spans = func_spans(src2)
        tuples = arm_tuples(src2)
        rows = changed_lines(diff)

        tally = {}
        ungated = []
        docs = docstring_lines(src_lines, spans)
        for lineno, sign, text in rows:
            fn = func_at(spans, lineno)
            stripped = text.strip()
            if stripped == '' or stripped.startswith('#') or lineno in docs:
                kind = 'cosmetic'
            elif fn in OFF_PATH:
                kind = 'off-path function'
            elif fn == '<module>':
                kind = 'module-level'
            else:
                kind = classify_line(lineno, text, src_lines, tuples, arm,
                                     fn_start_of(spans, fn))
                if kind == 'ungated':
                    ungated.append((fn, lineno, sign, text))
            tally[kind] = tally.get(kind, 0) + 1
        print('  changed lines: %s' % dict(sorted(tally.items())))
        if a.verbose:
            for lineno, sign, text in rows:
                print('    %-28s %5d %s %s' % (func_at(spans, lineno), lineno, sign, text))

        check('%s no ungated change on the numeric path' % arm, not ungated,
              '%d ungated line(s)%s' % (len(ungated),
                                        ': ' + str(ungated[:5]) if ungated else ''))

    ok = all(RESULTS)
    print('\nSNAPSHOT AUDIT %s (%d/%d)'
          % ('OK' if ok else 'NOT OK', sum(RESULTS), len(RESULTS)))
    print('NOTE: static audit only. Bit-identical runs still need the one-cell '
          'reproduction (M3B-FD001-s101, preds_md5 684345ff...).')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
