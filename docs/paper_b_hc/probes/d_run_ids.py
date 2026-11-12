"""48 vs 48: the dispatcher's ids against the registry's, line by line.

Two lists that can drift are worse than one that is wrong, so this compares
rather than generates: the ids come from `train_d.run_id` (the same function
that names the output directory), and the comparison target is whatever the
registry says. Any mismatch means the run would write a directory the ledger
cannot find, so it must block the launch.

Usage:
  python docs/paper_b_hc/probes/d_run_ids.py                    # emit + self-check
  python docs/paper_b_hc/probes/d_run_ids.py --registry <file>  # compare to a list
"""
import argparse
import re
from pathlib import Path

from paper_b.hc.train_d import run_id

ROOT = Path('/media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc/d')
ARMS = ['B0', 'B1', 'B2', 'B3']
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
SEEDS = [101, 202, 303]


def canonical():
    return [run_id(a, s, d) for s in SUBSETS for a in ARMS for d in SEEDS]


def from_commands(path):
    """Ids parsed out of the emitted command list, for a cross-check."""
    out = []
    for ln in Path(path).read_text(encoding='utf-8').splitlines():
        if '--subset' not in ln or not ln.strip().startswith('/'):
            continue
        d = dict(re.findall(r'--(\w+) (\S+)', ln))
        out.append(run_id(d['arm'], d['subset'], int(d['seed'])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default=None,
                    help='file with one registered id per line (registry section 3.6)')
    ap.add_argument('--commands', default=str(ROOT / 'matrix_commands.txt'))
    ap.add_argument('--out', default=str(ROOT / 'matrix_run_ids.txt'))
    a = ap.parse_args()

    ids = canonical()
    Path(a.out).write_text('\n'.join(ids) + '\n', encoding='utf-8')
    print('canonical: %d ids, %d unique -> %s'
          % (len(ids), len(set(ids)), a.out))
    bad = False

    if Path(a.commands).exists():
        got = from_commands(a.commands)
        same = got == ids
        print('dispatcher command list: %d ids, identical order: %s'
              % (len(got), same))
        if not same:
            bad = True
            for i, (x, y) in enumerate(zip(got, ids)):
                if x != y:
                    print('  first difference at %d: %s vs %s' % (i, x, y))
                    break
    else:
        print('dispatcher command list missing, skipped: %s' % a.commands)

    if a.registry:
        reg = [ln.strip() for ln in Path(a.registry).read_text(encoding='utf-8')
               .splitlines() if ln.strip() and not ln.startswith('#')]
        missing = [i for i in ids if i not in set(reg)]
        extra = [r for r in reg if r not in set(ids)]
        print('registry: %d ids; missing from registry: %d; not in canonical: %d'
              % (len(reg), len(missing), len(extra)))
        if missing:
            bad = True
            print('  missing (first 5): %s' % missing[:5])
        if extra:
            bad = True
            print('  extra (first 5): %s' % extra[:5])
        if reg != ids:
            print('  NOTE: same set but different order from the canonical listing')
    print('RESULT %s' % ('MISMATCH' if bad else 'MATCH'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
