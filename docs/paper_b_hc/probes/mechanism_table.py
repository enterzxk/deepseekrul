"""One table for the two mechanism numbers, over every candidate arm.

The stage-4 requirement is that every candidate has the same two numbers -- same-
age ranking accuracy and sensor-permutation degradation -- recorded with both the
absolute and the relative reading, and that arms without a checkpoint are listed
as `not_run` with a reason rather than silently missing.

Why a driver instead of the old shell loop: the loop skipped missing checkpoints
without a record, covered only seven arms, and hard-coded the a40 paths. A
missing arm must be a row with a reason, because "absent" and "zero" look the
same in a table and mean opposite things.

The arrival caveat is emitted as its own CSV column and printed after the table,
so it travels with the numbers instead of living in a footnote:

    arms whose prediction goes through first_passage accumulate damage
    monotonically, which itself suppresses permutation degradation; their
    degradation_pct is therefore NOT a share of "sensor contribution".

Usage
  python docs/paper_b_hc/probes/mechanism_table.py \
      --root experiments/hc/d80 --data-dir data/CMAPSS \
      --init-from experiments/hc/tmp/wave1_init_A1_s101.pt \
      --shared-dir experiments/hc/d80 --out MECHANISM_TABLE.csv --device cpu
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROBE = Path(__file__).resolve().parent / 'd_mechanism_m3b.py'

ARMS = ['B0', 'B1', 'B2', 'B3', 'M2', 'M3B', 'M3C', 'M4', 'M5', 'M6', 'M7',
        'AAR', 'AGE']

# Arms whose RUL prediction is produced by the arrival operator.
ARRIVAL = {'B2', 'B3', 'M3B', 'M3C', 'M6', 'M7'}

NOTE_ARRIVAL = ('arrival operator accumulates damage monotonically, which itself '
                'suppresses permutation degradation; pct is NOT a sensor share')
NOTE_DIRECT = 'prediction is reg_head(z) (+correction); no arrival operator'
NOTE_AGE = 'age-only control: never reads sensors, so 0 degradation is by construction'


def note_for(arm):
    if arm == 'AGE':
        return NOTE_AGE
    return NOTE_ARRIVAL if arm in ARRIVAL else NOTE_DIRECT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='experiments/hc/d80')
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--data-dir', default='data/CMAPSS')
    ap.add_argument('--init-from',
                    default='experiments/hc/tmp/wave1_init_A1_s101.pt')
    ap.add_argument('--shared-dir', default=None,
                    help='dir holding <subset>_shared.json (default: --root)')
    ap.add_argument('--out', default='MECHANISM_TABLE.csv')
    ap.add_argument('--arms', nargs='+', default=ARMS)
    ap.add_argument('--perm-seeds', nargs='+', type=int, default=[0, 1, 2])
    a = ap.parse_args()

    root = Path(a.root)
    shared = Path(a.shared_dir) if a.shared_dir else root
    recdir = root / '_mech'
    recdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for arm in a.arms:
        run_dir = root / ('HC-D-%s-%s-s%d-D1' % (arm, a.subset, a.seed))
        row = dict(arm=arm, subset=a.subset, seed=a.seed, run_dir=str(run_dir),
                   status='not_run', reason='', val_rmse='', rank_acc='',
                   pairs='', perm_mean='', degrad_abs='', degrad_pct='',
                   note=note_for(arm))
        ck = run_dir / 'ckpt' / 'best.pt'
        if not ck.exists():
            row['reason'] = 'no ckpt/best.pt at %s' % ck
            rows.append(row)
            print('%-4s not_run  %s' % (arm, row['reason']))
            continue
        cmd = [sys.executable, str(PROBE), '--arm', arm, '--subset', a.subset,
               '--run-dir', str(run_dir), '--device', a.device,
               '--root', str(ROOT), '--data-dir', a.data_dir,
               '--init-from', a.init_from, '--shared-dir', str(shared),
               '--out-dir', str(recdir),
               '--perm-seeds'] + [str(s) for s in a.perm_seeds]
        r = subprocess.run(cmd, capture_output=True, text=True)
        rec_path = recdir / ('mech_%s_%s.json' % (arm, a.subset))
        if r.returncode != 0 or not rec_path.exists():
            # Keep the line that names the failure, not just the last line: a
            # traceback's final source line ("sd = torch.load(...)") is not the
            # error, and reporting only that sent a whole night's run down the
            # wrong path. Fall back to the tail when no marker matches.
            msg = [l for l in (r.stderr or r.stdout or '').splitlines() if l.strip()]
            key = next((l for l in reversed(msg)
                        if re.search(r'(Error|Refusing|REFUSED|SystemExit)', l)), None)
            tail = ' / '.join(l.strip() for l in msg[-2:]) if msg else 'no record'
            row['status'] = 'error'
            row['reason'] = ('%s || tail: %s' % (key.strip(), tail)) if key else tail
            rows.append(row)
            print('%-4s error    %s' % (arm, row['reason']))
            continue
        rec = json.loads(rec_path.read_text(encoding='utf-8'))
        rk = rec.get('ranking_exact_cycle') or {}
        row.update(
            status='ok',
            val_rmse='%.4f' % rec['val_engine_mean_rmse'],
            rank_acc=('%.4f' % rk['accuracy']) if rk.get('accuracy') is not None else 'n/a',
            pairs='%d' % (rk.get('agree', 0) + rk.get('disagree', 0)),
            perm_mean='%.4f' % rec['permuted_mean'],
            degrad_abs='%+.4f' % rec['degradation'],
            degrad_pct='%+.2f%%' % rec['degradation_pct'])
        rows.append(row)
        print('%-4s ok       rmse=%s rank=%s pairs=%s perm=%s d=%s (%s)'
              % (arm, row['val_rmse'], row['rank_acc'], row['pairs'],
                 row['perm_mean'], row['degrad_abs'], row['degrad_pct']))

    cols = ['arm', 'subset', 'seed', 'status', 'reason', 'val_rmse', 'rank_acc',
            'pairs', 'perm_mean', 'degrad_abs', 'degrad_pct', 'run_dir', 'note']
    out = Path(a.out)
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in cols})

    n_ok = sum(1 for r in rows if r['status'] == 'ok')
    print('\nwrote %s  (%d ok / %d rows)' % (out, n_ok, len(rows)))
    print('ranking accuracy: 0.5 = age timer by construction; >0.5 = per-engine '
          'information the cycle alone cannot supply.')
    print('absolute degradation (cycles) and relative (%) are both reported: the '
          'relative reading is not comparable across arms with different RMSE.')
    print('CAVEAT (arrival arms): %s' % NOTE_ARRIVAL)
    print('CAVEAT (age control):  %s' % NOTE_AGE)
    return 0 if n_ok else 1


if __name__ == '__main__':
    sys.exit(main())
