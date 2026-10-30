import sys, numpy as np, pandas as pd
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
import run_candidate as rc

# reproduce only the partition, no training
def parts(subset, split_seed, fold):
    from dataset import load_cmapss_train
    from pathlib import Path
    raw = load_cmapss_train(Path('data/CMAPSS'), subset)
    units = np.sort(raw['unit'].unique())
    rng = np.random.default_rng(split_seed + 20261020)
    sh = rng.permutation(units)
    n = max(1, int(round(units.size * 0.2)))
    if fold >= 0:
        v = sorted(int(x) for x in sh[fold*n:(fold+1)*n])
        t = sorted(int(x) for x in units if int(x) not in set(v))
    else:
        v = sorted(int(x) for x in sh[:n])
        t = sorted(int(x) for x in sh[n:])
    return v, t

for sub, ref in (('FD002', 'experiments/e3/A_s2027/FD002/A_s2027/val_units.csv'),
                 ('FD001', 'experiments/e3_fd001/A_s2027/FD001/A_s2027/val_units.csv')):
    old = sorted(pd.read_csv(ref).unit_id.tolist())
    v0, _ = parts(sub, 7, 0)
    print(f'{sub}: fold0 == archived split7 ? {v0 == old}   (n={len(v0)})')
    allv = []
    for f in range(5):
        v, t = parts(sub, 7, f)
        allv.append(v)
        print(f'   fold{f}: {len(v)} val, {len(t)} train')
    flat = [x for v in allv for x in v]
    print(f'   union={len(set(flat))} of {len(set(flat))} total, duplicates={len(flat)-len(set(flat))}')
