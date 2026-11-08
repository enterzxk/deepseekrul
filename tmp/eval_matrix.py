"""Evaluate the full matrix on the official test: per-model + ensemble, four subsets."""
import glob, json, os, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('/home/user/rulb')
PY = '/home/user/miniconda3/envs/pmv/bin/python'

def run_eval(subset, fold, seed, ckpt, out):
    cmd = [PY, str(ROOT/'scripts'/'eval_dspt.py'), '--subset', subset, '--fold', str(fold),
           '--seed', str(seed), '--layers', '2', '--ckpt', str(ckpt), '--out', str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0

def main():
    cache = ROOT/'experiments'/'TESTCACHE'
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(str(ROOT/'experiments'/'M_*'))):
        name = os.path.basename(d)
        parts = name.split('_')
        if len(parts) < 5: continue
        subset = parts[1]; seed = parts[2].lstrip('s'); fold = parts[3].lstrip('f')
        ckpt = Path(d)/subset/f'dspt_s{seed}.pt'
        if not ckpt.is_file(): continue
        out = cache/f'{subset}_s{seed}_f{fold}.csv'
        if not out.is_file():
            if not run_eval(subset, fold, seed, ckpt, out):
                print('FAIL', name); continue
        df = pd.read_csv(out)
        rows.append(dict(subset=subset, seed=int(seed), fold=int(fold),
                         rmse=float(np.sqrt(np.mean((df.y_pred-df.y_true)**2))), n=len(df)))
    if not rows:
        print('no results yet'); return
    r = pd.DataFrame(rows)
    print('=== per-model official test RMSE ===')
    for subset in ['FD001','FD002','FD003','FD004']:
        s = r[r.subset == subset]
        if s.empty: continue
        print('%s: n=%d models  mean %.4f  sd %.4f  min %.4f  max %.4f'
              % (subset, len(s), s.rmse.mean(), s.rmse.std(), s.rmse.min(), s.rmse.max()))
    print()
    print('=== ensemble (all models per subset) ===')
    for subset in ['FD001','FD002','FD003','FD004']:
        files = sorted(cache.glob(f'{subset}_s*_f*.csv'))
        if not files: continue
        dfs = [pd.read_csv(f).sort_values('unit').reset_index(drop=True) for f in files]
        y = dfs[0].y_true.to_numpy()
        P = np.stack([d.y_pred.to_numpy() for d in dfs], axis=0)
        ens = P.mean(axis=0)
        print('%s: %d models  ENSEMBLE %.4f' % (subset, len(dfs), np.sqrt(np.mean((ens-y)**2))))

if __name__ == '__main__':
    main()