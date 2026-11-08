import glob
import numpy as np, pandas as pd
from pathlib import Path
C = Path('/home/user/rulb/experiments/TESTCACHE')
def nasa(e): return np.where(e > 0, np.expm1(e/10.0), np.expm1(-e/13.0))
rows = []
for subset in ['FD001','FD002','FD003','FD004']:
    fs = sorted(C.glob(f'{subset}_s*_f*.csv'))
    ds = [pd.read_csv(f).sort_values('unit').reset_index(drop=True) for f in fs]
    y = ds[0].y_true.to_numpy()
    P = np.stack([d.y_pred.to_numpy() for d in ds])
    def met(p):
        e = p - y
        g = pd.DataFrame({'u': ds[0].unit, 'e2': e**2, 'ae': np.abs(e), 's': nasa(e), 'b': e}).groupby('u').mean()
        return float(np.sqrt(g.e2.mean())), float(g.ae.mean()), float(g.s.mean()), float(g.b.mean())
    per = [met(p) for p in P]
    rm = np.array([x[0] for x in per])
    ens = met(P.mean(axis=0))
    rows.append(dict(subset=subset, models=len(fs), n_eng=len(ds[0]),
                     single_rmse=round(rm.mean(),4), single_sd=round(rm.std(),4),
                     single_min=round(rm.min(),4), single_max=round(rm.max(),4),
                     ens_rmse=round(ens[0],4), ens_mae=round(ens[1],4),
                     ens_nasa=round(ens[2],4), ens_bias=round(ens[3],4)))
r = pd.DataFrame(rows)
r.to_csv('/home/user/rulb/RESULTS_V4.csv', index=False)
print(r.to_string(index=False))

ref = {'FD001':10.43,'FD002':12.02,'FD003':10.49,'FD004':14.11}
T2 = {'FD001':10.60,'FD002':12.20,'FD003':10.70,'FD004':13.60}
T1 = {'FD001':10.30,'FD002':11.90,'FD003':10.40,'FD004':13.40}
lines = ['\\begin{table}[t]','\\centering',
 '\\caption{Official-test results (one final window per engine, targets capped at 125). Single is the mean over 15 models (5 folds x 3 seeds); ensemble averages their predictions.}',
 '\\label{tab:main}','\\small','\\begin{tabular}{lrrrrr}','\\toprule',
 'Fleet & Single & Ensemble & MAE & NASA & vs.~\cite{ast2026} \\\\','\\midrule']
for _, x in r.iterrows():
    lines.append('%s & %.2f$\\pm$%.2f & \\textbf{%.2f} & %.2f & %.2f & %+.1f\\%% \\\\'
                 % (x.subset, x.single_rmse, x.single_sd, x.ens_rmse, x.ens_mae, x.ens_nasa,
                    100*(x.ens_rmse/ref[x.subset]-1)))
lines += ['\\bottomrule','\\end{tabular}','\\end{table}','']
Path('/home/user/rulb/paper_tables_v4').mkdir(exist_ok=True)
Path('/home/user/rulb/paper_tables_v4/main.tex').write_text('\n'.join(lines))
print()
print('\n'.join(lines))