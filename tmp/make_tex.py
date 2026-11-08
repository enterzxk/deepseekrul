"""Generate LaTeX tables for the manuscript."""
from pathlib import Path
import pandas as pd
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
OUT=HERE/'paper_tables'
r=pd.read_csv(HERE/'RESULTS.csv')
r=r[r.arm.isin(['M0','OPS','FULL','NORM'])]
lines=[r'\begin{table}[t]', r'\centering',
       r'\caption{Pooled metrics on the four C-MAPSS subsets. RMSE and MAE in cycles; the asymmetric score is the per-window mean. n is the number of seeds; the interval is an engine-clustered paired bootstrap against the locked reference.}',
       r'\label{tab:main}', r'\small', r'\begin{tabular}{llcccccr}', r'\toprule',
       r'subset & arm & n & RMSE & MAE & score & bias & $\Delta$ [95\% CI] \\', r'\midrule']
seen=set()
for subset in ['FD001','FD002','FD003','FD004']:
    g=r[r.subset==subset]
    for _,row in g.iterrows():
        ci = '' if pd.isna(row.get('delta_rmse')) else '%.2f [%.2f, %.2f]'%(row.delta_rmse,row.ci_lo,row.ci_hi)
        name = row.arm if row.arm!='FULL' else r'\textbf{FULL}'
        first = subset not in seen
        seen.add(subset)
        lines.append('%s & %s & %d & %.2f & %.2f & %.2f & %+.2f & %s \\\\'%(
            subset if first else '', name, int(row.n_seeds),
            row.rmse_mean, row.mae, row.nasa, row.bias, ci))
    lines.append(r'\midrule')
lines[-1]=r'\bottomrule'
lines += [r'\end{tabular}', r'\end{table}']
(OUT/'main_table.tex').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:12]))