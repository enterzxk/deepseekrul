import pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
r=pd.read_csv(H/'RESULTS.csv')
arms=[('M0','Plain backbone (PE)'),('FULL','Prior, additive'),('SPAN_GUARD','Prior + span + guard'),('MIXUP_SPAN','Mixup + span + guard')]
lines=['\\begin{table*}[t]','\\centering',
 '\\caption{Four metrics on the four C-MAPSS fleets. Gain is relative to the prior arm on the same fleet. Seeds: two on FD002 and FD004, one on FD001 and FD003.}',
 '\\label{tab:main}','\\small','\\begin{tabular}{llcrrrrr}','\\toprule',
 'Fleet & Arm & Seeds & RMSE & MAE & NASA & Bias & Gain \\\\','\\midrule']
for subset in ['FD001','FD002','FD003','FD004']:
    for arm,lab in arms:
        d=r[(r.subset==subset)&(r.arm==arm)]
        if d.empty: continue
        d=d.iloc[0]
        g='%+.2f\\%%'%(100*d.gain) if arm!='FULL' else '--'
        lines.append('%s & %s & %d & %.3f & %.3f & %.2f & %+.3f & %s \\\\'%(subset,lab,int(d.n_seeds),d.rmse_mean,d.mae,d.nasa,d.bias,g))
    lines.append('\\midrule')
lines[-1]='\\bottomrule'
lines += ['\\end{tabular}','\\end{table*}','']
out='\n'.join(lines)
(H/'paper_tables'/'main_table.tex').write_text(out)
print(out)