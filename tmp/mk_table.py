import pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
r=pd.read_csv(H/'RESULTS.csv')
order=['M0','OPS','FULL','NORM','SPAN_GUARD','MIXUP_SPAN']
label={'M0':'Plain (PE)','OPS':'Prior, additive (FULL)','FULL':'Prior, additive (FULL)','NORM':'Prior, time base','SPAN_GUARD':'Span + guard','MIXUP_SPAN':'Mixup + span + guard'}
rows=[]
for subset in ['FD001','FD002','FD003','FD004']:
    for arm in order:
        d=r[(r.subset==subset)&(r.arm==arm)]
        if d.empty: continue
        d=d.iloc[0]
        ci='[%.2f, %.2f]'%(d.ci_lo,d.ci_hi) if pd.notna(d.ci_lo) and str(d.ci_lo)!='' else '--'
        lab=label.get(arm,arm)
        if arm=='M0' and subset in ('FD002','FD004'): lab='Plain (PE)'
        rows.append((subset,lab,d.rmse_mean,d.mae,d.nasa,d.bias,d.gain,ci))
lines=['\\begin{table*}[t]','\\centering',
 '\\caption{Four metrics on the four C-MAPSS fleets. Gain is relative to the prior arm (FULL) on the same fleet; intervals are engine-clustered paired bootstrap (5000 resamples).}',
 '\\label{tab:main}','\\small','\\begin{tabular}{llrrrrrr}','\\toprule',
 'Fleet & Arm & RMSE & MAE & NASA & Bias & Gain & 95\\% CI \\\\','\\midrule']
last=None
for s,lab,rm,ma,na,bi,ga,ci in rows:
    if last is not None and s!=last: lines.append('\\midrule')
    lines.append('%s & %s & %.3f & %.3f & %.2f & %+.3f & %+.2f\\%% & %s \\\\'%(s,lab,rm,ma,na,bi,100*ga,ci))
    last=s
lines += ['\\bottomrule','\\end{tabular}','\\end{table*}','']
out='\n'.join(lines)
(H/'paper_tables'/'main_table.tex').write_text(out)
print(out[:1200])