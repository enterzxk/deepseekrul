import pandas as pd, numpy as np, json
all=pd.read_csv('ba_parsed/all_cycles.csv')
meta=json.load(open('celllist.json'))['query_result']['data']['rows']; M={r['cell_id']:r for r in meta}
out=[]; summ=[]
for cell,g in all.groupby('cell'):
    g=g.sort_values('cycle').copy()
    nom=M.get(cell,{}).get('ah',np.nan)
    # valid cycles: full-ish charge/discharge, plausible capacity
    ok=(g.ah_d>0.2*nom)&(g.ah_d<1.4*nom)&(g.ah_c>0.2*nom)&(g.ah_c<1.4*nom)
    ok &= (g.ah_d/g.ah_c>0.8)&(g.ah_d/g.ah_c<1.2)
    g=g[ok].copy()
    if len(g)<20: continue
    # initial capacity from first up to 5 valid cycles
    init=np.median(g.ah_d.values[:5])
    g['soh']=g.ah_d/init; g['nom']=nom
    # smooth target (centered median 9) and causal input (median 5 trailing)
    g['soh_s']=g.soh.rolling(9,center=True,min_periods=3).median()
    g['soh_c']=g.soh.rolling(5,center=True,min_periods=1).median()
    # threshold crossings on smoothed
    def first(th):
        idx=np.where(g.soh_s.values<=th)[0]
        return int(g.cycle.values[idx[0]]) if len(idx) else None
    e80=first(0.8); e70=first(0.7)
    summ.append(dict(cell=cell,source=g.source.iloc[0],chem=g.cathode.iloc[0],nom=nom,init=init,n=len(g),
                     cyc_min=int(g.cycle.min()),cyc_max=int(g.cycle.max()),soh_min=float(g.soh.min()),soh_max=float(g.soh.max()),
                     eol80=e80,eol70=e70,temp=g.temperature.iloc[0],crate_c=g.crate_c.iloc[0],crate_d=g.crate_d.iloc[0]))
    out.append(g[['cell','cycle','soh','soh_s','soh_c','ah_d','ah_c','source','cathode','nom','temperature','crate_c','crate_d']])
clean=pd.concat(out,ignore_index=True)
clean.to_csv('ba_clean.csv',index=False)
S=pd.DataFrame(summ); S.to_csv('ba_cell_summary.csv',index=False)
print('cells',len(S),'rows',len(clean))
print(S.groupby('source').agg(n=('cell','size'),n_eol80=('eol80','count'),n_eol70=('eol70','count'),cyc_med=('cyc_max','median')).to_string())
print(S.sort_values(['source','cell']).to_string(index=False))
