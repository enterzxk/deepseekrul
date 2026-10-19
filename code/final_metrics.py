import pandas as pd,numpy as np,json
S=pd.read_csv('ba_cell_summary.csv').set_index('cell')
H=pd.read_csv('hof_eval_clean.csv'); H=H[['cell','source','k0','h','true','pmean','pred','fold']].rename(columns={'pred':'hof'}); H['err']=H.hof-H.true
P=pd.read_csv('unified_percell.csv'); P=P[P.k0>=100]
Pv=P.pivot_table(index=['cell','k0','h'],columns='model',values='pred').reset_index()
G=pd.read_csv('gru_pred.csv'); G=G[G.k0>=100][['cell','k0','h','pred']].rename(columns={'pred':'gru'})
RG=pd.read_csv('resgru_pred.csv'); RG=RG[RG.k0>=100][['cell','k0','h','pred']].rename(columns={'pred':'resgru'})
M=H.merge(Pv,on=['cell','k0','h']).merge(G,on=['cell','k0','h']).merge(RG,on=['cell','k0','h'])
M['source']=M.cell.map(S.source)
methods=['persist','linear100','exp100','power','pmean','hof','gru','resgru']
for m in methods: M['e_'+m]=M[m]-M.true
out={'n_cells_common':int(M.cell.nunique()),'n_windows_common':int(len(M)),'horizons':{}}
for h in [50,100,200,300]:
 x=M[M.h==h]; d={}
 for m in methods:
  d[m]={'mae_cell':float(x.groupby('cell').apply(lambda z:np.mean(np.abs(z['e_'+m]))).mean()),
        'mae_window':float(x['e_'+m].abs().mean()),'bias':float(x['e_'+m].mean())}
 out['horizons'][str(h)]=d
# per-source h300
x=M[M.h==300]; ps={}
for src in sorted(x.source.unique()):
 y=x[x.source==src]; ps[src]={'n_cells':int(y.cell.nunique())}
 for m in ['power','pmean','hof','gru','resgru']:
  ps[src][m]=float(y.groupby('cell').apply(lambda z:np.mean(np.abs(z['e_'+m]))).mean())
out['per_source_h300']=ps
# paired wins
w=x.groupby('cell').apply(lambda z:np.mean(np.abs(z.e_hof))-np.mean(np.abs(z.e_gru)))
w2=x.groupby('cell').apply(lambda z:np.mean(np.abs(z.e_hof))-np.mean(np.abs(z.e_power)))
out['paired_hof_vs_gru_h300']={'hof_wins':int((w<0).sum()),'gru_wins':int((w>0).sum()),'median_delta':float(w.median())}
out['paired_hof_vs_power_h300']={'hof_wins':int((w2<0).sum()),'power_wins':int((w2>0).sum()),'median_delta':float(w2.median())}
# EOL HOF clean
E=pd.read_csv('hof_eol_clean.csv'); e=E.dropna(subset=['err'])
def eol_stats(z):
 return {'n_cases':int(len(z)),'n_cells':int(z.cell.nunique()),'mae_case':float(z.err.abs().mean()),'mae_cell':float(z.groupby('cell').err.apply(lambda e:np.mean(np.abs(e))).mean()),'median_abs':float(z.err.abs().median()),'within20':float(np.mean(z.err.abs()/z.eol<0.2)),'bias':float(z.err.mean())}
out['eol_hof']=eol_stats(e); out['eol_hof_censored_cases']=int(E.pred.isna().sum())
out['eol_hof_by_source']={src:eol_stats(e[e.source==src]) for src in sorted(e.source.unique())}
# EOL compare
C=pd.read_csv('eol_compare_clean.csv'); out['eol_models']={}
for m in ['hof','pmean','linear','exp','power']:
 z=C[C.model==m]; zz=z.dropna(subset=['err'])
 out['eol_models'][m]={'n_cases':int(len(zz)),'mae_case':float(zz.err.abs().mean()),'mae_cell':float(zz.groupby('cell').err.apply(lambda e:np.mean(np.abs(e))).mean()),'median_abs':float(zz.err.abs().median()),'within20':float(np.mean(zz.err.abs()/zz.eol<0.2)),'bias':float(zz.err.mean()),'censored':int(z.pred.isna().sum())}
# LO source
L=pd.read_csv('losource_hof_pred.csv'); l=L[L.h==300]
out['losource_h300']={'n_cells':int(l.cell.nunique()),'mae_cell':float(l.groupby('cell').err.apply(lambda e:np.mean(np.abs(e))).mean()),'mae_window':float(l.err.abs().mean()),'bias':float(l.err.mean()),
 'by_source':{src:float(l[l.source==src].groupby('cell').err.apply(lambda e:np.mean(np.abs(e))).mean()) for src in sorted(l.source.unique())}}
# conformal
conf={}
for h in [50,100,200,300]:
 cov={}; wid={}
 for level in [0.8,0.9]:
  cs=[]; ws=[]
  for f in range(5):
   tr=H[(H.fold!=f)&(H.h==h)]; te=H[(H.fold==f)&(H.h==h)]
   q=np.quantile(np.abs(tr.err),level); cs.append(np.mean(np.abs(te.err)<=q)); ws.append(q)
  cov[str(level)]=float(np.mean(cs)); wid[str(level)]=float(np.mean(ws))
 conf[str(h)]={'coverage':cov,'width':wid}
out['conformal_soh']=conf
# EOL conformal
ec={}
for level in [0.8,0.9]:
 cs=[]; ws=[]
 for f in range(5):
  tr=e[e.fold!=f]; te=e[e.fold==f]
  q=np.quantile(np.abs(tr.err),level); cs.append(np.mean(np.abs(te.err)<=q)); ws.append(q)
 ec[str(level)]={'coverage':float(np.mean(cs)),'width':float(np.mean(ws))}
out['conformal_eol']=ec
# data summary
summ=S.reset_index().groupby('source').agg(n=('cell','size'),eol80_obs=('eol80','count'),eol70_obs=('eol70','count'),cycles_median=('cyc_max','median')).to_dict(orient='index')
out['data_summary']={k:{kk:(int(vv) if kk!='cycles_median' else float(vv)) for kk,vv in v.items()} for k,v in summ.items()}
out['total_cells']=int(len(S)); out['total_eol80_obs']=int(S.eol80.notna().sum())
json.dump(out,open('final_metrics.json','w'),indent=2)
print(json.dumps(out,indent=2)[:8000])
