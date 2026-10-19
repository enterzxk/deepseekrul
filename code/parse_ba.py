import json,glob,os,re
import pandas as pd,numpy as np
meta=json.load(open('celllist.json'))['query_result']['data']['rows']
M={r['cell_id']:r for r in meta}
rows=[]
for p in glob.glob('ba_raw2/*.json'):
 d=json.load(open(p)); rs=d['query_result']['data']['rows']
 if not rs: continue
 # infer cell id from series
 s=rs[0]['series']; cell=s.split(': ',1)[1] if ': ' in s else os.path.basename(p)
 recs={}
 for r in rs:
  key=r['series'].split(':')[0]
  ci=int(float(r['cycle_index']))
  recs.setdefault(ci,{})[key]=float(r['value']) if r['value'] not in (None,'') else np.nan
  recs[ci]['test_time']=float(r['test_time']) if r['test_time'] not in (None,'') else np.nan
 dd=pd.DataFrame.from_dict(recs,orient='index').sort_index().reset_index().rename(columns={'index':'cycle'})
 dd['cell']=cell
 m=M.get(cell,{})
 for k in ['source','cathode','ah','temperature','soc_min','soc_max','crate_c','crate_d','form_factor']:
  dd[k]=m.get(k)
 rows.append(dd)
all=pd.concat(rows,ignore_index=True)
all.to_csv('ba_parsed/all_cycles.csv',index=False)
print('cells',all.cell.nunique(),'rows',len(all))
for cell,g in all.groupby('cell'):
 g=g.sort_values('cycle'); nom=M.get(cell,{}).get('ah',np.nan)
 soh=g.ah_d/nom if 'ah_d' in g else pd.Series(dtype=float)
 # only valid positive
 v=soh[(soh>0.1)&np.isfinite(soh)]
 print(f"{cell:55s} src={g.source.iloc[0]:7s} n={len(g):5d} cyc={int(g.cycle.min())}-{int(g.cycle.max()):5d} soh[min/med/max]={v.min():.3f}/{v.median():.3f}/{v.max():.3f} eol80={int(g.cycle[soh<=0.8].min()) if (soh<=0.8).any() else None}")
