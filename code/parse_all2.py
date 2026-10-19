import sys,glob,os
import pandas as pd, numpy as np

def info_of(path):
    xl=pd.ExcelFile(path)
    if 'Info' not in xl.sheet_names: return {}
    info=pd.read_excel(path,sheet_name='Info',header=None)
    hdr=None
    for i in range(min(10,len(info))):
        if str(info.iloc[i,0]).strip()=='Channel': hdr=i; break
    if hdr is None: return {}
    row=info.iloc[hdr+1]
    cols={str(info.iloc[hdr,j]).strip():j for j in range(info.shape[1])}
    def g(k,d):
        j=cols.get(k,d); 
        return row.iloc[j] if j is not None and j<len(row) else None
    return {'schedule':str(g('Schedule_File_Name',2)), 'start':pd.to_datetime(g('Start_DateTime',1),errors='coerce')}

def parse_file(path):
    xl=pd.ExcelFile(path)
    ch=[s for s in xl.sheet_names if s.startswith('Channel_')]
    if not ch: return None
    meta=info_of(path)
    df=pd.read_excel(path,sheet_name=ch[0])
    if 'Cycle_Index' not in df: return None
    for c in ['Discharge_Capacity(Ah)','Charge_Capacity(Ah)','Voltage(V)','Internal_Resistance(Ohm)','Date_Time']:
        if c not in df.columns: df[c]=np.nan
    g=df.groupby('Cycle_Index',sort=True)
    out=g.agg(cap_dis=('Discharge_Capacity(Ah)',lambda x:x.max()-x.min() if len(x) else np.nan),
              cap_chg=('Charge_Capacity(Ah)',lambda x:x.max()-x.min() if len(x) else np.nan),
              vmax=('Voltage(V)','max'),vmin=('Voltage(V)','min'),
              ir=('Internal_Resistance(Ohm)','median'),
              t_start=('Date_Time','min'),t_end=('Date_Time','max'),
              n=('Voltage(V)','size')).reset_index()
    out['file']=os.path.basename(path); out['schedule']=meta.get('schedule',''); out['file_start']=meta.get('start',pd.NaT)
    return out

def parse_cell(cell, root):
    files=sorted(glob.glob(os.path.join(root,cell,'*.xlsx')))
    parts=[]
    for p in files:
        try:
            d=parse_file(p)
            if d is not None and len(d): parts.append(d)
        except Exception as e:
            print('skip',cell,os.path.basename(p),repr(e),flush=True)
    if not parts: raise ValueError('no parts')
    parts.sort(key=lambda d:(str(d.file_start.iloc[0]),d.file.iloc[0]))
    offset=0; out=[]
    for d in parts:
        d=d.copy(); d['cycle_local']=d.Cycle_Index; d['cycle_global']=d.cycle_local+offset
        offset=int(d.cycle_global.max()); out.append(d)
    d=pd.concat(out,ignore_index=True); d['cell']=cell
    return d

if __name__=='__main__':
    cells=sorted([os.path.basename(d) for d in glob.glob('raw/*') if os.path.isdir(d)])
    os.makedirs('parsed2',exist_ok=True)
    for cell in cells:
        try:
            d=parse_cell(cell,'raw'); d.to_csv(f'parsed2/{cell}.csv',index=False)
            print(cell,len(d),'cyc',d.cycle_global.max(),'capmax',round(d.cap_dis.max(),3),'capmin',round(d.cap_dis.min(),3),'sched',d.schedule.unique()[:3],flush=True)
        except Exception as e: print('ERR',cell,repr(e),flush=True)
