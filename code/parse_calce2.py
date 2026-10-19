import sys, glob, os
import pandas as pd, numpy as np

def info_of(path):
    info=pd.read_excel(path,sheet_name='Info',header=None)
    # locate header row containing 'Channel'
    hdr=None
    for i in range(min(10,len(info))):
        if str(info.iloc[i,0]).strip()=='Channel': hdr=i; break
    if hdr is None: return {}
    row=info.iloc[hdr+1]
    cols={str(info.iloc[hdr,j]).strip():j for j in range(info.shape[1])}
    return {'schedule':str(row[cols.get('Schedule_File_Name',2)]), 'start':pd.to_datetime(row[cols.get('Start_DateTime',1)],errors='coerce'), 'testdate':pd.to_datetime(info.iloc[hdr-1,3],errors='coerce') if hdr>=1 else pd.NaT}

def parse_file(path):
    meta=info_of(path)
    xl=pd.ExcelFile(path); ch=[s for s in xl.sheet_names if s.startswith('Channel_')]
    if not ch: return None
    df=pd.read_excel(path,sheet_name=ch[0])
    if 'Cycle_Index' not in df: return None
    for c in ['Discharge_Capacity(Ah)','Charge_Capacity(Ah)','Voltage(V)','Internal_Resistance(Ohm)','Date_Time','Current(A)']:
        if c not in df.columns: df[c]=np.nan
    g=df.groupby('Cycle_Index',sort=True)
    out=g.agg(cap_dis=('Discharge_Capacity(Ah)',lambda x:x.max()-x.min() if len(x) else np.nan),
              cap_chg=('Charge_Capacity(Ah)',lambda x:x.max()-x.min() if len(x) else np.nan),
              vmax=('Voltage(V)','max'),vmin=('Voltage(V)','min'),
              ir=('Internal_Resistance(Ohm)','median'),
              t_start=('Date_Time','min'),t_end=('Date_Time','max'),
              n=('Voltage(V)','size')).reset_index()
    out['file']=os.path.basename(path); out['schedule']=meta.get('schedule','')
    out['file_start']=meta.get('start',pd.NaT)
    return out

def parse_cell(cell, root):
    files=glob.glob(os.path.join(root,cell,'*.xlsx'))
    parts=[parse_file(p) for p in files]
    parts=[p for p in parts if p is not None and len(p)]
    # sort by file start datetime; fallback filename
    parts.sort(key=lambda d: (str(d.file_start.iloc[0]), d.file.iloc[0]))
    offset=0; out=[]
    for d in parts:
        d=d.copy(); d['cycle_local']=d.Cycle_Index; d['cycle_global']=d.cycle_local+offset
        offset=int(d.cycle_global.max()); out.append(d)
    d=pd.concat(out,ignore_index=True)
    d['cell']=cell
    return d

if __name__=='__main__':
    cell,root,out=sys.argv[1:4]
    d=parse_cell(cell,root); d.to_csv(out,index=False)
    print(d.shape); print(d.head(4).to_string()); print(d.tail(4).to_string())
    print('schedules:'); print(d.groupby('schedule').agg(n=('cap_dis','size'),cmin=('cap_dis','min'),cmax=('cap_dis','max'),cyc0=('cycle_global','min'),cyc1=('cycle_global','max')).to_string())
