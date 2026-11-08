from pathlib import Path
base=Path('/home/user/rulb/tim_20261105/full_20261105/runs')
for s in ['FD002','FD004','FD001','FD003']:
    line=[]
    for f in range(5):
        d=base/s/str(f)
        arms=sorted(p.name.replace('_101.csv','') for p in d.glob('*_101.csv')) if d.is_dir() else []
        line.append('f%d[%s]'%(f,','.join(arms)))
    print(s, ' '.join(line))
