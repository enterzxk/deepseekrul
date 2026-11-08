import glob, os
for f in sorted(glob.glob('/home/user/rulb/logs/EMA_*.log')):
    txt = open(f).read().strip().split('\n')
    tail = [l for l in txt if 'DONE' in l or 'using EMA' in l]
    print(os.path.basename(f)[:-4], '|', ' ; '.join(tail) if tail else txt[-1])