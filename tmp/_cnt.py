
import glob, os
for sub in ["FD001","FD002","FD003","FD004"]:
    tr = f"data/train_{sub}.txt"; te = f"data/test_{sub}.txt"; ru = f"data/RUL_{sub}.txt"
    def stats(p):
        if not os.path.exists(p): return None
        ids=set(); n=0
        for line in open(p):
            line=line.strip()
            if not line: continue
            n+=1; ids.add(line.split()[0])
        return len(ids), n
    s1=stats(tr); s2=stats(te); 
    rul = sum(1 for l in open(ru) if l.strip()) if os.path.exists(ru) else 0
    print(sub, "train(engines,rows)=",s1, "test=",s2, "RUL lines=",rul)

