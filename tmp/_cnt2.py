
import os, glob
base = "data/CMAPSS"
files = glob.glob(base + "/**/*", recursive=True)
print("sample files:", [f for f in files if f.lower().endswith(".txt")][:8])
def stats(p):
    ids=set(); n=0
    for line in open(p, errors="ignore"):
        line=line.strip()
        if not line: continue
        n+=1; ids.add(line.split()[0])
    return len(ids), n
for sub in ["FD001","FD002","FD003","FD004"]:
    tr=[f for f in files if f.lower().endswith(f"train_{sub.lower()}.txt")]
    te=[f for f in files if f.lower().endswith(f"test_{sub.lower()}.txt")]
    ru=[f for f in files if f.lower().endswith(f"rul_{sub.lower()}.txt")]
    if tr and te:
        print(sub, "train=", stats(tr[0]), "test=", stats(te[0]), "rul_lines=", (sum(1 for l in open(ru[0]) if l.strip()) if ru else None))
    else:
        print(sub, "missing", tr, te)

