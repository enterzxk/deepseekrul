import json, os, statistics as st
root = "experiments/punet_e2e"
subs = ["FD001","FD002","FD003","FD004"]
def load(s,f,n):
    p = "%s/%s/f%d_s42/%s.json" % (root,s,f,n)
    return json.load(open(p)) if os.path.exists(p) else None

main = [(s,f) for s in subs for f in range(5) if load(s,f,"e2e_cqr_none")]
base = [(s,f) for s in subs for f in range(5) if load(s,f,"e2e_point_none")]
a4   = [(s,f) for s in subs for f in range(5) if load(s,f,"e2e_cqr_A4")]
print("counts main/base/A4: %d/%d/%d" % (len(main), len(base), len(a4)))

picps = [load(s,f,"e2e_cqr_none")["picp"] for s,f in main]
print("mean PICP %.3f | folds <0.88: %d/%d" % (st.mean(picps), sum(1 for p in picps if p<0.88), len(picps)))

d = []
for s,f in main:
    a, b = load(s,f,"e2e_cqr_none"), load(s,f,"e2e_point_none")
    if a and b: d.append((a["winkler"]-b["winkler"])/b["winkler"]*100)
print("paired: n=%d wins=%d (%.0f%%) mean %+.1f%% median %+.1f%%" % (
    len(d), sum(1 for x in d if x<0), 100*sum(1 for x in d if x<0)/len(d), st.mean(d), st.median(d)))

ad = []
for s,f in main:
    a, b = load(s,f,"e2e_cqr_none"), load(s,f,"e2e_cqr_A4")
    if a and b: ad.append((b["winkler"]-a["winkler"])/a["winkler"]*100)
print("A4: n=%d worse=%d mean %+.1f%% median %+.1f%%" % (
    len(ad), sum(1 for x in ad if x>0), st.mean(ad), st.median(ad)))
