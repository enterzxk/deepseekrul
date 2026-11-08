import json, os, statistics as st
root = "experiments/punet_e2e"
subs = ["FD001","FD002","FD003","FD004"]
def load(s,f,n):
    p = "%s/%s/f%d_s42/%s.json" % (root,s,f,n)
    return json.load(open(p)) if os.path.exists(p) else None
d = []
for s in subs:
    for f in range(5):
        a, b = load(s,f,"e2e_cqr_none"), load(s,f,"e2e_cqr_A4")
        if a and b:
            d.append((s,f,(b["winkler"]-a["winkler"])/a["winkler"]*100))
print("A4 folds available:", len(d))
for s,f,x in d:
    print("  %s f%d %+.1f%%" % (s,f,x))
if d:
    worse = sum(1 for _,_,x in d if x > 0)
    print("A4 worse: %d/%d | mean %+.1f%% | median %+.1f%%" % (
        worse, len(d), st.mean(x for _,_,x in d), st.median(x for _,_,x in d)))
