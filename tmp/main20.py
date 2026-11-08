import json, os, statistics as st
root = "experiments/punet_e2e"
subs = ["FD001","FD002","FD003","FD004"]
def load(s,f,n):
    p = "%s/%s/f%d_s42/%s.json" % (root,s,f,n)
    return json.load(open(p)) if os.path.exists(p) else None
print("== main 20 folds (delta=0.1) ==")
for s in subs:
    r = [load(s,f,"e2e_cqr_none") for f in range(5)]
    r = [x for x in r if x]
    print("%s PICP %.3f+-%.3f Winkler %.1f+-%.1f PINAW %.3f engmin %.3f" % (
        s, st.mean(x["picp"] for x in r), st.stdev(x["picp"] for x in r),
        st.mean(x["winkler"] for x in r), st.stdev(x["winkler"] for x in r),
        st.mean(x["pinaw"] for x in r), st.mean(x["engine_min"] for x in r)))
d = []
for s in subs:
    for f in range(5):
        a, b = load(s,f,"e2e_cqr_none"), load(s,f,"e2e_point_none")
        if a and b: d.append((a["winkler"]-b["winkler"])/b["winkler"]*100)
print("paired n=%d wins=%d (%.0f%%) mean %+.1f%% median %+.1f%%" % (
    len(d), sum(1 for x in d if x<0), 100*sum(1 for x in d if x<0)/len(d), st.mean(d), st.median(d)))
