import json, os, statistics as st
root = "experiments/punet_e2e"
subs = ["FD001","FD002","FD003","FD004"]
def load(s,f,n):
    p = "%s/%s/f%d_s42/%s.json" % (root,s,f,n)
    return json.load(open(p)) if os.path.exists(p) else None

print("== per-subset CQR (5 folds) ==")
allp = []
for s in subs:
    rows = [load(s,f,"e2e_cqr_none") for f in range(5)]
    rows = [r for r in rows if r]
    p = [r["picp"] for r in rows]; allp += p
    print("%s n=%d PICP %.3f+-%.3f Winkler %.1f+-%.1f" % (
        s, len(rows), st.mean(p), st.stdev(p),
        st.mean(r["winkler"] for r in rows), st.stdev(r["winkler"] for r in rows)))

print("\n== paired ==")
d = []
for s in subs:
    for f in range(5):
        a, b = load(s,f,"e2e_cqr_none"), load(s,f,"e2e_point_none")
        if a and b:
            d.append((a["winkler"]-b["winkler"])/b["winkler"]*100)
wins = sum(1 for x in d if x < 0)
print("folds=%d wins=%d (%.0f%%) mean=%+.1f%% median=%+.1f%% worst=%+.1f%% best=%+.1f%%" % (
    len(d), wins, 100*wins/len(d), st.mean(d), st.median(d), max(d), min(d)))
print("overall mean PICP across all folds = %.3f" % st.mean(allp))
print("folds with PICP<0.88: %d/%d" % (sum(1 for x in allp if x < 0.88), len(allp)))
