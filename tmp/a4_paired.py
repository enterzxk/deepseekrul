import json, os
subs = ["FD001", "FD002", "FD003", "FD004"]
print("subset | CQR_f0 | A4_f0 | delta_pct")
for s in subs:
    a = "experiments/punet_e2e/%s/f0_s42/e2e_cqr_none.json" % s
    b = "experiments/punet_e2e/%s/f0_s42/e2e_cqr_A4.json" % s
    if not (os.path.exists(a) and os.path.exists(b)):
        continue
    da, db = json.load(open(a)), json.load(open(b))
    d = (db["winkler"] - da["winkler"]) / da["winkler"] * 100
    print("%s | %.1f | %.1f | %+.1f%%" % (s, da["winkler"], db["winkler"], d))
