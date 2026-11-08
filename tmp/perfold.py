import json, glob, os
for sub in ["FD001", "FD002", "FD003", "FD004"]:
    for v in ["none", "A4"]:
        rows = []
        for f in range(5):
            p = "experiments/punet_e2e/%s/f%d_s42/e2e_cqr_%s.json" % (sub, f, v)
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            rows.append("f%d:W%.1f/P%.3f" % (f, d["winkler"], d["picp"]))
        if rows:
            print(sub, v, " ".join(rows))
