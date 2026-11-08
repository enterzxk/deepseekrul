import json, os
root = "experiments/punet_e2e"
subs = ["FD001","FD002","FD003","FD004"]
names = [("constant-width", "e2e_point_none"), ("QR (no calib)", "e2e_qr_raw_none"),
         ("PU-CQR (ours)", "e2e_cqr_none"), ("MC-Dropout", "mcdropout")]
print("%-16s | %-8s %-8s %-8s %-8s" % ("method", "FD001", "FD002", "FD003", "FD004"))
for label, fn in names:
    row = []
    for s in subs:
        p = "%s/%s/f0_s42/%s.json" % (root, s, fn)
        row.append("%.1f" % json.load(open(p))["winkler"] if os.path.exists(p) else "-")
    print("%-16s | %-8s %-8s %-8s %-8s" % (label, *row))
print()
print("%-16s | %-8s %-8s %-8s %-8s" % ("PICP", "FD001", "FD002", "FD003", "FD004"))
for label, fn in names:
    row = []
    for s in subs:
        p = "%s/%s/f0_s42/%s.json" % (root, s, fn)
        row.append("%.3f" % json.load(open(p))["picp"] if os.path.exists(p) else "-")
    print("%-16s | %-8s %-8s %-8s %-8s" % (label, *row))
