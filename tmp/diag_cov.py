import json, glob
for sub in ["FD001", "FD002", "FD003", "FD004"]:
    files = glob.glob("experiments/punet_e2e/%s/f0_s42/e2e_cqr_none.json" % sub)
    if not files:
        print(sub, "missing")
        continue
    d = json.load(open(files[0]))
    print(sub, "picp", round(d["picp"], 3), "q_hat", round(d["q_hat"], 2), "pinaw", round(d["pinaw"], 3))
    for b in d["diagnostics"]["stage_bins"]:
        print("   tau", [round(x, 2) for x in b["tau"]], "n", b["n"], "cov", round(b["coverage"], 3))
