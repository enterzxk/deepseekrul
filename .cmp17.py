import json, glob, os
R = "/home/yndx/zxk/bearing-rul-pi/experiments"
for f in sorted(glob.glob(f"{R}/e17/*/FD002/*/metrics.json")):
    d = json.load(open(f)); arm = d["variant"]; fold = d["fold"]
    base = f"{R}/e11/{arm}_s2027_f{fold}/FD002/{arm}_s2027/metrics.json"
    if not os.path.exists(base):
        print(f"{arm} f{fold}  no e11 counterpart"); continue
    b = json.load(open(base))
    dv, bv = d["best_val_rmse"], b["best_val_rmse"]
    print(f"{arm:6s} f{fold}  e17={dv:.6f}  e11={bv:.6f}  identical={abs(dv-bv)<1e-9}  "
          f"params e17={d.get('num_params')} e11={b.get('num_params')}")
