import json, glob
from pathlib import Path
for root, tag in [("levelb","M4_racdr"), ("levelb","M0_tf"), ("order03","M4_racdr"), ("order03","M0_raw_tf")]:
    fs = sorted(glob.glob(f"experiments/{root}/{tag}/FD002/*/metrics.json"))
    if not fs:
        print(f"{root}/{tag}: none"); continue
    ms = [json.load(open(f)) for f in fs]
    m = ms[0]
    cfg = m.get("config", {})
    keys = ["variant","params","d_model","num_layers","epochs","late_weight","ref_mode","norm_extras"]
    print(f"=== {root}/{tag} (n_runs={len(ms)})")
    print("   params:", m.get("params"), "| variant:", m.get("variant"))
    for k in ("d_model","num_layers","epochs","late_weight","ref_mode","norm_extras","features","use_extras","gate_mode"):
        if k in cfg: print(f"   {k}: {cfg[k]}")
    print("   config keys:", ",".join(sorted(cfg)[:14]))
