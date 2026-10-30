import json, glob
for root, tag in [("levelb","M4_racdr"), ("order03","M4_racdr")]:
    fs = sorted(glob.glob(f"experiments/{root}/{tag}/FD002/*/metrics.json"))
    if not fs: continue
    m = json.load(open(fs[0]))
    print("===", root, tag)
    print("command:", m.get("command"))
    cfg = m.get("config", {})
    print("config:", json.dumps(cfg, ensure_ascii=False)[:600])
    print("val_units:", len(m.get("val_units", [])), "best_val:", m.get("best_val_rmse"))
    print("matched:", m.get("matched_cutoff_rmse"), "protocolC:", m.get("protocol_c_rmse_by_cycles"))
