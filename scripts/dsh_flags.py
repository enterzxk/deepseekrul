import json, glob
for root, tag in [("levelb","M4_racdr"), ("levelb","M0_tf"), ("order03","M4_racdr"), ("order03","M6_tfA")]:
    fs = sorted(glob.glob(f"experiments/{root}/{tag}/FD002/*/metrics.json"))
    if not fs: print(f"{root}/{tag}: none"); continue
    cfg = json.load(open(fs[0])).get("config", {})
    flags = {k: v for k, v in cfg.items() if isinstance(v, bool)}
    print(f"{root}/{tag}: " + ", ".join(f"{k}={v}" for k, v in sorted(flags.items())))
