import json, glob
rows = []
for f in sorted(glob.glob("experiments/*/*/FD002/*/metrics.json")):
    m = json.load(open(f))
    parts = f.split("/")
    cmd = m.get("command") or ""
    runner = "frozen" if "frozen" in cmd else ("current" if cmd else "unknown")
    rows.append((parts[1], parts[2], runner, m.get("config", {}).get("ref_mode", "-"), round(m.get("matched_cutoff_rmse", float('nan')), 2)))
seen = {}
for r in rows:
    seen.setdefault(r[2], []).append(r)
for k, v in seen.items():
    print(f"--- runner={k}: {len(v)} runs")
    for r in v[:8]:
        print("   ", r[0], r[1], "ref=", r[3], "matched=", r[4])
