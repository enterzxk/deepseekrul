import glob, json
print(f"{'cell':16s} {'graph':>6s} {'dual':>5s} {'reliab':>7s} {'gate_mode':>10s} {'params':>8s}")
seen = set()
for mf in sorted(glob.glob('experiments/order03/*/FD002/*/metrics.json')):
    cell = mf.split('/')[2]
    if cell in seen: continue
    seen.add(cell)
    try:
        d = json.load(open(mf)); c = d.get('config', {})
    except Exception: continue
    print(f"{cell:16s} {str(c.get('use_graph')):>6s} {str(c.get('use_dual_head')):>5s} {str(c.get('use_reliability')):>7s} {str(c.get('gate_mode')):>10s} {str(d.get('params')):>8s}")
