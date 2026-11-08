from pathlib import Path
p=Path('make_paper_assets.py'); s=p.read_text()
old = """            rmses = [pooled(t)[\"rmse\"] for t in per_seed.values()]
            first = pooled(next(iter(per_seed.values())))
            row = dict(subset=subset, arm=arm, n_seeds=len(per_seed),
                       rmse_mean=float(np.mean(rmses)), rmse_sd=float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
                       mae=first[\"mae\"], nasa=first[\"nasa\"], bias=first[\"bias\"],
                       engines=first[\"engines\"], windows=first[\"windows\"])"""
new = """            mets = [pooled(t) for t in per_seed.values()]
            rmses = [m[\"rmse\"] for m in mets]
            first = mets[0]
            row = dict(subset=subset, arm=arm, n_seeds=len(per_seed),
                       rmse_mean=float(np.mean(rmses)), rmse_sd=float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
                       mae=float(np.mean([m[\"mae\"] for m in mets])),
                       nasa=float(np.mean([m[\"nasa\"] for m in mets])),
                       bias=float(np.mean([m[\"bias\"] for m in mets])),
                       engines=first[\"engines\"], windows=first[\"windows\"])"""
if old in s:
    s = s.replace(old, new); p.write_text(s); print('patched')
else:
    print('anchor missing')