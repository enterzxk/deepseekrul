import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS

df = load_cmapss_train(Path("data/CMAPSS"), "FD002")
cols = list(SELECTED_SENSOR_COLUMNS)
fired, spans, fr = 0, [], []
for u, g in df.groupby("unit"):
    g = g.sort_values("cycle")
    t = g.cycle.to_numpy(float); x = g[cols].to_numpy(float); n = len(g)
    w = max(5, n // 20)
    mu = x[:w].mean(0); sd = x[:w].std(0) + 1e-9
    hi = ((x - mu) / sd).mean(1)
    k = max(5, n // 20)
    sm = np.convolve(hi, np.ones(k) / k, mode="valid")
    pad = n - len(sm); sm = np.concatenate([np.full(pad // 2, sm[0]), sm,
                                            np.full(pad - pad // 2, sm[-1])])
    base = sm[: max(5, n // 10)]
    thr = base.mean() + 3.0 * (base.std() + 1e-9)
    ab = np.where(sm > thr)[0]
    if ab.size:
        fired += 1
        cp = int(ab[0])
        spans.append(t[-1] - t[min(max(cp, 5), n - 1)])
        fr.append(cp / n)
    else:
        spans.append(np.nan); fr.append(np.nan)
spans = np.array(spans); fr = np.array(fr)
ok = ~np.isnan(spans)
print(f"fired {fired}/{len(spans)} ({100*fired/len(spans):.1f}%)")
if fired:
    print(f"  onset fraction of life: mean={np.nanmean(fr):.2f} median={np.nanmedian(fr):.2f} range=[{np.nanmin(fr):.2f},{np.nanmax(fr):.2f}]")
    print(f"  span T-t_cp: mean={np.nanmean(spans):.1f} median={np.nanmedian(spans):.1f}")
    print(f"  cap(125) binds in {100*(np.nanmean(spans)<125):.1f}% of fired engines")
