"""Is the learned half-width actually varying, per subset?

If s(x) is nearly constant the conformal interval reduces to a constant-width
interval by construction, which would explain why CQR matches the constant
baseline on some subsets.
"""

import numpy as np
from pathlib import Path

root = Path("experiments/punet_matrix")
print("subset  n_cells  width_mean  width_std  cv(rel)  corr(s,|err|)")
for sub in ["FD001", "FD002", "FD003", "FD004"]:
    means, stds, cvs, rhos = [], [], [], []
    for seed in [42, 7]:
        for fold in range(5):
            p = root / sub / f"f{fold}_s{seed}" / "preds_cqr.csv"
            if not p.exists():
                continue
            c = np.genfromtxt(p, delimiter=",", names=True)
            w = (c["y_hi"] - c["y_lo"]) / 2.0
            centre = (c["y_hi"] + c["y_lo"]) / 2.0
            err = np.abs(c["y_true"] - centre)
            means.append(w.mean()); stds.append(w.std()); cvs.append(w.std() / max(w.mean(), 1e-9))
            if w.std() > 1e-9:
                rhos.append(float(np.corrcoef(w, err)[0, 1]))
    print("%s  %2d      %8.1f   %8.1f  %.3f    %+.3f" % (
        sub, len(means), np.mean(means), np.mean(stds), np.mean(cvs), np.mean(rhos) if rhos else float("nan")))
