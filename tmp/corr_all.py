import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
for sub in ["FD001","FD002","FD003","FD004"]:
    p = "experiments/punet_e2e/%s/f0_s42/preds.csv" % sub
    if not os.path.exists(p):
        print(sub, "missing"); continue
    d = pd.read_csv(p)
    err = np.abs(d.y_true - (d.y_lo + d.y_hi)/2)
    print("%s n=%d | rho(sigma,err)=%+.3f rho(mu,err)=%+.3f rho(sigma,width)=%+.3f" % (
        sub, len(d),
        spearmanr(d.sigma_T, err).statistic,
        spearmanr(d.mu_T, err).statistic,
        spearmanr(d.sigma_T, d.y_hi - d.y_lo).statistic))
