import pandas as pd, numpy as np
from scipy.stats import spearmanr
d = pd.read_csv("experiments/punet_e2e/FD001/f0_s42/preds.csv")
err = np.abs(d.y_true - (d.y_lo + d.y_hi) / 2)
print("n:", len(d))
print("spearman(sigma, |err|) =", round(float(spearmanr(d.sigma_T, err).statistic), 3))
print("spearman(mu,    |err|) =", round(float(spearmanr(d.mu_T, err).statistic), 3))
print("spearman(sigma, width) =", round(float(spearmanr(d.sigma_T, d.y_hi - d.y_lo).statistic), 3))
print("sigma range:", round(d.sigma_T.min(), 2), round(d.sigma_T.max(), 2))
