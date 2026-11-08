import pandas as pd, numpy as np
df = pd.read_csv("docs/tables/cells_all.csv")
print("cells:", len(df), "| seeds:", sorted(df.seed.unique()))
for base in ["const", "cqr", "mcd"]:
    pass
rel = (df["cqr"] - df["const"]) / df["const"] * 100
print("\nCQR vs const: wins %d/%d (%.0f%%) mean %+.1f%% median %+.1f%%" % (
    int((rel<0).sum()), len(rel), 100*(rel<0).mean(), rel.mean(), rel.median()))
print("\nper subset:")
for s in ["FD001","FD002","FD003","FD004"]:
    d = df[df.subset==s]
    r = (d["cqr"] - d["const"]) / d["const"] * 100
    print("  %s n=%2d wins %2d mean %+6.1f%% median %+6.1f%%" % (s, len(d), int((r<0).sum()), r.mean(), r.median()))
print("\nper seed:")
for sd in sorted(df.seed.unique()):
    d = df[df.seed==sd]
    r = (d["cqr"] - d["const"]) / d["const"] * 100
    print("  seed %-4d n=%2d wins %2d mean %+6.1f%%" % (sd, len(d), int((r<0).sum()), r.mean()))
# bootstrap CI on the mean, engines not available here so cell-level
rng = np.random.default_rng(0)
boot = [rel.sample(len(rel), replace=True, random_state=int(rng.integers(1e6))).mean() for _ in range(2000)]
print("\ncell-level bootstrap 95%% CI for mean: [%.1f%%, %.1f%%]" % (np.percentile(boot,2.5), np.percentile(boot,97.5)))
