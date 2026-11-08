import sys, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from pathlib import Path
df = pd.read_csv("docs/tables/cells_all.csv")
print("cells:", len(df), "| per subset:", df.groupby("subset").size().to_dict())
print()
for base in ["const", "mcd", "cqr"]:
    sub = df.dropna(subset=[base, "loc"])
    rel = (sub["loc"] - sub[base]) / sub[base] * 100
    print("Loc vs %-5s: n=%2d wins=%2d (%.0f%%) mean %+.1f%% median %+.1f%%" % (
        base, len(sub), int((rel < 0).sum()), 100*(rel < 0).mean(), rel.mean(), rel.median()))
print()
print("per-subset Loc vs cqr and Loc vs const (mean rel %):")
for s in ["FD001","FD002","FD003","FD004"]:
    d = df[df.subset == s]
    r1 = ((d["loc"] - d["cqr"]) / d["cqr"] * 100).mean()
    r2 = ((d["loc"] - d["const"]) / d["const"] * 100).mean()
    r3 = ((d["cqr"] - d["const"]) / d["const"] * 100).mean()
    print("  %s loc-vs-cqr %+.1f%%  loc-vs-const %+.1f%%  cqr-vs-const %+.1f%%" % (s, r1, r2, r3))
print()
print("paired cqr vs const (matched coverage):")
rel = (df["cqr"] - df["const"]) / df["const"] * 100
print("  n=%d wins=%d (%.0f%%) mean %+.1f%% median %+.1f%%" % (
    len(rel), int((rel<0).sum()), 100*(rel<0).mean(), rel.mean(), rel.median()))
