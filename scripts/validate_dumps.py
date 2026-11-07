"""Integrity check for the prediction dumps.

Guards against the failure mode we hit earlier: two processes writing the same
cell, which silently produces mismatched or truncated prediction files.
"""

import numpy as np, sys
from pathlib import Path
root = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/punet_matrix")
bad = []
ok = 0
for pq in sorted(root.glob("*/*/preds_cqr.csv")):
    d = pq.parent
    pp = d / "preds_point.csv"
    try:
        c = np.genfromtxt(pq, delimiter=",", names=True)
        if not pp.exists():
            print("partial:", d.relative_to(root)); continue
        p = np.genfromtxt(pp, delimiter=",", names=True)
    except Exception as e:
        bad.append((str(d), "read error " + str(e)[:60])); continue
    issues = []
    if len(c) != len(p):
        issues.append("len %d vs %d" % (len(c), len(p)))
    if len(c) > 0 and len(p) == len(c):
        if not np.allclose(c["y_true"], p["y_true"], atol=1e-6):
            issues.append("y_true mismatch")
        if not np.allclose(c["unit"], p["unit"]):
            issues.append("unit mismatch")
    for name, arr in (("cqr", c["y_lo"]), ("cqr_hi", c["y_hi"]), ("pt_lo", p["y_lo"])):
        if not np.all(np.isfinite(arr)):
            issues.append(name + " non-finite")
    if np.any(c["y_lo"] > c["y_hi"]):
        issues.append("cqr lo>hi")
    if issues:
        bad.append((str(d.relative_to(root)), "; ".join(issues)))
    else:
        ok += 1
print("valid cells:", ok, "| problems:", len(bad))
for d, msg in bad:
    print("  ", d, "->", msg)
