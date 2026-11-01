"""Collapse N-CMAPSS timestamps to flight cycles and emit compact windowed arrays.

The raw h5 is 2-4 GB per fleet because every flight cycle carries 6,000-13,000
timestamps. Treating those as independent samples would inflate n by four orders of
magnitude and make every confidence interval meaningless, so the frozen protocol
aggregates to cycles first.

Output per fleet: `<fleet>_dev.npz` and `<fleet>_test.npz` with

    X     [n, 54]  mean/std/delta of 18 cycle features over a 10-cycle window
    y     [n]      RUL at the target cycle
    unit  [n]      unit id, the bootstrap block
    cyc   [n]      cycle index of the window end

Grouping uses np.bincount over a cycle-group id rather than a Python loop over units:
the latter costs one full pass over 6M rows per unit and there are nine of them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

L = 10          # window length, frozen in paper/NCMAPSS_PROTOCOL.md
FEATS = 18      # 14 sensors + 4 flight conditions


def cycle_table(h, split):
    """Return (unit, cycle, features, y, within-cycle y std) at cycle resolution."""
    A = np.array(h[f"A_{split}"], dtype=np.float64)
    Y = np.array(h[f"Y_{split}"], dtype=np.float64).ravel()
    Xs = np.array(h[f"X_s_{split}"], dtype=np.float64)
    W = np.array(h[f"W_{split}"], dtype=np.float64)

    avar = [x.decode() if isinstance(x, bytes) else str(x) for x in np.array(h["A_var"])]
    iu, ic = avar.index("unit"), avar.index("cycle")

    key = A[:, [iu, ic]]
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    n = len(first)

    M = np.concatenate([Xs, W], axis=1)          # [N, 18]
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    feat = np.empty((n, FEATS), dtype=np.float64)
    for j in range(FEATS):
        feat[:, j] = np.bincount(inv, weights=M[:, j], minlength=n) / cnt

    y = np.bincount(inv, weights=Y, minlength=n) / cnt
    y2 = np.bincount(inv, weights=Y * Y, minlength=n) / cnt
    ysd = np.sqrt(np.maximum(y2 - y * y, 0.0))

    unit = key[first, 0]
    cyc = key[first, 1]
    return unit.astype(int), cyc.astype(int), feat, y, ysd, len(A)


def windows(unit, cyc, feat, y, L=L):
    """Expand each unit's cycle series into L-length windows ending at each cycle."""
    Xs, ys, us, cs = [], [], [], []
    for u in np.unique(unit):
        m = unit == u
        f, yy = feat[m], y[m]                      # already sorted by cycle
        for t in range(L - 1, len(f)):
            w = f[t - L + 1:t + 1]
            Xs.append(np.concatenate([w.mean(0), w.std(0), w[-1] - w[0]]))
            ys.append(yy[t])
            us.append(u)
            cs.append(cyc[m][t])
    return (np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32),
            np.asarray(us, dtype=np.int32), np.asarray(cs, dtype=np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5dir", default="C:/Users/20562/AppData/Local/Temp/ncmapss/extracted")
    ap.add_argument("--out", default="experiments/ncmapss/data")
    ap.add_argument("--fleets", nargs="*", default=None)
    a = ap.parse_args()

    src, out = Path(a.h5dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("N-CMAPSS_DS*.h5"))
    # DS08d is excluded on data-integrity grounds, not convenience: its zip member is
    # 2,885,034,848 bytes while the HDF5 superblock records 2,885,034,880, so h5py refuses
    # it. CRC and declared size both pass, so this is an upstream defect. Zero-padding it
    # would produce a file indistinguishable from the original, so it is dropped.
    files = [f for f in files if "DS08d" not in f.name]
    if a.fleets:
        files = [f for f in files if any(k in f.name for k in a.fleets)]

    summary = {}
    for f in files:
        # DS01-005, DS04, DS08a-009 -> DS01, DS04, DS08a
        import re
        name = re.match(r"N-CMAPSS_(DS\d+[a-z]?)", f.name).group(1)
        if name in summary:
            continue
        t0 = time.time()
        rec = {}
        with h5py.File(f, "r") as h:
            for split in ["dev", "test"]:
                unit, cyc, feat, y, ysd, nrows = cycle_table(h, split)
                X, yy, uu, cc = windows(unit, cyc, feat, y)
                np.savez_compressed(out / f"{name}_{split}.npz",
                                    X=X, y=yy, unit=uu, cyc=cc)
                rec[split] = {
                    "timestamps": int(nrows), "cycles": int(len(unit)),
                    "units": int(len(np.unique(unit))), "samples": int(len(yy)),
                    "retained_frac": round(len(yy) / max(len(unit), 1), 4),
                    "y_min": round(float(yy.min()), 2), "y_max": round(float(yy.max()), 2),
                    "y_mean": round(float(yy.mean()), 2),
                    "within_cycle_ysd": round(float(ysd.mean()), 3),
                    "capped_share": round(float(np.mean(yy >= 125)), 4),
                }
                print(f"{name} {split:4s} ts={nrows:>9d} cyc={len(unit):>4d} "
                      f"units={len(np.unique(unit)):>2d} samples={len(yy):>4d} "
                      f"y∈[{yy.min():.0f},{yy.max():.0f}] "
                      f"within-cyc sd={ysd.mean():.2f}")
        rec["seconds"] = round(time.time() - t0, 1)
        rec["h5_bytes"] = f.stat().st_size
        summary[name] = rec
        print(f"   -> {rec['seconds']}s\n")

    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {len(summary)} fleets to {out}")


if __name__ == "__main__":
    main()
