"""Extract per-snapshot vibration features from the FEMTO/PRONOSTIA dataset.

Each bearing directory holds ~600-3300 CSV snapshots (0.1 s at 25.6 kHz, two
channels: horizontal and vertical acceleration). We summarise every snapshot by
a small set of classical condition-monitoring indicators, which become the
"sensor vector" for the RUL pipeline.

Output: data/femto_features/<bearing>.csv with one row per snapshot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def snapshot_features(path: Path) -> dict | None:
    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        return None
    if df.shape[1] < 6 or df.shape[0] < 16:
        return None
    h = df.iloc[:, 4].to_numpy(dtype=np.float64)
    v = df.iloc[:, 5].to_numpy(dtype=np.float64)
    feats = {}
    for name, x in (("h", h), ("v", v)):
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)))
        feats[f"{name}_rms"] = rms
        feats[f"{name}_kurt"] = float(kurtosis(x, fisher=False))
        feats[f"{name}_skew"] = float(skew(x))
        feats[f"{name}_peak"] = peak
        feats[f"{name}_crest"] = peak / rms if rms > 1e-12 else 0.0
    feats["h_v_ratio"] = feats["h_rms"] / feats["v_rms"] if feats["v_rms"] > 1e-12 else 0.0
    return feats


def process_bearing(bdir: Path, out_dir: Path) -> int:
    files = sorted(bdir.glob("acc_*.csv"))
    rows = []
    for f in files:
        feats = snapshot_features(f)
        if feats is None:
            continue
        rec = {"snapshot": len(rows)}
        rec.update(feats)
        rows.append(rec)
    out = out_dir / f"{bdir.name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "femto"))
    ap.add_argument("--out", default="data/femto_features")
    ap.add_argument("--sets", nargs="+", default=["Learning_set", "Full_Test_Set"])
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in args.sets:
        d = root / s
        if not d.exists():
            print("missing", d)
            continue
        for b in sorted(d.iterdir()):
            if not b.is_dir():
                continue
            n = process_bearing(b, out_dir)
            print(f"{s}/{b.name}: {n} snapshots", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
