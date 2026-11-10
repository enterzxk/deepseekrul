"""Causal spectral features from raw FEMTO vibration snapshots (25.6 kHz).

Each snapshot is a 0.1 s measurement (2560 samples, two channels). Every feature
below is computed from THAT snapshot only, so using a snapshot at time t never
sees the future.

Features (per channel h / v)
  rms, kurt, crest, skew
  band energy fractions in 5 bands up to Nyquist (12.8 kHz)
  envelope (Hilbert) band energies + envelope peak frequency
  h/v total energy ratio
  magnitude-squared coherence between channels (mean over 0-8 kHz)

Output: data/femto_spec/<bearing>.csv with one row per snapshot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

FS = 25600
BANDS = [(0, 1000), (1000, 2500), (2500, 5000), (5000, 8000), (8000, 12800)]


def feats(x: np.ndarray) -> dict:
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    n = len(x)
    X = np.fft.rfft(x * np.hanning(n))
    P = np.abs(X) ** 2
    f = np.fft.rfftfreq(n, 1.0 / FS)
    tot = float(P.sum() + 1e-12)
    out = dict(rms=rms, kurt=float(kurtosis(x)), crest=float(np.max(np.abs(x)) / rms),
               skew=float(skew(x)))
    for i, (lo, hi) in enumerate(BANDS):
        m = (f >= lo) & (f < hi)
        out[f"band{i}"] = float(P[m].sum() / tot)
    # envelope spectrum
    env = np.abs(hilbert(x))
    E = np.abs(np.fft.rfft(env - env.mean()))
    Ef = np.fft.rfftfreq(len(env), 1.0 / FS)
    m = (Ef > 20) & (Ef < 2000)
    out["env_peak_f"] = float(Ef[m][np.argmax(E[m])]) if m.any() else 0.0
    out["env_energy"] = float(np.sum(E[m] ** 2) / (np.sum(E ** 2) + 1e-12))
    out["_P"] = P
    out["_f"] = f
    return out


def bearing_features(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("acc_*.csv"))
    rows = []
    for i, fp in enumerate(files):
        try:
            d = pd.read_csv(fp, header=None, sep=None, engine="python").to_numpy(np.float64)
        except Exception:
            continue
        h = feats(d[:, 4]); v = feats(d[:, 5])
        row = {"snapshot": i}
        for k, val in h.items():
            if not k.startswith("_"):
                row[f"h_{k}"] = val
        for k, val in v.items():
            if not k.startswith("_"):
                row[f"v_{k}"] = val
        row["hv_energy_ratio"] = float(h["rms"] ** 2 / (v["rms"] ** 2 + 1e-12))
        # coherence between channels, 0-8 kHz
        Ph, Pv = h["_P"], v["_P"]
        f = h["_f"]
        m = (f > 0) & (f < 8000)
        Xh = np.fft.rfft((d[:, 4] - d[:, 4].mean()) * np.hanning(len(d)))
        Xv = np.fft.rfft((d[:, 5] - d[:, 5].mean()) * np.hanning(len(d)))
        coh = np.abs(Xh * np.conj(Xv)) ** 2 / ((np.abs(Xh) ** 2) * (np.abs(Xv) ** 2) + 1e-12)
        row["coh_mean"] = float(np.mean(coh[m])) if m.any() else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/yndx/femto")
    ap.add_argument("--out", default="data/femto_spec")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for base in ["Learning_set", "Full_Test_Set"]:
        d = root / base
        if not d.exists():
            continue
        for b in sorted(p.name for p in d.iterdir() if p.is_dir()):
            name = f"{base[:4]}_{b}"
            dst = out / f"{name}.csv"
            if dst.exists():
                print(f"skip {name}")
                continue
            df = bearing_features(d / b)
            df.to_csv(dst, index=False)
            print(f"{name}: {len(df)} snapshots -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
