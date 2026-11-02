"""Side-by-side of the two backbones on the same constructions.

Reads experiments/tim/matrix.json (primary backbone) and matrix_B.json (second backbone)
and prints, per fleet, the coverage and width of the constructions that the paper's claims
depend on, plus the residual scale of each backbone so the reader can judge whether the
second one is a competent predictor or a straw man.

Writes experiments/tim/backbones.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C  # noqa: E402

OUT = C.ROOT / "experiments" / "tim"
METHODS = ("layer", "layer-short", "CV+-sym", "CV+-asym", "EAC-sym", "EAC-eq", "EAC")


def rmse_of(root, tag, subset):
    import csv

    d = Path(root) / "cal" / "dev_preds" / subset
    ys, ps = [], []
    for f in sorted(d.glob(f"{tag}_s*_f*.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                ys.append(float(r["y_true"]))
                ps.append(float(r["y_pred"]))
    ys, ps = np.array(ys), np.array(ps)
    return float(np.sqrt(np.mean((ys - ps) ** 2))), float(np.abs(ys - ps).mean()), len(ys)


def main():
    a = json.loads((OUT / "matrix.json").read_text())["subsets"]
    bb_path = OUT / "matrix_B.json"
    b = json.loads(bb_path.read_text())["subsets"]

    report = {"methods": list(METHODS), "subsets": {}}
    print(f"{'':6s}{'':14s}" + "".join(f"{s:>20s}" for s in C.SUBSETS))
    for lv in ("0.90", "0.50"):
        print(f"\n=== nominal {lv}   (coverage / width)")
        for m in METHODS:
            row = f"{m:20s}"
            for s in C.SUBSETS:
                va = a[s].get(f"{m}@{lv}")
                vb = b[s].get(f"{m}@{lv}")
                ca = f"{va['cov']:.3f}/{va['width']:5.1f}" if va else "   -   "
                cb = f"{vb['cov']:.3f}/{vb['width']:5.1f}" if vb else "   -   "
                row += f"  {ca} | {cb}"
            print(row)

    for s in C.SUBSETS:
        ra = rmse_of(C.ROOT / "experiments", "A", s)
        rb = rmse_of(OUT / "bb", "B", s)
        report["subsets"][s] = dict(A=dict(rmse=ra[0], mae=ra[1], n=ra[2]),
                                    B=dict(rmse=rb[0], mae=rb[1], n=rb[2]))
        print(f"{s}: A dev RMSE {ra[0]:5.2f} |r| {ra[1]:5.2f}   "
              f"B dev RMSE {rb[0]:5.2f} |r| {rb[1]:5.2f}")

    # width ratio of EAC to the layer, per backbone
    for s in C.SUBSETS:
        for tag, d in (("A", a), ("B", b)):
            e = d[s].get("EAC@0.90")
            l = d[s].get("layer@0.90")
            if e and l:
                print(f"{s} {tag}: EAC {e['width']:5.1f} vs layer {l['width']:5.1f} "
                      f"({100 * (e['width'] / l['width'] - 1):+.1f}%)")
    (OUT / "backbones.json").write_text(json.dumps(report, indent=1))
    print("wrote", OUT / "backbones.json")


if __name__ == "__main__":
    main()
