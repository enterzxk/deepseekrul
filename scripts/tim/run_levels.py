"""Level sweep: do the two width conventions keep their ordering away from 0.90?

Reads matrix.json (0.90, 0.50) plus matrix_lv80.json / matrix_lv95.json and
reports, per fleet and per nominal level:

  layer central -> shortest   the convention cost of a symmetric tail split
  EAC vs CV+                  the matched-object gain

Both are relative width changes, so they are comparable across levels where the
absolute widths are not.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C

OUT = C.ROOT / "experiments" / "tim"
SUBSETS = C.SUBSETS


def load():
    base = json.loads((OUT / "matrix.json").read_text())
    files = {0.90: base, 0.50: base}
    for lv, name in ((0.80, "matrix_lv80.json"), (0.95, "matrix_lv95.json")):
        p = OUT / name
        if p.exists():
            files[lv] = json.loads(p.read_text())
    return files


def get(doc, subset, key):
    row = doc["subsets"][subset].get(key)
    return None if row is None else row.get("width")


def main():
    docs = load()
    levels = sorted(docs)
    print("levels:", levels)

    tab = {}
    for lv in levels:
        doc = docs[lv]
        tag = f"{lv:.2f}"
        tab[tag] = {}
        for s in SUBSETS:
            cen = get(doc, s, f"layer@{tag}")
            sht = get(doc, s, f"layer-short@{tag}")
            eac = get(doc, s, f"EAC@{tag}")
            cvp = get(doc, s, f"CV+-sym@{tag}")
            row = {}
            if cen and sht:
                row["conv"] = 100.0 * (cen - sht) / cen          # >0: shortest is narrower
            if eac and cvp:
                row["match"] = 100.0 * (cvp - eac) / cvp         # >0: EAC is narrower
            if eac and cen:
                row["vs_layer"] = 100.0 * (cen - eac) / cen
            row["w_layer_c"] = cen
            row["w_layer_s"] = sht
            row["w_eac"] = eac
            row["w_cvp"] = cvp
            tab[tag][s] = row

    hdr = f"{'level':>5} {'fleet':>6} {'cen':>7} {'sht':>7} {'EAC':>7} {'CV+':>7} " \
          f"{'conv%':>7} {'match%':>7} {'vsLayer%':>9}"
    print(hdr)
    for lv in levels:
        tag = f"{lv:.2f}"
        for s in SUBSETS:
            r = tab[tag][s]
            print(f"{tag:>5} {s:>6} {r['w_layer_c']:7.1f} {r['w_layer_s']:7.1f} "
                  f"{r['w_eac']:7.1f} {r['w_cvp']:7.1f} "
                  f"{r['conv']:7.1f} {r['match']:7.1f} {r['vs_layer']:9.1f}")
        conv = [tab[tag][s]["conv"] for s in SUBSETS]
        mat = [tab[tag][s]["match"] for s in SUBSETS]
        print(f"{tag:>5} {'mean':>6} {'':>7} {'':>7} {'':>7} {'':>7} "
              f"{sum(conv)/4:7.1f} {sum(mat)/4:7.1f}")
        print()

    (OUT / "levels.json").write_text(json.dumps(tab, indent=1))
    print("wrote", OUT / "levels.json")


if __name__ == "__main__":
    main()
