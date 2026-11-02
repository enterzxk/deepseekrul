"""What the interval width is worth at the maintenance decision.

A planner with a maintenance window of T cycles acts on the upper bound: if the engine's
upper bound falls inside the window, the engine is flagged for the next opportunity.
Two errors follow, and they are not symmetric in cost:

  late     -- the engine is not flagged and fails before T
  false    -- the engine is flagged and would have lasted past T

A band that is too wide inflates the upper bound, so it flags less often and is late more
often. That is the decision-level form of the over-coverage reported in the matrix: the
symmetric band covers more of the distribution and, for exactly that reason, is a worse
alarm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C
from run_extras import build, intervals

OUT = C.ROOT / "experiments" / "tim"
HORIZONS = (20, 30, 40)


def main():
    rep = {}
    for s in C.SUBSETS:
        for T in HORIZONS:
            for k in ("EAC", "EAC-sym", "layer"):
                late, false, flag, wid = [], [], [], []
                for seed in C.SEEDS:
                    d = build(s, seed)
                    lo, hi = intervals(d)[k]
                    y = d["y"]
                    f = hi < T
                    late.append(float(np.mean((~f) & (y < T))))
                    false.append(float(np.mean(f & (y >= T))))
                    flag.append(float(np.mean(f)))
                    wid.append(float(np.mean(hi - lo)))
                rep[f"{s}|{T}"] = rep.get(f"{s}|{T}", {})
                rep[f"{s}|{T}"][k] = dict(late=float(np.mean(late)), late_sd=float(np.std(late)),
                                          false=float(np.mean(false)), flag=float(np.mean(flag)),
                                          width=float(np.mean(wid)))
            e, sy = rep[f"{s}|{T}"]["EAC"], rep[f"{s}|{T}"]["EAC-sym"]
            print(f"{s} T={T}: late EAC {e['late']:.3f} vs sym {sy['late']:.3f} "
                  f"(x{sy['late']/max(e['late'],1e-9):.2f}); flag {e['flag']:.2f} vs {sy['flag']:.2f}")
    (OUT / "decision.json").write_text(json.dumps(rep, indent=1))
    print("wrote", OUT / "decision.json")


if __name__ == "__main__":
    main()
