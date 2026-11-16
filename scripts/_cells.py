"""Split-aware run-cell resolution.

`runs/` now holds every split side by side, and run ids share the
``<arm>-<subset>-s<seed>-`` prefix across splits. A prefix glob therefore
returns cells from more than one split, and picking ``sorted(...)[0]`` silently
picks whichever hash sorts first -- in this study that was a half-finished
confirmation cell with 2 engines instead of the 20-engine discovery cell.

Every analysis script must resolve cells through here:
  * filter by the split recorded in ``config.json`` (never by directory name)
  * refuse ambiguity (exactly one match, or raise)

``config.json`` is written at the end of training, so an in-flight cell has no
config and is correctly invisible here.
"""
from __future__ import annotations

import glob
import json
import os


def resolve(root, arm, subset, seed, split_seed):
    hits = sorted(glob.glob(os.path.join(str(root), f'{arm}-{subset}-s{seed}-*')))
    ok = []
    for h in hits:
        c = os.path.join(h, subset, f's{seed}')
        cfgp = os.path.join(c, 'config.json')
        if not os.path.exists(cfgp):
            continue
        try:
            cfg = json.load(open(cfgp))
        except Exception:
            continue
        if int(cfg.get('split_seed', -1)) == int(split_seed):
            ok.append(c)
    if len(ok) != 1:
        raise RuntimeError(
            f'{arm}-{subset}-s{seed} split{split_seed}: {len(ok)} matches '
            f'out of {len(hits)} prefix hits')
    return ok[0]


def try_resolve(root, arm, subset, seed, split_seed):
    try:
        return resolve(root, arm, subset, seed, split_seed)
    except RuntimeError:
        return None
