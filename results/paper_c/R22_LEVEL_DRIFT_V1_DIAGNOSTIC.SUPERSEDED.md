# SUPERSEDED - do not cite

The per-cell dlev values in R22_LEVEL_DRIFT_V1_DIAGNOSTIC.csv were computed from the v1
products, which were trained with the wrong data module.

open_source/data.py:19 uses points = range(length, n + 1), dropping every engine's first
window - 1 windows. The anchor used open_source_matched_20261112/data.py:19,
points = range(1, n + 1). Per cell that is 80 * (window - 1) windows: 3920 for aidgn
(window 50), 3520 for two_phase (window 45), i.e. -23%.

Measured on aidgn/FD001/s101:

| run | n_train_windows | val at best |
|---|---:|---:|
| anchor (A40) | 16762 | 11.1004 |
| v1/v5/v7 (wrong data) | 12842 | 10.7021 |
| v8 (matched data) | 16762 | 11.1240 |

So |anchor - local| falls from 0.398 to 0.024 once the data module is right. The 0.2-0.5
"environment drift" reported here was mostly a 23% smaller training set.

**Every number in the CSV is void as a drift reading.** The authoritative record will be
econ_A.json produced from out_v8.

What survives is the set-membership observation, not the magnitudes: an exclusion list built
from "this cell cannot be reproduced on this machine" does not coincide with the set that
carries the offset, so removing 9 cells does not remove the offset.
