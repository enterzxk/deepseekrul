# E4 — Result: the pipeline reproduces exactly from a clean directory

Run 2026-10-29 23:43–00:20 UTC+8. Design: `paper/E4_PREREGISTRATION.md`
(sha `dc83ea8b…`, frozen before the run). Verdict: **PASS**, both parts, with
zero measurable drift.

| | reference `e3/A_s2027` | replay from `best_model.pt` | from scratch in `/home/yndx/zxk/e4_clean` | acceptance |
|---|---:|---:|---:|---|
| val RMSE | 12.5434 | — | **12.5434** | \|Δ\| ≤ 0.15 |
| matched RMSE | 13.08570 | **13.08570** | **13.08570** | \|Δ\| ≤ 0.15 |
| best epoch | 26 | — | **26** | — |
| max \|Δpred\| | — | **7.2 × 10⁻⁶** | — | ≤ 1e-3 |

ΔRMSE = **0.00000** in both directions; the replay's prediction difference
(7e-6) is float32 round-off in the forward pass, not a behavioural difference.

## What this establishes (G3)

1. **Code + weights → predictions is deterministic.** Reloading `best_model.pt`
   and re-evaluating reproduces the stored predictions to 7e-6.
2. **The whole run is reproducible from an empty directory**, with only `src/`,
   `scripts/` and a `data/` symlink: same split, same windows, same best epoch,
   same metrics. No hidden state in the working tree.
3. **The provenance patch is behaviour-neutral.** The reference run used runner
   `89aa90aa…`; the fresh run used `13bb6ede…` (the version that writes
   `prereg_sha` / `started_at` / `finished_at`). Results are bit-identical, which
   independently confirms the added fields do not touch training.

## Environment (recorded for §7)

```
host      YNDXTEA0034
python    3.9.25
torch     2.5.1+cu121
cuda      12.1
gpu       NVIDIA A40 (46 GB)
model     4c576971f32c915b…   (use_pe branch, default off)
runner    13bb6ede1f9c2d7f…   (provenance fields added)
data      official C-MAPSS training files only; test never opened
```

## Boundaries

* One seed (2027), one configuration (A), one machine. This is **determinism**,
  not cross-platform portability: `torch.backends.cudnn.deterministic=True` is set
  in `set_seed`, and a different GPU/driver may still differ at the last bits.
* The fresh run took 35 min wall on the A40 (~46 epochs with patience 20).
* The reference's `metrics.json` predates the provenance fields, so its
  `started_at` is absent; the E4 run is the first with full wall-clock bounds.

## Consequence

G3's "E4 passes" condition is satisfied for the A configuration. The paper can
state: *the reported run is reproducible end-to-end from a clean checkout on the
same host class*, with the code and environment hashes above — and nothing stronger.
