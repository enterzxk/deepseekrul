# Paper B experiment registry

Status values: PLANNED / RUNNING / DONE / INVALID / ARCHIVED.
A run is DONE only when `paper_b/results/raw/<tag>/<subset>/s<seed>/metrics.json`
exists; a test number is DONE only when `test_metrics.json` also exists and the
tag is in `results/FREEZE.json`.

## 1. Historical grids (pre-takeover)

| grid | cells | location | status | note |
|---|---:|---|---|---|
| OOF 5-fold x 3 seeds | 60 | `lab3090:~/rulb/experiments/OOF*` | **INVALID** | truncation points sampled from official test RUL distribution |
| DSPT 8-config ablation | 32 | `lab3090:~/rulb/experiments/ABL_*` | **INVALID as a table** | 1 seed, 1 fold; test evaluator cannot load the checkpoints |
| Vanilla 8-config ablation | 96 | `lab3090:~/rulb/experiments/VB_*` | **ARCHIVED** | 3 seeds; window-level early stopping; cross-model claim confounded |
| Window sweep | 24 | `experiments/SWEEP_*` | **ARCHIVED** | pre-P0-1/P0-2 code |
| OOF test / ensemble | several | `experiments/OOFTEST*`, `experiments/ENS*` | **INVALID** | prediction averaging reported as single model (NUMBER_PROTOCOL.md) |

## 2. Protocol-fixed grids (this takeover)

All grids: `split_seed=7`, `fold=0`, `rul_cap=125`, `epochs=40`,
`patience=12`, `batch=512`, engine-level validation (5 cuts).

| grid file | tag prefix | runs | subsets | seeds | purpose | status |
|---|---|---|---:|---|---|---|
| `ablation8.json` | `A8_<cfg>_L70` | 96 | 4 | 3 | DSPT internal ablation: base, MS, SA, HI, MSSA, MSHI, SAHI, FULL | RUNNING |
| `vanilla.json` | `VAN_<variant>_L70` | 24 | 4 | 3 | independent Vanilla (raw) + feature-matched control (DSPT 3-channel + cycle) | RUNNING |
| `nocycle.json` | `NC_base_L70` | 12 | 4 | 3 | absolute-cycle control | RUNNING |
| `ms_mech.json` | `MSM_<mode>_L70` | 18 | FD001, FD004 | 3 | single_proj / multi_noproj / multi3 vs off / full | RUNNING |
| `sa_mech.json` | `SAM_<mode>_L70` | 24 | 4 | 3 | sensor identity vs operating-condition context | RUNNING |
| `hi_mech.json` | `HIM_<variant>_L70` | 18 | FD001, FD004 | 3 | HI source (hf/hB), monotone sign, no monotone | RUNNING |
| `attribution.json` | `ATTR_<variant>` | 7 | FD001 | 101 | historical 10.91 -> 12.70 attribution (not paper evidence) | DONE (a40) |
| `nomask.json` | `NM_<cfg>` | 24 | 4 | 3 | padding-mask control (attribution suggests unmasking helps) | PLANNED |
| `sens_cuts.json` | `SENS_<cfg>` | 12 | FD001, FD004 | 3 | validation-cut sensitivity (earlier 30% cut) | PLANNED |
| `repro.json` | `REPRO_<cfg>` | 2 | FD001 | 101 | same-machine determinism check | PLANNED |
| `window.json` | `WIN_L<L>_<cfg>` | TBD | 4 | 3 | window search after structure freeze | PLANNED |

### Grid definitions

- `base`: MS off (single k=3 conv), SA off (learned sensor softmax still active), HI off.
- `MS`: `ms_mode=full` (k=3/5/7 + projection).
- `SA`: `sa_mode=basic` (one cross-sensor Transformer layer, no sensor identity).
- `HI`: `hi_source=hf`, correct decreasing monotone penalty, lambda=0.5, mu=0.1.
- Pair configs enable both modules; `FULL` enables all three.
- `MSM_single_proj`: one k=3 conv + projection (isolates the projection).
- `MSM_multi_noproj`: three scales, no projection (isolates multi-scale).
- `MSM_multi3`: three k=3 convs + projection (parameter-matched to full, single scale).
- `SAM_identity`: SA + learned sensor-identity embedding.
- `SAM_cond`: SA + operating-condition embedding.
- `HIM_hf_wrong`: HI on h_f with the inverted monotone sign.
- `HIM_hf_nomono`: HI on h_f without the monotone penalty.
- `HIM_hB_correct`: HI read from h_B, the representation used by the main RUL output in long mode.

## 3. Result layout

```
paper_b/results/
  raw/<tag>/<subset>/s<seed>/
    config.json        full run configuration
    metrics.json       engine-level validation metrics, params, best epoch
    val_pred.csv       unit, cut, y_true, y_pred
    model.pt           checkpoint
    test_metrics.json  only after freeze
    test_pred.csv      only after freeze
  all_runs.csv         flat table (analyze.py)
  FREEZE.json          tags allowed to touch the official test set
```

## 4. Compute budget

2 x RTX 3090, 1 process per GPU. Measured single-process cost at L=70,
batch 512: base 76 ms/step, MS 133 ms/step, SA 162 ms/step, FULL 174 ms/step.
The 8-config grid is the dominant cost; it runs first.
