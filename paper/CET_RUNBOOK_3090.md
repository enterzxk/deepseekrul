# CET-RUL runbook (3090 box)

## Machines

| role | host | path | env |
|---|---|---|---|
| training | `lab3090` (`10.50.1.240:2022`, user `user`) | `/home/user/yr/zxk/bearing-rul-pi` | `/home/user/miniconda3/envs/baseline/bin/python` |
| reference / older runs | `lab-a40` (`172.25.52.97`, user `yndx`) | `/home/yndx/zxk/bearing-rul-pi` | `/home/yndx/miniconda3/envs/a40_env/bin/python` |

The 3090 box has **2 × RTX 3090 (24 GB)**. The environment used is
`miniconda3/envs/baseline`: Python 3.10.0, torch 2.10.0+cu126, numpy 1.26.4, pandas 2.3.3,
scipy 1.15.3, scikit-learn 1.7.2 — verified with a real CUDA matmul on the 3090, not just
`cuda.is_available()`.

Environment discovery notes, so this is not repeated:

* `anaconda3/envs/ptYin` has torch 1.8+cu111 and works on the 3090, but it is **Python 3.6**,
  which rejects `from __future__ import annotations`. Do not use it.
* `anaconda3/envs/gnn` is torch 1.8+**cu102**; CUDA 10.2 does not support the 3090's sm_86.
* Other envs with working torch on this box: `gwk` (1.13+cu117), `PDFormer` (1.10),
  `miniconda3/{ccfm,dfm}` (2.5/2.6+cu124), `miniconda3/diff` (2.0+cu118),
  `miniconda3/baseline` (2.10+cu126), `rocky`, `stgncde`, `md`.

## Path configuration

`ROOT` is no longer hard-coded: `scripts/dsh_e21_test_all.py` and `scripts/cet_train.py`
read `RUL_ROOT` from the environment and fall back to the A40 path. Always export it:

```bash
export RUL_ROOT=/home/user/yr/zxk/bearing-rul-pi
export PYTHONPATH=$RUL_ROOT/src:$RUL_ROOT/scripts
```

## Commands

```bash
# one run
cd /home/user/yr/zxk/bearing-rul-pi
./run_cet.sh FD001 0 2027 40                 # subset fold seed epochs
./run_cet.sh FD001 0 2027 40 no_gate         # ablation arm

# logs
tail -f logs/cet_<subset>_f<fold>_s<seed>.log
```

`run_cet.sh` writes to `experiments/cet/<subset>/f<fold>_s<seed>[_<ablation>]/`, holding
`model.pt` and `metrics.json` (full history plus final metrics).

## What was transferred

`src/`, `scripts/{run_candidate.py, dsh_e21_test_all.py, cet_train.py}`, `scripts/cido/`,
`data/CMAPSS/` (26 MB). Everything else — earlier experiments, paper drafts, result JSONs —
stays on the A40 and in git; it is not needed to train.

## First results on the 3090 (to be replaced by the full grid)

FD001, fold 0, seed 2027, 4 epochs (smoke): `rmse_last` 97.1 → 41.2, gate 0.00 → 0.55.
The gate is live and the belief is being driven by evidence; the absolute error is not yet
competitive (the comparators sit near 10–14 on FD001), so the 40-epoch run and the
architecture/prior fixes are still in progress.
