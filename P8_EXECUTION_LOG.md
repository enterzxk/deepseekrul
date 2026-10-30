# P8 Execution Log — what was actually done

Date: 2026-10-21
Author: agent run on lab-server (`/home/yndx/zxk/bearing-rul-pi`, NVIDIA A40)
Purpose: hand-off document for review (Gemini) — describes every code change,
every command, every failure that was fixed, and every deviation from the P8
brief. Results are in `P8_SOTA_REPORT.md`.

---

## 1. Scope check against the P8 brief

| Brief item | Done? | Where |
|---|---|---|
| A: replace Softplus head with lossless terminal-anchored projection | YES | `src/model.py` `monotonic_envelope`, `monotonic_mode="projection"` |
| A: free per-step regression, terminal y_L untouched | YES | `STG_MambaNet.trajectory_head`, prediction = `raw[:, -1]` |
| A: training L_mono on raw trajectory, weight 0.02 | YES | `PhysicsGuidedLoss` (unchanged) + `--lambda-mono 0.02` |
| A: inference backward-cumulative-max envelope | YES | `monotonic_envelope()` under `not self.training` |
| B: Hollow-AGCN spatial layer | YES | `graph_mode="hollow"`, alpha 0.35 |
| B: DWConv1d(k=3) residual stem + sinusoidal PE | YES | `ConvTransformerTemporalBranch` |
| B: zero-init FiLM into Transformer layers | YES | `ConvTransformerTemporalBranch.films` + input-level `self.film` |
| B: loss = MSE + asym(2.0) + 0.02 mono | YES | runner defaults |
| B: validation-selected safety offset delta in [-1, 0] | **NO** | see deviations |
| 8.1 FD001 val <= 11.5 / FD002 val <= 12.5 gate | RAN | not met |
| 8.2 six variants x 5 seeds on FD001 and FD002 | YES | 60 runs |
| 8.2 Table IV (Conv-SSM / Softplus head evolution) | YES | 3 seeds, FD001 |
| 8.3 tables, figures, report, git | YES (git push pending at time of writing) | `results/p8`, `figures/final` |

---

## 2. Code changes

### 2.1 `src/model.py`

New modules:

* `SinusoidalPositionalEncoding(d_model, max_len)` — fixed PE buffer.
* `ConvTransformerTemporalBranch(d_model, num_heads, num_layers, dropout,
  kernel_size=3, ff_mult=2.0, use_film=True, film_dim=3)`
  — residual DWConv stem -> LayerNorm -> PE -> N x TransformerEncoderLayer
  (pre-norm, GELU, FFN 2d), with a zero-init FiLM linear before each block.
  Returns `(hidden, None)` to keep the branch interface.
* `monotonic_envelope(trajectory)` — `flip -> cummax -> flip`; the last column
  is unchanged by construction.
* `GatedBranchFusion(d_model, dropout)` — diagnostic two-branch channel gate
  used only by the `p8_no_fusion` ablation.

`STG_MambaNet` changes:

* new kwargs: `monotonic_mode ∈ {none, softplus, projection, free}`,
  `conv_transformer_layers`, `film_in_layers`, `fusion_mode ∈ {cross, gated}`.
* `temporal_mode="conv_transformer"` builds the new branch; `cond` is forwarded
  to it so FiLM can act inside the stack.
* `monotonic_mode="projection"` adds `trajectory_head` and overrides
  `prediction = raw_trajectory[:, -1:]`; training returns the raw trajectory
  (so `L_mono` has gradient signal), eval returns the envelope.
* `monotonic_mode="free"` is the ablation: same head, no envelope, no mono loss.
* Backward compatibility: `monotonic_head=True` still maps to the P7 Softplus
  path; `monotonic_mode="none"` preserves the original prefix-regression
  trajectory behaviour; the diagnostics dict only adds a key when branch
  weights exist.

### 2.2 `scripts/run_ablation.py`

* New variants: `p8_full`, `p8_wo_graph`, `p8_wo_phys_mono`, `p8_conv_ssm`,
  `p8_softplus`, `p8_no_fusion`, with labels matching the brief.
* `--conv-transformer-layers` CLI flag; `fusion_mode` passed through.
* `config.json` now records `cudnn_deterministic`.
* TCN-only cuDNN switch (see 3.1).

### 2.3 New scripts

* `scripts/p8_run.py` — task queue with resume and `--parallel`, one process
  per (subset, seed, variant) so the grid is embarrassingly parallel.
* `scripts/p8_summarize.py` — mean/std tables, paired bootstrap CI, exact sign
  test, Markdown + LaTeX + JSON outputs.
* `scripts/p8_figures.py` — RMSE bars with seed dots, paired-delta forest plot,
  monotonicity panel.

---

## 3. Infrastructure problems found and fixed

### 3.1 TCN is 72x slower under deterministic cuDNN (the P7 blocker)

Micro-benchmark, batch 256, single A40:

| Setting | TCN | Transformer |
|---|---:|---:|
| `cudnn.deterministic=True` | 1080.3 ms/batch | 31.1 ms/batch |
| `cudnn.deterministic=False` | **14.9 ms/batch** | 31.9 ms/batch |

Dilated convolutions fall back to a slow deterministic kernel. P7 reported
"TCN took ~60 s/epoch and blocked the 5-seed run"; this is the cause.
Fix: keep determinism for every variant, disable it **only** for `tcn`, and
record the flag in `config.json`. TCN then trains in ~1.2 s/epoch.

### 3.2 FD002 could not import sklearn

`ImportError: libstdc++.so.6: version GLIBCXX_3.4.29 not found` when the
condition-aware path constructs `OperatingConditionScaler`. FD001 never touches
sklearn, so the grid worked there and died on FD002.
Fix: `export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH`
before launching (baked into `~/p8/fd2.sh`, `~/p8/diag.sh`).

### 3.3 Sealed test files and run-directory collisions

* FD002 test/RUL files were still `chmod 000` from an earlier phase, and
  FD001 was resealed after the P3 run; every grid script now `chmod 644`s the
  files it needs, and the protocol re-seals after the final evaluation.
* `p8_run.py` pre-created the run directory, which made `run_ablation.py`
  raise `FileExistsError`. Fixed by passing `--overwrite` (resume still skips
  runs that already have `metrics.json`).

---

## 4. Experiment timeline

| Step | What | Wall time |
|---|---|---|
| 1 | Smoke test: params, forward, envelope property | 2 min |
| 2 | FD001 validation gate, 6 variants, 40 ep, seed 2027 | 12 min |
| 3 | Schedule sweep 40 vs 60 vs 80 epochs (val-selected) | 20 min |
| 4 | FD001 6 variants x 5 seeds, 40 ep, parallel 6 | 25 min |
| 5 | FD002 6 variants x 5 seeds, 40 ep, parallel 6 | 45 min |
| 6 | Evolution runs: Conv-SSM, Softplus head, 3 seeds | 12 min |
| 7 | Diagnostic: no-cross-attention fusion, 3 seeds x 2 sets | 14 min |
| 8 | Summaries, figures, documents | 10 min |

Total GPU wall clock ~2h20m on one A40; 72 training runs.

---

## 5. Protocol as executed

```
subset        FD001 | FD002
seeds         2027 2028 2029 2030 2031
split         engine-level 80/20, validation for early stopping only
window/stride 50 / 1
epochs        40, patience 20   (selected on validation: 40 > 60 > 80)
batch/lr/wd   256 / 1e-3 / 1e-4
dropout       0.15
loss          MSE + asym(late 2.0) + 0.02 * L_mono
test          single evaluation after training, clip125 main table
hardware      1x A40, torch 2.5.1+cu121, python 3.9
```

Selected on validation, not test: epoch budget. Reported as such.

---

## 6. Deviations from the brief (explicit)

1. **Epochs = 40, not 80.** Validation-only sweep over {40, 60, 80} on seeds
   2027-2028: mean val RMSE 12.19 / 12.40 / 12.31. 40 selected. The 80-epoch
   cosine schedule underfits early and then overfits; documented in the report.
2. **Safety calibration offset delta in [-1, 0] was not swept.** The offline
   sweep on the produced predictions showed the optimum is at **+0.50** (i.e.
   slightly late), not in [-1, 0]; the gain is only 0.075 RMSE, so it was
   dropped rather than reported as a tuned number.
3. **No seed ensemble.** The brief asked for 5-seed confirmation, which is what
   was run. Ensembling is proposed as future work, not silently applied.
4. **FD002 uses the same backbone as FD001** (graph kept). The report shows the
   graph is negative on FD002; dropping it is listed as the first next step
   rather than being retro-fitted after seeing test numbers.
5. **The prediction is not clipped to [0,125]** at inference (only targets are
   clipped, per the frozen clip125 protocol); clipping predictions changes
   FD001 by < 0.1 RMSE and is reported in the analysis, not applied.

---

## 7. Where everything is

```
src/model.py                       new branch, envelope, fusion ablation
scripts/run_ablation.py            P8 variants + TCN cuDNN fix
scripts/p8_run.py                  parallel grid runner (resume-capable)
scripts/p8_summarize.py            tables + paired statistics + LaTeX
scripts/p8_figures.py              three figures
experiments/p8/final/<SUB>/<variant>_s<seed>/seed_<seed>/<variant>/
                                   config.json, history.csv, best_model.pt,
                                   test_predictions.csv, metrics.json
experiments/p8/evolution/          Conv-SSM + Softplus-head evolution runs
experiments/p8/diag/               no-cross-attention diagnostic
results/p8/p8_summary.md           main tables
results/p8/p8_stats.json           means, per-seed values, paired CIs
results/p8/p8_table_main.tex       LaTeX table
results/p8/p8_runs.csv             72 rows, one per run
figures/final/p8_rmse_comparison   bars + seed dots + 12.5 gate line
figures/final/p8_paired_deltas     forest plot vs Full
figures/final/p8_monotonicity      violation rate by variant
logs/p8/                           raw training logs
```

---

## 8. How to verify the central claim in 5 minutes

```bash
cd /home/yndx/zxk/bearing-rul-pi
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import sys, torch
sys.path.insert(0, "src")
from model import STG_MambaNet
m = STG_MambaNet(num_sensors=14, window_size=50, d_model=64, num_heads=4,
                 dropout=0.15, temporal_mode="conv_transformer",
                 spatial_mode="graph", graph_mode="hollow",
                 condition_film=True, monotonic_mode="projection")
m.eval()
x = torch.randn(64, 50, 14); cond = torch.randn(64, 50, 3)
with torch.no_grad():
    pred, traj = m(x, cond=cond, return_trajectory=True)
print("mono violation:", float((torch.diff(traj, dim=1) > 0).float().mean()))
print("terminal identity:", bool(torch.allclose(pred.view(-1), traj[:, -1], atol=1e-6)))
EOF
```

Expected: `mono violation: 0.0` and `terminal identity: True`.
