# DeepSeek Battery Trajectory Research — fixed-origin capacity-only SOH forecasting

This repository is a **lean, reviewable snapshot** of an independent research line on **fixed-origin, capacity-only long-horizon battery SOH trajectory forecasting** across multiple public battery sources.

It is intentionally **not** an online-RUL / survival project (that is the separate Codex line, `/home/yndx/zxk/rul`, which was not read or modified). It also does **not** claim SOTA or “first” results. The main deliverable is a **multi-source diagnostic + a minimal hierarchical baseline**.

## One-paragraph result

On **101 full-SOC cells from 5 public sources** (SNL, HNEI, UL-PUR, Oxford, CALCE; 69 cells with observed 80% EOL), a one-parameter **Hierarchical Offset Forecaster (HOF)** — a protocol-group mean trajectory `μ_g(k)` plus the cell’s current causal offset `r_c(k0)` — is a strong baseline for capacity-only long-horizon forecasting. At h=300 cycles, cell-equal SOH MAE is **0.0224** for HOF vs **0.0262** for a GRU, **0.0424** for exp-100, **0.0477** for power-law, and **0.0785** for persistence. HOF also improves 80% EOL error over protocol mean and power-law, but **fails when the test source has no matching protocol prior** (leave-one-source-out h=300 MAE **0.0820**; EOL 194/194 no crossing).

## Key numbers (5-fold, cell-level; development set, not an independent test set)

| Method | h=50 | h=100 | h=200 | h=300 |
|---|---:|---:|---:|---:|
| Persistence | 0.0162 | 0.0266 | 0.0505 | 0.0785 |
| Linear-100 | 0.0142 | 0.0218 | 0.0387 | 0.0552 |
| Exp-100 | 0.0119 | 0.0178 | 0.0302 | 0.0424 |
| Power-law | 0.0145 | 0.0204 | 0.0335 | 0.0477 |
| Protocol mean | 0.0240 | 0.0249 | 0.0281 | 0.0352 |
| GRU | 0.0128 | 0.0140 | 0.0183 | 0.0262 |
| Residual GRU | 0.0151 | 0.0185 | 0.0252 | 0.0344 |
| **HOF** | **0.0076** | **0.0103** | **0.0143** | **0.0224** |

EOL80 (observed-EOL cell-origin cases): HOF has 157 predicted crossings and 37 right-censored/undecided cases; MAE **165.3 cycles**, median 48, within-20% **75.8%** among predicted, **61.3%** if the 37 non-crossings count as failures. Protocol mean: 290.1 cycles / 95 / 53.1% / 43.8%. Linear/exp/power MAE: 422.6 / 358.7 / 285.6 cycles.

**Per-source h=300 cell-equal SOH MAE (HOF / GRU):** HNEI 0.0199 / 0.0214; CALCE 0.0368 / 0.0379; Oxford **0.0077 / 0.0051** (GRU better); SNL 0.0233 / 0.0289.

**Leave-one-source-out (HOF, h=300):** overall 0.0820; HNEI 0.1535, CALCE 0.0808, SNL 0.0728, Oxford 0.0190. EOL: 194/194 cases no predicted crossing. This is the main negative result.

**Intervals (HOF, out-of-fold split-conformal):** SOH 80% coverage 0.797–0.802, 90% 0.897–0.899; EOL 80% coverage 0.809 (width 208 cycles), 90% 0.857 (width 391 cycles; under-coverage).

## What is in this repo

| Path | Content |
|---|---|
| `reports/DEEPSEEK_RESEARCH_REPORT_20261019_ZH.md` | **Main report (Chinese)** — route rationale, closest-paper conflicts, data audit, results, failures, next actions |
| `DATA_DECISION.md` | Data sources, physical cell IDs, SOH/EOL semantics, censoring, limitations |
| `BASELINE_RESULTS.md` | Main tables, per-source, EOL, intervals, per-cell worst cases |
| `NOVELTY_CONFLICT_MATRIX.md` | Nearest-neighbor papers (BatteryMFormer KDD 2026, 2024 curvature knee, 2025 Applied Energy, 2025 transferable physics-informed, 2026 multi-fidelity, 2026 EA-LSTM) and what cannot be claimed |
| `PROJECT_PROGRESS.md` | Status, completed facts, risks, next atomic action |
| `docs/NEXT_SESSION.md` | Minimal recovery context |
| `README_RESEARCH.md` | Reproduction notes and directory map |
| `code/` | Data scraping/cleaning, HOF, GRU/residual GRU, EOL, cross-source, conformal, figures |
| `results/final_metrics.json` | All aggregate numbers (machine-readable) |
| `results/per_cell_h300.csv` | Per-cell h=300 metrics (HOF/pmean/GRU/power/exp) |
| `results/eol_compare_clean.csv` | HOF vs pmean/linear/exp/power EOL comparison |
| `results/hof_eol_clean.csv` | HOF EOL predictions/errors/censoring |
| `results/losource_hof_eol.csv` | Leave-one-source-out EOL (all no-crossing) |
| `results/diag_*.csv` | CALCE raw-file independent audit |
| `data/ba_cell_summary.csv` | Per-cell EOL/cycles/chemistry/protocol |
| `data/celllist.json` | Battery Archive cell metadata (query 29) |
| `data/calce_manifest.json` | CALCE zip URLs + SHA256 + parse status |
| `data/calce_parsed/` | 12 CALCE raw XLSX cells parsed to per-cycle CSV |
| `figures/` | MAE vs horizon, EOL error by source, cross-source comparison |

## Full row-level data

This lean repo keeps the reviewable aggregate tables but omits several large row-level files from the tree. They are bundled in:

```text
release/full_data_bundle.tar.gz   (23 MB)
release/full_data_bundle.sha256
```

Extract at repo root to restore `data/ba_clean.csv`, `data/ba_raw2.tar.gz`, `results/hof_eval_clean.csv`, `results/unified_percell.csv`, `results/gru_pred.csv`, `results/resgru_pred.csv`, `results/losource_hof_pred.csv`.

The full server snapshot is also at `/home/yndx/zxk/deepseek-reaserch/deliverable_snapshot_20261019.tar.gz` (SHA256 `1ec7ccf21ce494af14f846a63be0b3095293a1352499659fa9f3ed0dab662030`).

## Data provenance (short)

- Battery Archive public `Cycle Test Data` dashboard, query 29 (cell metadata) and query 26 (per-cycle capacity/energy); retrieved 2026-10-19. 101 full-SOC cells: SNL 61, HNEI 15, UL-PUR 10, Oxford 8, CALCE 7.
- CALCE official `https://calce.umd.edu/data`; 12 parsed XLSX cells (6 CS2 + 6 CX2); raw zip URLs/SHA256 in `data/calce_manifest.json`.
- No MATR/NASA data, no Codex `/home/yndx/zxk/rul` files, no private keys, no copyrighted PDFs.
- **Caveat:** BatteryMFormer uses a 13-subdataset public lifetime database; SNL/HNEI/CALCE/Oxford may overlap with its training data. We cannot claim full data independence from BatteryMFormer.

## What this repo does **not** claim

- Not SOTA; BatteryMFormer and the 2025 Applied Energy trajectory model were **not** run on a common task.
- 5-fold cell-level CV is a **development set**, not a frozen independent confirmation set.
- HOF is **not** a cross-source general method: leave-one-source-out fails.
- Oxford EOL is sparse; 3/4 observed-EOL cells were not predicted to cross.
- 2025 Applied Energy is closed access; only metadata/title was available. No full-text claim.

## Nearest-neighbor papers (for orientation)

- BatteryMFormer, KDD 2026 — `arXiv:2605.27044`; early voltage/current → full-life SOH trajectory; multi-level Transformer.
- Zhang, Altaf, Wik, 2024, J. Power Sources — `10.1016/j.jpowsour.2024.234619`; curvature-based knee/knee-onset.
- Multi-time scale RUL/knee, 2025, J. Energy Storage — `10.1016/j.est.2025.116024`.
- Coupling capacity fade model with ML, 2025, Applied Energy — `10.1016/j.apenergy.2025.125703` (closed).
- Transferable physics-informed knee detection, 2025, J. Power Sources — `10.1016/j.jpowsour.2025.238028`; `arXiv:2501.14573`.
- Multi-fidelity coupled degradation, 2026, GEITS — `10.1016/j.geits.2025.100367`.
- EA-LSTM early KO/KP/EOL, 2026, Results in Engineering — `10.1016/j.rineng.2026.109664`.

## Quick start

```bash
# Optional: restore full row-level data
tar -xzf release/full_data_bundle.tar.gz

# Recompute aggregate metrics (needs numpy/pandas/scipy/scikit-learn)
PYTHON=<python-with-deps> bash code/run_all.sh
```

`code/run_all.sh` creates `work/`, links the required `data/`/`results/` files, and runs HOF/EOL/final metrics. On the server, `PYTHON=/home/yndx/miniconda3/envs/a40_env/bin/python` was verified.

## Contact/usage

This is a research snapshot for review and GPT-assisted reading. If you use the data, cite Battery Archive and the original studies; if you use the HOF baseline, cite this repo/report and the original data sources.
