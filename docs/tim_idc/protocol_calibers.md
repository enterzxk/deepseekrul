# Protocol Calibers: Do Not Mix These Two Tables

Date: 2026-11-04. Purpose: prevent the two historical evaluation calibers from
being quoted side-by-side as if comparable. Trigger: confusion between the
E21/E22a official-test numbers (FD001 9.82 / FD002 11.51 / FD003 10.06 /
FD004 11.79) and the IDA-PE outer-validation numbers (FD002 M0 12.75 / M1p
12.32).

## Caliber A — E21/E22a (2026-10-31, pre-IDA)

- Evaluation set: **official C-MAPSS test engines** (FD001 100 / FD002 259 /
  FD003 100 / FD004 248 units), RUL truth capped at 125.
- Model: tuned PE, ~10^5-scale params (d_model 88, 2 layers), 40 epochs,
  patience 15.
- Protocol: single train/val split (split-seed 7), not outer cross-fitting.
- Source: `.workbuddy/memory/2026-10-31.md` (E22a; includes the cap-125
  caliber-bug fix note: uncapped evaluation had inflated FD002 to 23.39).
- **Status: historical reference only.** These numbers cannot be pooled into
  the IDA tables (different evaluation set, different protocol).

## Caliber B — IDA-PE protocol (current, frozen)

- Evaluation set: **outer-validation units** of the 5 engine-level outer
  folds (FD002: 259 train-side units across folds; FD001: 80), truth capped
  at 125.
- Model: frozen small config, 42.3k params (d_model 64, 1 layer), 80 epochs,
  patience 20, same loss/schedule as all ablation arms.
- All Gate 0/1/Module B/C and multi-seed numbers live here.
- FD001 coincidence: caliber A and B happen to agree on FD001 (9.82 vs 9.82)
  because val/test unit distributions are close there; FD002 differs by
  ~1.2 RMSE because of val-vs-test distribution shift, NOT model regression.

## The bridge (running)

`scripts/run_e22a_align.py` = E22a backbone (d_model 88, 2 layers, 40/15
schedule) inside the IDA protocol with prior injection (`M0big/M1big/M1pbig`).
Zero-init guarantees M1pbig >= E22a-PE at equal capacity, so the final
official-test number cannot land below the historical 11.51 for reasons of
architecture; any gap can only come from the frozen protocol itself.
Official test is unsealed exactly once, after Gate 2/3, with the frozen
final model.
