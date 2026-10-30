# QUEUE — overnight 2026-10-29 → 2026-10-30

Head of queue = first row whose status is `NEXT`, else the first `pending`.

| id | task | gpu | depends | acceptance | status |
|---|---|---|---|---|---|
| T0 | E3 recheck: converged bootstrap, amendment, hash-table fix | 0 | — | 5 × 100k reported; verdict consistent; prereg §12 amended | **DONE** (FAIL: +0.309…+0.324, mean +0.317) |
| T1 | BE fixed-budget sensitivity (`--patience 80`) | 3 run | T0 | BEfix − A ≥ 2.0 with CI excluding 0 keeps "load-bearing" | **DONE** (identical to BE to 4 d.p.; +2.69 stands) |
| T2 | E4 from-scratch reproduction | 2 run | T0 | replay ≤1e-3; \|ΔRMSE\| ≤ 0.15 | **DONE** (Δ = 0.00000; replay 7.2e-6) |
| T3 | E6 statistical audit | 0 | — | one CI recipe; MC error reported; unreproducible claims listed | **DONE** (C3 withdrawn, C2/C10 downgraded) |
| T4 | Rewrite §4.3 in English | 0 | T0–T3 | no "non-inferior" as a conclusion | **DONE** (`paper/SECTION_4_3.md`) |
| T5 | Morning report | 0 | T3, T4 | three sections | **DONE** (`paper/MORNING_20261030.md`) |
| T6 | E1 references → 12–15 verified entries | 0 | — | every row has DOI/URL + supporting sentence | **DONE** (12 rows, 6 added; PASS) |
| T7 | Draft §2 Background from verified refs only | 0 | T6 | `paper/SECTION_2.md`; every citation in `refs_verified.csv` | **DONE** (`paper/SECTION_2.md`) |

## Evidence for the completed rows

* T0 — `paper/E3_AMENDMENT_1.md`, `results/dsh/e3/bootstrap_stability.{json,csv}`.
* T1 — `paper/T1_BE_SENSITIVITY.md`, `experiments/e3fix/BEfix_s*/…/metrics.json`.
* T2 — `paper/E4_RESULT.md`, `results/dsh/e4/replay.json`, `experiments/e4/A_s2027/…`.
* T3 — `paper/E6_STATS_AUDIT.md`, `results/dsh/e6/E6_stats.json`, `scripts/paper_e6_stats.py`.
* T4 — `paper/SECTION_4_3.md`.
* T5 — `paper/MORNING_20261030.md`.
* T6 — `paper/refs_verified.csv`, `paper/E1_LITERATURE.md` §2/§4/§5, `results/dsh/e1/refs_metadata.json`.

## Not in this queue

§1/§3/§5 drafting, E5 re-scan, E8 submission package — these need the human in the
loop (framing, author block, disclosure sign-off).

| T14 | E10 5-fold CV: A/PE x 3 seeds x folds 1-4 (24 runs), then pooled estimand + fold-stratified bootstrap | GPU | T13 | CI on 260 engines; verdict per `paper/E10_CV_PREREGISTRATION.md` | **RUNNING** |
| T15 | §4.1 variance decomposition (two designs + contrast-level) | 0 | T13 | `paper/SECTION_4_1.md`; `results/dsh/e10/variance_two_designs.json` | **DONE** |
| T16 | §5 ten-check guideline | 0 | T15 | `paper/SECTION_5.md` | **DONE** |
| T17 | §4.4 lifespan scale and estimator failure | 0 | — | `paper/SECTION_4_4.md` | **DONE** |
| T18 | §4.2 resolution (MDE, multi-fold, multi-cut) — needs T14 | 0 | T14 | `paper/SECTION_4_2.md` | pending |
