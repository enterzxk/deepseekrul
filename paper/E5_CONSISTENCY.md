# E5 — Consistency inventory of superseded numbers

Scope: every `.md` / `.tex` in the repository, scanned 2026-10-29.
Method: literal occurrence count per token, then manual classification of each
file's context (a token can be legitimate in one file and void in another).

Totals: **49 occurrences in 15 files** for the tokens
`18.18`, `18.47`, `38%`, `13.209`, `12.868`, `14.155`.

## 1. Classification rules (why we do not bulk-replace)

1. **Historical reports are append-only logs.** `P7_*`, `P8_*`, `P9_*`, `P10_*`,
   `MORNING_*`, `results/*/summary.md` record what was believed at the time.
   Rewriting their numbers would destroy the audit trail that the paper's
   reproducibility section relies on. They get a **banner**, not an edit.
2. **Entry-point documents must not mislead.** `README.md`,
   `PROJECT_PROGRESS.md`, `docs/director/FINAL_DELIVERABLE.md` are what a reader
   opens first; each void number there must be labelled at the point of use.
3. **The manuscript never uses any of them.** If a number is void, it does not
   appear in `paper/main.tex`, full stop — not even as a historical remark.

## 2. Ambiguity that would have caused a false strike

`38%` appears with **two unrelated meanings**:

| meaning | where | status |
|---|---|---|
| multi-cutoff evaluation narrows the CI by 38% | `REVIEW_20261022_R1.md` §41, `FINAL_DELIVERABLE.md` §1 | **valid**, keep |
| deployable `c/T̂` recovers 38% of the oracle gain | `REVIEW_20261022_R3.md` §I | **withdrawn** (§J) |

A naive global replace of `38%` would have deleted a supported result. This is
exactly the failure mode the task book forbids.

## 3. Inventory

| file | hits | what the numbers are | disposition |
|---|---:|---|---|
| `docs/director/REVIEW_20261022_R3.md` | 8 | the withdrawn §I arm (18.18, 18.47, "38%") | ✅ already labelled: banner at top + §J + struck-through rows |
| `paper/CLAIMS.md` | 7 | ledger rows for withdrawn values | ✅ intentional (the ledger must name what is dead) |
| `README.md` | 6 | P9 record 13.209/12.868; P7 14.155 | ⚠️ P9/P10 covered by the existing invalidation notice; **P7 14.155 needs a label** |
| `docs/director/REVIEW_20261022_R1.md` | 5 | CI-width 38% | ✅ valid, no action |
| `P9_DIRECTOR_REPORT.md` | 5 | 13.209 / 12.868 | ⚠️ needs banner |
| `P7_FINAL_VERDICT.md` | 5 | 14.155 | ⚠️ needs banner |
| `P10_FINAL_VICTORY_REPORT.md` | 4 | 13.209 / 12.868 (as the P9 comparison) | ⚠️ needs banner |
| `results/p9/p9_summary.md`, `results/p9/p9_table_main.tex` | 2 | 13.209 in a generated table | ⚠️ generated artifact: add a `SUPERSEDED` companion file, do not edit the table |
| `results/final_FD001_P7_5seeds/summary.md` | 1 | 14.155 | ⚠️ same treatment |
| `docs/director/FINAL_DELIVERABLE.md` | 1 | CI-width 38% | ✅ valid |
| `docs/LITERATURE_BENCHMARK.md` | 1 | 13.209 in the P9 row | ⚠️ needs a footnote |
| `TIM_AUDIT_AND_STAGE2_REPORT.md` | 1 | 13.209 (already described as void in prose) | ✅ consistent |
| `PROJECT_PROGRESS.md` | 1 | 14.155 | ⚠️ needs banner (doc is dated 2026-10-21 and is stale overall) |
| `P8_SOTA_REPORT.md` | 1 | 14.155 as an input to P8 | ⚠️ needs banner |

## 4. Open question: is 14.155 actually void?

The documented bug (`CRGraphTransformer._smooth_terminal`, tail window reversed)
lives in the **synergy head**. P7's Full model is BiLSTM-Attn + zero-init FiLM +
monotone head, i.e. it does **not** run that code path, so 14.155 is probably
unaffected. The 2026-10-29 task book nevertheless lists it as void.

Disposition: treat it as **void for the paper** (we cite nothing from the P-line),
and record in the banner that the technical scope is "void by instruction;
mechanism-specific re-audit not performed". This avoids both an unlabelled
number and a silent contradiction of the task book.

## 5. Actions

- [x] banner + §J in `REVIEW_20261022_R3.md`
- [x] ledger of withdrawn values in `paper/CLAIMS.md`
- [x] banners in `P7_FINAL_VERDICT.md`, `P9_DIRECTOR_REPORT.md`, `P10_FINAL_VICTORY_REPORT.md`, `PROJECT_PROGRESS.md`, `P8_SOTA_REPORT.md`
- [x] `SUPERSEDED.md` companions next to `results/p9/` and `results/final_FD001_P7_5seeds/`
- [x] footnote in `docs/LITERATURE_BENCHMARK.md` P9 row
- [x] `README.md`: label the P7 line
- [ ] re-run this scan after E7 to prove the manuscript itself is clean
