# ROUND_PROMPT

## Overnight driver prompt (2026-10-29 23:40 → 2026-10-30 09:00) — use this one

> Read `paper/OVERNIGHT_CONTRACT.md` and `paper/QUEUE.md`. Execute **only** the
> head unfinished task. When it is finished, update `paper/STATUS.json`,
> `paper/QUEUE.md` and `paper/RUNLOG.md`, then exit immediately. Do not start the
> next task.

---

## Persistent objective for manual rounds (reference; not the overnight prompt)

Feed this file at the start of every round. It replaces a goal object; it is not
a chat message and must not be rewritten mid-round.

## Objective

Produce a submission-ready IEEE TIM Regular Paper (Track E) on the resolution
and reproducibility of C-MAPSS RUL evaluation, plus a reproducible artefact and
submission package. "Done" means **submission-ready and approved by the human**,
not accepted.

Authority: human, 2026-10-29. Frozen protocol stays frozen; official test files
stay unread; FD003/FD004 stay sealed.

## Thesis

Measured at engine-level resolution, most method differences on C-MAPSS are
undecidable; accuracy differences are driven mainly by a time shortcut and by
lifespan-scale identifiability rather than by module design; therefore we
propose a reproducible evaluation protocol.

## Evidence baseline (do not re-derive)

* variance decomposition 2.6% / 31.5% / 65.9%; MDE 1.60 (260), 1.14 (509), 0.97 (709)
* 21 paired comparisons, 0 survive Bonferroni; time shortcut +6.89 / +8.86;
  frozen pipeline BE−A = +1.26
* lifespan: oracle +8.23 [5.30, 11.02]; deployable est +0.08 (CI crosses 0);
  the "38%" is withdrawn (leak)
* E2 closed FAIL: monotonicity 0.505→1.000 and Spearman 0.377→0.976, but
  corr(T̂,T) only 0.319→0.347 (<0.5); T̂/T = 1.92 is structural
* E1: 6 verified references, no "first" claims
* E5: 49 occurrences in 15 files classified; "38%" has two meanings — R1's
  CI-width 38% is valid, only §I's is withdrawn

## Five-step round contract (all five, or the round did not happen)

1. First output line: `MODE=manual round=N next_action=...`
2. If the round computes anything, write `paper/PREREG_*.md` and record its
   sha256 **before** any number exists.
3. Do only `STATUS.json.next_action`. Do not bundle the next item.
4. Update `paper/STATUS.json` + `paper/CLAIMS.md` + `paper/ROUNDS.md`.
5. One-page result: action / numbers+CI / artefact paths / sha / gate / the
   single next action.

## Circuit breakers (stop, do not push through)

* STATUS.json sha unchanged → stop.
* Gate FAIL → close that line, move on.
* Same human blocker 3 rounds running → mark `blocked`.
* Tempted to change a claim to fit a result → stop; change the claim status first.

## Standing rules

* Register before computing; no hand-written numbers (rebuilt by
  `scripts/paper_*.py`, diff = 0).
* Every citation needs DOI/URL + a supporting sentence.
* Withdrawals are explicit, never silent edits.
* No test files, no FD003/FD004, no fabrication, no SOTA/"first" claims,
  no HARKing, no salami slicing.
* E3/E4 change frozen code: back up first, record hashes, prove `use_pe=False`
  is bit-identical, then run.

## Human blockers that do not stop work

* G1 front matter (authors, affiliations, funding, COI) — blocks only the title
  page and G5; write §2–§6 first.
* Journal/title are decided: IEEE TIM Regular; title option C.
* Challenge threshold is decided: 260 engines main table; 509/709 are labelled
  arithmetic projections.
* Test-data disclosure wording is approved in `paper/SUPPLEMENT_DATA_USE.md`.
