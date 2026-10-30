# Overnight driver — hop memory

Purpose: one-line-per-hop record so later hops do not redo work. Full detail lives in
`paper/DRIVER_LOG.md`, `paper/RUNLOG.md`, `paper/QUEUE.md`.

## Standing state

- Queue `paper/QUEUE.md`: T0..T7 all DONE as of hop-0100.
- `paper/GOAL_MET.md` written 2026-10-30T01:00:52+08:00 by hop-0100; predicate satisfied
  (8/8 rows DONE; 18 claims = 15 supported / 1 withdrawn / 2 partial / 0 open).
- Consequence: every subsequent hop hits guard 0b and exits without taking work.
  Do not re-evaluate STEP 1, do not touch QUEUE/STATUS/RUNLOG, do not acquire the lock.
- No `paper/STOP` file. No `HOP_LOCK.json` left behind by any hop.

## Hop outcomes

| hop | local | outcome |
|---|---|---|
| hop-0040 | 00:40 | T6 executed, PASS, 12 refs verified |
| hop-0100 | 01:00 | GOAL MET declared, GOAL_MET.md written, lock released |
| hop-0120 | 01:20 | guard 0b — goal already met, log line only |
| hop-0140 | 01:40 | guard 0b — goal already met, log line only |

## Remaining outside the queue (needs the human, not a hop)

§1/§3/§5 drafting, E5 re-scan, E8 package; STATUS gates G1–G5, E7, E8; author order /
affiliations / corresponding author / funding / COI; 260-vs-509 main-table choice;
test-disclosure wording; C10 artefact pinning and C14 non-result wording.

## Operating note

Hops are one-shot and must not chain. If a future hop finds GOAL_MET.md removed and the
queue reopened, re-run STEP 0 → STEP 2 from scratch and follow `paper/OVERNIGHT_CONTRACT.md`.
