# Overnight driver — automation memory

## 2026-10-30T07:59 (hop-0759) — exited at STEP 0a, STOP still present

* **Outcome:** guard printed `FREE` (no `paper/LOCK` within 7200 s); `paper/STOP` present,
  mtime 2026-10-30 02:54:42, size 2257 B — identical to the breaker hop-0256 wrote.
  STEP 0 orders a stop on STOP → no STEP 1–4 work.
* **Action:** none. QUEUE/OVERNIGHT_CONTRACT/STATUS/RUNLOG/CLAIMS untouched; no claim,
  no preregistration, no compute, no SSH, no artefact sync; STOP not deleted.
* **Interpretation:** sixth consecutive gate exit (0256 tripped; 0355/0456/0557/0658/0759
  held). Breaker functioning as designed; requires human action to clear.
* **Resume path unchanged (four steps, all required):** delete `paper/STOP`; mark ≥1 queue
  row `NEXT`; delete `paper/GOAL_MET.md`; re-arm `scheduledAt`.

## 2026-10-30T04:56 (hop-0456) — exited at STEP 0a, STOP still present

* **Outcome:** guard printed `FREE` (no `paper/LOCK`); `paper/STOP` present, mtime
  2026-10-30 02:54:42, i.e. byte-for-byte the same breaker hop-0256 wrote. STEP 0 orders
  a stop on STOP, so no STEP 1–4 work was performed.
* **Action:** none beyond this note. QUEUE/OVERNIGHT_CONTRACT/STATUS/RUNLOG/CLAIMS
  untouched, no task claimed, no preregistration, no compute, no SSH, no artefact sync.
  STOP not deleted.
* **Interpretation:** third consecutive exit at this gate (hop-0256 tripped it, 0355 and
  0456 held it). Not a new failure; the breaker is functioning as designed and requires
  human action to clear.
* **Resume path unchanged (four steps, all required):** delete `paper/STOP`; mark ≥1 queue
  row `NEXT`; delete `paper/GOAL_MET.md`; re-arm `scheduledAt`.

## 2026-10-30T03:55 (hop-0355) — exited at STEP 0a, STOP present

* **Outcome:** guard printed `FREE` (no `paper/LOCK`), but `paper/STOP` exists
  (mtime 2026-10-30 02:54, written by hop-0256). STEP 0 orders a stop on STOP.
* **Action:** none beyond this note. No QUEUE read beyond existence check, no claim,
  no preregistration, no compute, no SSH, no artefact sync. STATUS/QUEUE/RUNLOG/CLAIMS
  untouched; `paper/STOP` not deleted.
* **Interpretation:** expected behaviour, not a new failure. The hop-0256 breaker is
  doing its job. Future hops will keep exiting here until a human runs the four-step
  resume path documented inside `paper/STOP`.
* **Note for the record:** `paper/DRIVER_PROMPT.md` §Stop-and-safety (two identical
  `STATUS.json` hashes) fired at hop-0256 after the queue drained 8/8 `DONE`, not
  because of any result change.

## 2026-10-30T02:56 (hop-0256) — chain terminated

* **Outcome:** guard FREE, no `paper/STOP` on entry, no `HOP_LOCK.json`; queue still 8/8
  `DONE`, zero `open` claims. No head task existed, none invented.
* **Action:** wrote `paper/STOP` (circuit breaker per `paper/DRIVER_PROMPT.md`
  §Stop and safety — `STATUS.json` sha256 unchanged for six consecutive hops) plus one
  row in `paper/DRIVER_LOG.md`. STATUS/QUEUE/CLAIMS/RUNLOG left byte-identical.
* **Superseded decision:** hop-0151 had declined to trip the breaker; that was reversed
  because `DRIVER_PROMPT.md` is the declared source of truth and states the rule
  explicitly. Do not re-litigate.
* **Resume path (all four steps):** delete `paper/STOP`; mark ≥1 queue row `NEXT`;
  delete `paper/GOAL_MET.md` (STEP 0b short-circuits on its presence); re-arm
  `scheduledAt`. Documented inside the STOP file itself.
* **This automation should not fire again** while `paper/STOP` exists — every future hop
  will exit at STEP 0a in well under a minute.

## 2026-10-30T01:52 (hop-0151)

* **Outcome:** no-op hop. Guard FREE, no `paper/STOP`, no `HOP_LOCK.json`.
* **Reason:** `paper/QUEUE.md` is drained — 8/8 rows `DONE` (T0..T7), zero `NEXT`/`pending`.
  `paper/CLAIMS.md` has 0 `open` claims (15 supported, 1 withdrawn C3, 2 partial C10/C14).
  sha256 of STATUS.json / CLAIMS.md / QUEUE.md all identical to `paper/GOAL_MET.md`
  (written 01:00 by hop-0100), so the goal predicate still holds.
* **Action taken:** one row appended to `paper/DRIVER_LOG.md` only. No task invented,
  no preregistration, no compute, no SSH. STATUS/QUEUE/RUNLOG untouched.
* **Decision:** `paper/STOP` deliberately NOT written — a drained queue is the designed
  cheap-exit case; STOP is reserved for a stalled queue and would block a re-armed queue.

## Standing notes for later runs

* Queue exhausted as of hop-0100. Remaining work is human-in-the-loop and out of queue:
  §1/§3/§5 drafting, E5 re-scan, E8 package, plus author/affiliation/funding/COI and the
  260-vs-509 main-table decision.
* Morning entry point: `paper/MORNING_20261030.md`.
* If the queue is re-armed, the standard STEP 1→4 procedure applies again (preregister
  before computing; never move seed/margin/percentile).

## 2026-10-30T06:58 (hop-0658) — exited at STEP 0a, STOP still present

* **Outcome:** guard printed `FREE` (no `paper/LOCK` under 7200 s); `paper/STOP` present,
  mtime 2026-10-30 02:54, unchanged from hop-0256. STEP 0 orders a stop on STOP → no
  STEP 1–4 work.
* **Action:** none. QUEUE/OVERNIGHT_CONTRACT/STATUS/RUNLOG/CLAIMS untouched; no claim,
  no preregistration, no compute, no SSH, no artefact sync, STOP not deleted.
* **Interpretation:** fifth consecutive gate exit (0256 tripped, 0355/0456/0557/0658 held).
  Breaker working as designed and still needs human action.
* **Resume path unchanged:** delete `paper/STOP`; mark ≥1 queue row `NEXT`; delete
  `paper/GOAL_MET.md`; re-arm `scheduledAt`.

## 2026-10-30T05:57 — guard exit (paper/STOP present)

* **Outcome:** guard FREE (no fresh LOCK), but `paper/STOP` exists → exited at the
  pre-flight safety check before reading QUEUE.md / OVERNIGHT_CONTRACT.md.
* **Action:** none. No task taken, no preregistration, no compute, no SSH, no artefact
  writes; QUEUE/STATUS/RUNLOG/CLAIMS untouched. STOP is still the deliberate circuit
  breaker written by hop-0256 (queue drained 8/8 DONE, STATUS.json hash frozen).
* **Resume path unchanged:** delete `paper/STOP`, mark a queue row `NEXT`, delete
  `paper/GOAL_MET.md`, re-arm the schedule.
