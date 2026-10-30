# DRIVER_PROMPT — single-hop unattended driver (goal-mode equivalent)

Canonical text for every scheduled hop. Automations should paste this verbatim; this
file is the source of truth. Supersedes the "hourly driver" prompt of 23:53.

Why this shape: WorkBuddy's automation schema cannot express a 20-minute recurring
cadence, and the server ignores `BYMINUTE` when computing `nextRunAt` (evidence in
`paper/SCHEDULER_PROBE.md`). One-shot automations *are* honoured to the minute, so the
cadence is built from discrete hops instead of a recurrence rule. Goal semantics are
carried by an explicit, checkable condition rather than by a goal object.

---

## Hop prompt (paste verbatim)

```
Working directory: C:\Users\20562\Desktop\deepseek-rul

You are ONE hop of an unattended overnight driver. Do exactly one task, then exit.
Never chain two tasks. Never create or modify automations.

STEP 0 - guards, in order. On trigger: append one line to paper/DRIVER_LOG.md and stop.
  a. paper/STOP exists -> "STOP present".
  b. paper/GOAL_MET.md exists -> "goal already met".
  c. Lock: read paper/HOP_LOCK.json (absent = free). If started_at is younger than
     ttl_minutes (default 18) -> "lock held by <holder>". Otherwise write
     {"holder": "hop-<HHMM>", "started_at": "<local ISO 8601>", "ttl_minutes": 18}.
  d. paper/QUEUE.md has no row marked NEXT or pending -> write paper/GOAL_MET.md, log
     "GOAL MET: queue drained", delete the lock, stop.

STEP 1 - goal condition. GOAL holds when every row T0..T7 in paper/QUEUE.md is DONE
  AND paper/CLAIMS.md contains no claim whose status is open. If GOAL holds, write
  paper/GOAL_MET.md (timestamp + sha256 of STATUS.json and CLAIMS.md), log "GOAL MET",
  delete the lock, stop.

STEP 2 - take ONLY the head unfinished task: first row marked NEXT, else first pending.

STEP 3 - obey paper/OVERNIGHT_CONTRACT.md. Non-negotiable: preregister before
  computing; never move a seed, margin or percentile to turn FAIL into PASS; never read
  the official test files; never unseal FD003/FD004; never delete artefacts; never
  commit, push or publish; no hand-written numbers; FAIL and INDETERMINATE are valid
  outcomes. Items blocked on the human are recorded as pending and skipped - never wait
  for input.

STEP 4 - finish the task with artefacts, then:
  - one row in paper/RUNLOG.md (task | UTC | command | exit | artefacts | sha | verdict)
  - update paper/STATUS.json (next_action, gaps, round.last_completed)
  - mark the row in paper/QUEUE.md
  - one line in paper/DRIVER_LOG.md (hop label, task id, verdict, wall-clock minutes)

STEP 5 - delete paper/HOP_LOCK.json, then stop immediately. Do not start the next task.

Budget: keep the whole hop under 15 minutes of wall clock. If a command would exceed
it, write what exists as an explicit partial verdict and stop.
```

---

## Trigger plan (2026-10-30)

Hops are one-shot automations at 20-minute spacing until the queue drains; after that
every hop exits at STEP 0d and costs one guard turn. Paused rather than deleted so the
cadence can be re-armed by editing `scheduledAt`.

| hop | local time | state |
|---|---|---|
| hop-0040 | 00:40 | armed |
| hop-0100 | 01:00 | armed |
| hop-0120 | 01:20 | armed |
| hop-0140 | 01:40 | armed |
| hop-0200 | 02:00 | armed |

## Stop and safety

* `paper/STOP` - kills all future hops at STEP 0a.
* `paper/HOP_LOCK.json` - 18-minute TTL, shorter than the 20-minute cadence, so a
  crashed hop self-heals at the next one instead of stalling the night (the old
  90-minute lock suppressed four consecutive hops).
* Two consecutive hops whose `STATUS.json` sha256 is unchanged -> write `paper/STOP`
  (the contract's circuit breaker, now enforceable by machine rather than by memory).
* The desktop client must stay running: local automations do not fire when the app is
  not on.
