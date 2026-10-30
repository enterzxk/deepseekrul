# SCHEDULER_PROBE — overnight driver, 2026-10-29 23:49–23:56

Goal: get a repeating driver that executes one queue task per hop without a human.
Evidence only; nothing here is a claim that "the scheduler is running" beyond what
the logs show.

## Attempt 1 — WorkBuddy CLI (preferred path)

```
$env:ELECTRON_RUN_AS_NODE=1
& "C:\Program Files\WorkBuddy\WorkBuddy.exe" ^
  "C:\Program Files\WorkBuddy\resources\app.asar.unpacked\cli\dist\codebuddy-lite-wb.mjs" ^
  -p "Reply with exactly: OK" --output-format text
```

Result:

```
Error: listen EADDRINUSE: address already in use 127.0.0.1:5738
EXIT=1
```

Retried with `--port 5799`:

```
Authentication required. Please use /login command to sign in to your account
EXIT=0
```

`--output-format text` prints nothing on the auth failure and the process exits 0,
so a driver loop keyed on the exit code would treat an unauthenticated run as
success. That alone disqualifies this path.

**Decision: do not touch the desktop session token; abandon the CLI path.**

## Attempt 2 — automation (fallback in the brief)

Dry run: a one-shot automation scheduled at 23:51 whose only action is to append a
line to `paper/RUNLOG.md`. Evidence:

```
-rw-r--r-- 1 20562 197609 36 Oct 29 23:51 paper/RUNLOG.md
- 2026-10-29T15:51Z DRIVER PROBE OK
```

One hop verified: the automation fired on time, a fresh session executed the task,
and the artefact landed. This is the only positive scheduler evidence we have.

## Recurring driver

Created `Overnight paper driver (hourly)`, `FREQ=HOURLY;BYMINUTE=5`,
`validUntil 2026-10-30T09:00`.

Deviation from the brief, stated explicitly: **20-minute cadence is not
expressible** in the automation schema (DAILY / HOURLY / WEEKLY / MONTHLY / YEARLY
only). One hour is the finest granularity available, so the queue advances at most
once per hour while the interactive worker is idle.

## Concurrency guard

`paper/LOCK` holds an ISO timestamp. The driver's first action is to exit if that
timestamp is younger than 90 minutes. The interactive worker refreshes it while it
runs; if the interactive worker dies, the lock expires (23:53 + 90 min = 01:23) and
the driver takes over automatically.

Consequence, honestly: the 00:05 hop will almost certainly be **suppressed by the
lock**, so it will not produce a second "STATUS advanced" observation. The guard is
doing its job; the cost is that the two-consecutive-hops check the brief asked for
cannot be completed until the interactive worker stops touching the lock.

## What would make this fail silently

* The automation host losing the workspace path.
* A hop starting while a GPU batch from the previous hop is still running — the
  driver would see the same queue head and re-launch it. Mitigation: GPU tasks must
  write a DONE marker and the queue row is updated only after the batch finishes.

---

# Resolution — 2026-10-30 00:12–00:20 (round 6 follow-up)

## Attempt 1 re-diagnosed (not retried blindly)

The binary the probe called is the desktop's own bundled CLI:
`…\app.asar.unpacked\cli\bin\codebuddy`, version **2.165.0**, a `lite-wb` build.

* It has **no `login` subcommand** (commands list: config, mcp, sandbox, plugin,
  project, doctor, update, install, cleanup, daemon, ps, logs, attach, kill, rm,
  respawn, agents, auto-mode). The `Authentication required` message therefore asks
  for the *TUI slash command* `/login`, which needs a TTY and a browser round-trip —
  not automatable from a headless run.
* Run standalone it tries to **bind the desktop's service port**
  (`EADDRINUSE 127.0.0.1:59294`, earlier 5738); `CODEBUDDY_SERVICE_PROXY_URL` is an
  inbound hook endpoint of the app, not an outbound credential.
* With `node` instead of `ELECTRON_RUN_AS_NODE` it also raises
  `fetch config failed … setProxy: Right-hand side of 'instanceof' is not callable`,
  i.e. the bundle expects the Electron host.
* Credentials are injected as `CODEBUDDY_GATEWAY_PASSWORD` + `CODEBUDDY_GATEWAY_AUTH`
  for the duration of a desktop-spawned process. There is no portable token file.

`GoalService`, `goal-progress` and `"Goal achieved"` **are** present in the bundle, so
`/goal` exists in this build; it is unreachable only for want of a standalone
credential. **A single human `/login` in the TUI is the whole unblock.** No desktop
token was read, copied or written.

## Attempt 2 measured — the cadence finding was wrong in a useful way

| probe | sent | returned `nextRunAt` | reading |
|---|---|---|---|
| multi-value BYMINUTE | `FREQ=HOURLY;INTERVAL=1;BYMINUTE=5,25,45` | now + 3570 s (00:17:14 → 01:16:44) | accepted, but `BYMINUTE` is **ignored**; an hourly rule means "60 min after arming" |
| one-shot hops ×5 | 00:40, 01:00, 01:20, 01:40, 02:00 | 1793292000, 1793293200, 1793294400, 1793295600, 1793296800 | honoured to the minute, exactly 1200 s apart |

**Correction to the deviation recorded above.** "Hourly is the finest granularity" is
true only for a *recurrence rule*. Discrete one-shot automations accept arbitrary
spacing, so the 20-minute cadence is expressible after all — it just cannot be written
as one RRULE. The hourly driver was therefore not the best available scheduler; it was
the best *single rule*.

## Design change (paper/DRIVER_PROMPT.md)

1. **Cadence**: five one-shot hops at 20-minute spacing (00:40 → 02:00), each taking
   only the queue head. The hourly automation is **paused**, not deleted, and carries
   the same prompt so it can be re-armed by editing one field.
2. **Goal semantics**: no goal object exists, so the goal is a predicate checked at the
   head of every hop — all of T0–T7 `DONE` **and** no claim in `CLAIMS.md` still open.
   On success the hop writes `paper/GOAL_MET.md`; every later hop exits at guard (b).
3. **Lock TTL 90 → 18 min.** The old TTL was four times the cadence, so one dead hop
   suppressed four successors (exactly what happened at 00:05). At 18 min a crashed hop
   self-heals at the next one. Lock moved to `paper/HOP_LOCK.json` to avoid inheriting
   the old file's semantics.
4. **Circuit breaker made enforceable**: two consecutive hops with an unchanged
   `STATUS.json` sha256 → write `paper/STOP`. Previously this rule lived only in prose.
5. **Hop budget 15 min**: a hop that would overrun writes an explicit partial verdict
   instead of colliding with the next hop.

## Residual risks

* Local automations need the desktop client running; if it is closed the night stops
  silently. Cloud-side scheduling would remove this, but moves the runtime away from
  `C:\Users\20562\Desktop\deepseek-rul`.
* A hop is a full agent session: cost scales with cadence, not with queue depth. The
  queue has two writing tasks left (T6, T7), so five hops is deliberate over-provisioning
  for retries, not a sustained rate.
* If two hops ever start within the same minute, the lock file write is not atomic.
  Probability is low (20-minute spacing) and the failure is one duplicated writing task,
  not duplicated GPU work.
