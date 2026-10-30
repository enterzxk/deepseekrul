# OVERNIGHT CONTRACT — 2026-10-29 23:40 → 2026-10-30 09:00

Unattended execution rules. This file overrides the longer round instructions in
`paper/ROUND_PROMPT.md` for the overnight window.

## Execution discipline

1. **One task at a time**: take only the head of `paper/QUEUE.md`, finish it, write
   `paper/RUNLOG.md` + `paper/STATUS.json`, then stop this round. Never chain two
   tasks in one round.
2. **Register before computing**: any task that can produce a number gets a
   `paper/*_PREREGISTRATION.md` (or an entry in the existing one) **and its sha256
   recorded** before the run starts.
3. **FAIL is a valid outcome.** A verdict may be FAIL, INDETERMINATE, or PASS.
   It may never be moved by changing the RNG seed, the percentile, or the margin.

## Hard prohibitions

* Reading official test files; unsealing FD003/FD004.
* Deleting artefacts (withdrawn numbers stay, labelled).
* Editing a pre-registration's original text (amendments go in §12 only).
* Committing, pushing, or publishing anything.
* Writing FAIL as PASS, or a CI that crosses zero as a gain.
* Fabricating metrics, selecting seeds, or re-running to "get a better draw".

## Blocked-on-human items

Authors/affiliations/funding/COI, the test-disclosure wording, and the 260-vs-509
choice are recorded as `pending` in STATUS and **do not block** any task that does
not depend on them. If a command needs interactive approval, log it and skip; do
not wait.

## Stop conditions

09:00 local time; existence of `paper/STOP`; two consecutive rounds with an
unchanged `STATUS.json` hash; T0 judged INDETERMINATE with no path forward; the GPU
host unreachable after two retries (skip T1/T2, continue T3/T4).

## Evidence standard

Every completed task leaves: the command(s), the exit code, the artefact paths, the
sha256 of any new code, and the verdict. A task without artefacts is not done.
