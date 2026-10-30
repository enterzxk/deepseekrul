# E1 PREREGISTRATION — reference closure (T6)

Written and hashed **before** any search or verification call of this hop.
Contract: `paper/OVERNIGHT_CONTRACT.md` §2 (register before computing).

## Scope

Close the E1 gap recorded in `paper/E1_LITERATURE.md` §5 and `STATUS.json`
(`gaps.E1`: "6 refs verified … 6-9 more refs needed") by extending
`paper/refs_verified.csv` from 6 to 12–15 rows.

## Acceptance criteria (fixed before the run)

1. A row exists only if **all** of the following were obtained for that item:
   a. a resolvable DOI or stable URL (arXiv abs page / publisher landing page);
   b. author list, year, venue, and title as printed by that landing page;
   c. one **verbatim** sentence from the abstract, copied character-for-character,
      that the paper is allowed to lean on;
   d. an explicit `role` naming which part of §2 or §3 the citation supports.
2. Total verified rows must land in **[12, 15]** inclusive.
3. Coverage requirement: at least **two** method-side C-MAPSS baselines, at least
   **one** peer-reviewed operating-condition-normalisation source, and at least
   **one** source for the minimum-detectable-effect / power machinery.
4. Items whose metadata cannot be resolved are moved to the rejected table in
   `paper/E1_LITERATURE.md` §4 with the reason. They are **not** counted.

## Prohibitions carried from the contract

* No citation without a verbatim abstract sentence — a plausible-looking DOI is
  not evidence.
* No DOI reconstructed from memory or pattern; every DOI must be read off a page
  fetched during this run.
* No priority ("first") claim, no citation of non-peer-reviewed blog content for
  factual claims.
* No number in the table is hand-written from memory; counts in the result note
  are produced by a command whose output is kept.

## Verdict definitions

* **PASS** — 12–15 rows, every row meets 1a–1d, coverage §3 satisfied.
* **PARTIAL** — rows verified but count < 12, or coverage §3 unmet; the shortfall
  is stated numerically and the gap stays open in STATUS.
* **FAIL** — fewer than 6 *new* verifiable items could be established; E1 stays
  open with the search log recording which queries failed.

## Budget

Whole hop < 15 min wall clock. If the verification round would overrun, the hop
writes PARTIAL with the rows already verified and stops (contract §Execution 3).

## Recorded hash

This file's sha256 is written into `paper/RUNLOG.md` for T6 at completion.
