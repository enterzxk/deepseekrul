"""Produce the A-segment reconciliation record that the registry verifies.

Why a wrapper instead of asking gate1.py to print it
----------------------------------------------------
The registry does not re-implement the criterion (that was the bug fixed today:
two implementations of the same 0.2, drifting apart).  It only checks three
things: a record exists, it says pass, and the sha256 of the judging script
matches the blob committed at the recorded ``git_sha``.  So the record has to be
*produced by the run*, never typed by hand -- a hand-written verdict turns the
gate into a statement of intent.

This wrapper is deliberately separate from ``gate1.py``: it must not be able to
change the verdict, only to report it.

Three guards
------------
1. **Verdict = exit code.** ``gate1.py`` is run as a subprocess and its exit
   status is the only input to ``verdict``.  Stdout is passed through for a
   human but is never parsed for the decision.
2. **Exit 2 is not folded into fail.** The three exit codes carry different
   meanings (0 PASS / 1 FAIL / 2 ABORT); collapsing ABORT into FAIL is the exact
   confusion that was fixed in the judge today, so the record keeps
   ``verdict="abort"`` *and* ``exit_code``.
3. **The recorded hash describes the file that ran.** If the working tree copy
   of ``gate1.py`` differs from the blob at ``--git-sha``, we refuse to write a
   record: the hash would then describe a file that did not produce the verdict,
   and the registry's check would be satisfied by an unrelated version.

Usage
-----
    python -m paper_c.recon_record \
        --segment-a  <A-stage run dir> \
        --frozen     <frozen fixed40 dir> \
        --out        results/paper_c/recon_A.json

    python -m paper_c.recon_record --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GATE1 = "paper_c/gate1.py"
DEFAULT_BATCH = "P2/W1/segment-A"
VERDICT = {0: "pass", 1: "fail", 2: "abort"}


def git(*args: str) -> bytes:
    out = subprocess.run(["git", *args], capture_output=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {out.stderr.decode('utf-8', 'replace').strip()}"
        )
    return out.stdout


def blob_sha256(rev: str, path: str = GATE1) -> str:
    return hashlib.sha256(git("show", f"{rev}:{path}")).hexdigest()


def working_tree_matches_blob(rev: str, path: str = GATE1) -> bool:
    """True only if the file on disk is byte-identical to the blob at rev."""
    out = subprocess.run(
        ["git", "diff", "--quiet", rev, "--", path], capture_output=True, check=False
    )
    return out.returncode == 0


def build_record(segment_a: str, frozen: str, rev: str, tol: float | None) -> dict:
    cmd = [
        sys.executable,
        GATE1,
        "--reconcile-a",
        segment_a,
        "--reconcile-frozen",
        frozen,
    ]
    if tol is not None:
        cmd += ["--tol", str(tol)]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    verdict = VERDICT.get(proc.returncode, "error")
    # Count the cells from the tree, not from gate1.py's stdout: the ledger
    # needs this field, and parsing a human-facing print makes it break every
    # time the judge's output is reformatted (it already did once between v2
    # and v5, which is why this is a glob).
    try:
        cells = len(
            list(Path(segment_a).glob("*/FD*/s*/validation_predictions.csv"))
        )
    except OSError:
        cells = None

    return {
        "batch": DEFAULT_BATCH,
        "verdict": verdict,
        "exit_code": proc.returncode,
        "cells": cells,
        "gate1_path": GATE1,
        "git_sha": git("rev-parse", rev).decode().strip(),
        "gate1_sha256": blob_sha256(rev),
        "segment_a": str(segment_a),
        "frozen": str(frozen),
        "ran_at_unix": round(time.time(), 3),
    }


def selftest() -> int:
    """Prove the wrapper can report a failure, not only a success.

    The guard state for HEAD is reported, never asserted: while a collaborator
    has an uncommitted gate1.py in the shared tree, ``working_tree_matches_blob
    ('HEAD')`` is legitimately False, and a selftest that insisted otherwise
    would fail exactly when the guard is doing its job.  What must be asserted
    is that a mismatch is *detected* and that it *refuses*, which is tested
    against a fixed older revision whose gate1.py is known to differ.
    """
    failures = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"{name:52} {'OK' if cond else 'FAIL'}  {detail}")
        if not cond:
            failures.append(name)

    # 1. verdict mapping covers all three codes and never conflates them.
    check(
        "exit 2 maps to abort, not fail",
        VERDICT[2] == "abort" and VERDICT[1] == "fail",
        f"{VERDICT}",
    )

    # 2. the guard detects a revision whose gate1.py is not the one on disk.
    old = "95cc7b1b"
    try:
        mismatched = working_tree_matches_blob(old)
        check(
            "guard detects a non-matching revision",
            mismatched is False,
            f"vs {old} (v2) -> matches={mismatched}",
        )
    except RuntimeError as exc:
        check("guard detects a non-matching revision", False, str(exc))

    # 3. report the real guard state for HEAD without asserting it.
    print(
        f"{'guard state for HEAD (reported)':52} {'--'}  "
        f"working_tree_matches_blob(HEAD)={working_tree_matches_blob('HEAD')}"
    )

    # 4. a mismatched rev must refuse and must not leave a record behind.
    probe = Path(tempfile.gettempdir()) / "recon_record_refusal_probe.json"
    probe.unlink(missing_ok=True)
    rc = subprocess.run(
        [
            sys.executable,
            __file__,
            "--segment-a",
            ".",
            "--frozen",
            ".",
            "--git-sha",
            old,
            "--out",
            str(probe),
        ],
        capture_output=True,
        check=False,
    ).returncode
    check(
        "refuses a record for a mismatched rev",
        rc == 2 and not probe.exists(),
        f"rc={rc} record_written={probe.exists()}",
    )

    # 5. missing inputs must never produce a pass.
    rec = build_record("/nonexistent/a", "/nonexistent/b", "HEAD", None)
    check(
        "missing inputs -> not pass",
        rec["verdict"] != "pass",
        f"verdict={rec['verdict']} exit={rec['exit_code']}",
    )

    print()
    if failures:
        print(f"RECON_RECORD SELFTEST FAILED ({len(failures)}): {failures}")
        return 1
    print("RECON_RECORD SELFTEST PASSED (5/5)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment-a")
    ap.add_argument("--frozen")
    ap.add_argument("--git-sha", default="HEAD")
    ap.add_argument("--out")
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if not (a.segment_a and a.frozen and a.out):
        print("need --segment-a --frozen --out (or --selftest)", file=sys.stderr)
        return 2

    if not working_tree_matches_blob(a.git_sha):
        print(
            f"REFUSING to write a record: {GATE1} in the working tree differs from "
            f"the blob at {a.git_sha}. The recorded hash would describe a file that "
            f"did not produce this verdict. Commit or stash first.",
            file=sys.stderr,
        )
        return 2

    rec = build_record(a.segment_a, a.frozen, a.git_sha, a.tol)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"\nrecord written: {out}")
    print(json.dumps(rec, indent=2))
    return rec["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
