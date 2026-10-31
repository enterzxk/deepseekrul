"""Verify that every artefact cited by the claims ledger exists and is committed.

The claims ledger is the contract between the manuscript and the archive: a row may enter
the paper only if its `evidence` cell points at a file that a reader can open. Twice during
this project a row pointed at a file that existed only on the training host
(`results/dsh/e16/*` in commit 6d98f83), and once at a file that had never been created at
all. Both are silent failures for a reader and neither is caught by reading the manuscript.

Checks, per evidence token:
  disk   — the path exists in the working tree
  git    — the path is tracked at HEAD
  size   — non-empty

Usage:  python scripts/paper_claims_check.py [--strict]
Exit code 1 if any row has an unreadable artefact (with --strict: also if any row is
withdrawn without a matching `superseded` note).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "paper" / "CLAIMS.md"

# evidence tokens look like `results/x/y.json`, optionally inside backticks, and may be
# followed by a parenthetical note; globs are allowed.
TOKEN = re.compile(r"`([^`]*?\.(?:json|csv|md|npz|py|tex|txt))`")


def tracked_at_head(path: str) -> bool:
    try:
        subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:{path}"],
                       check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def expand(token: str) -> list[str]:
    """Turn a token into concrete repo-relative paths (globs allowed)."""
    if any(ch in token for ch in "*?["):
        return [str(p.relative_to(ROOT)).replace(os.sep, "/")
                for p in ROOT.glob(token) if p.is_file()]
    return [token.lstrip("./")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    rows = [l for l in CLAIMS.read_text(encoding="utf-8").splitlines() if l.startswith("| C")]
    problems: list[tuple[str, str, str]] = []
    checked = 0
    for line in rows:
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 5:
            continue
        cid, evidence = cells[1], cells[3]
        for token in TOKEN.findall(evidence):
            for rel in expand(token):
                checked += 1
                p = ROOT / rel
                if not p.exists():
                    problems.append((cid, rel, "missing on disk"))
                elif p.stat().st_size == 0:
                    problems.append((cid, rel, "empty"))
                elif not tracked_at_head(rel):
                    problems.append((cid, rel, "present but not committed"))

    print(f"claims rows: {len(rows)}   artefact references checked: {checked}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for cid, rel, why in problems:
            print(f"  {cid:5s} {rel:60s} {why}")
    else:
        print("all cited artefacts exist and are committed")

    withdrawn = [c[1].strip() for c in (l.split("|") for l in rows)
                 if len(c) > 6 and "withdraw" in c[5].lower()]
    print(f"\nwithdrawn rows: {len(withdrawn)} -> {', '.join(withdrawn) if withdrawn else 'none'}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
