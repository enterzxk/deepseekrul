"""Consistency gate for the IJPHM v2 manuscript drafts.

Three checks, all cheap and all meant to fail loudly:

  1. LEDGER COVERAGE  -- every claim id C1..C15 in CLAIM_LEDGER.md is cited at least once in a draft
                         (as an HTML comment <!-- C# -->), and no draft cites an unknown id.
  2. NUMBER ANCHORS   -- a curated set of manuscript numbers is present verbatim, and the value is
                         re-derived from the JSON produced by the evidence scripts (so a stale draft
                         fails when the underlying table changes).
  3. BANNED PHRASES   -- wording that D022/D023 forbade (over-claiming) is flagged for manual review.

Exit code 0 iff checks 1 and 2 pass (check 3 is advisory, printed for a human).

Usage: python scripts/check_manuscript_consistency.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(".")
MDIR = ROOT / "docs/manuscripts/ijphm_v2"
LEDGER = MDIR / "CLAIM_LEDGER.md"
DRAFTS = ["01_intro_related.md", "03_protocol.md", "04_results.md", "05_discussion.md", "06_appendix.md"]
EVID = ROOT / "tmp/freeze/paper_evidence_seeded.json"
H005 = ROOT / "tmp/freeze/h005_v2_corrected.json"

BANNED = [
    "first to", "the first", "only work", "unique", "novel method", "novelty of the method",
    "systematically overestimate", "guarantee is violated", "violates the guarantee",
    "unidentifiable", "cannot be identified", "impossible", "proves that",
]


def norm(s: str) -> str:
    """Normalise Unicode minus/en-dash so that '−0.712' matches '-0.712'."""
    return s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")


def main() -> int:
    text = {f: (MDIR / f).read_text(encoding="utf-8") for f in DRAFTS if (MDIR / f).exists()}
    all_text = "\n".join(text.values())
    all_norm = norm(all_text)
    ok = True

    # ---- 1. ledger coverage -------------------------------------------------
    ledger_ids = sorted({int(m) for m in re.findall(r"\|\s*\*\*C(\d+)\*\*\s*\|", LEDGER.read_text(encoding="utf-8"))})
    cited = sorted({int(m) for m in re.findall(r"<!--\s*C(\d+)\s*-->", all_text)})
    missing = [c for c in ledger_ids if c not in cited]
    unknown = [c for c in cited if c not in ledger_ids]
    print(f"[1] ledger ids: {ledger_ids}")
    print(f"    cited in drafts: {cited}")
    if missing:
        print(f"    FAIL missing from drafts: {missing}")
        ok = False
    if unknown:
        print(f"    FAIL cited but not in ledger: {unknown}")
        ok = False
    if not missing and not unknown:
        print("    PASS every ledger claim is cited, no orphan citations")

    # ---- 2. number anchors --------------------------------------------------
    ev = json.loads(EVID.read_text(encoding="utf-8")) if EVID.exists() else {}
    h5 = json.loads(H005.read_text(encoding="utf-8")) if H005.exists() else {}
    v = h5.get("verdict", {})
    anchors = [
        ("C1 cells", f"{ev['C1']['cells_wins']}/80"),
        ("C1 folds", f"{ev['C1']['folds_wins']}/20"),
        ("C1 mean", f"{abs(ev['C1']['cells_mean']):.2f}"),
        ("C2 folds", f"{ev['C2']['folds_wins']}/20"),
        ("C3 diff", f"{abs(ev['C3']['diff_pp']):.2f}"),
        ("C3 max cell", f"{ev['C3']['max_abs_pp']:.2f}"),
        ("C4b FD004", f"{ev['C4b']['FD004']:.4f}"),
        ("C4b FD002", f"{ev['C4b']['FD002']:.4f}"),
        ("C6 mean", f"{ev['C6']['mean']:.2f}"),
        ("C6 FD002 mean", f"{abs(ev['C6']['per_subset']['FD002']['mean']):.2f}"),
        ("C6 FD003 mean", f"{ev['C6']['per_subset']['FD003']['mean']:.2f}"),
        ("C8 multi", f"{ev['C8']['multi_better']}/10"),
        ("C8 single", f"{ev['C8']['single_better']}/10"),
        ("C8 p", f"{ev['C8']['fisher_p']:.4f}"),
        ("H005 G-A rho", f"{abs(v.get('spearman_A', 0)):.3f}"),
        ("H005 G-A CI", f"{v.get('ci_A', [0, 0])[1]:.3f}"),
        ("H005 G-C CI", f"{v.get('ci_C', [0, 0])[0]:.3f}"),
        ("H005 G-D mean", f"{v.get('rule_mean', 0):.2f}"),
        ("H005 G-D FD003", f"{v.get('rule_fd003', 0):.2f}"),
    ]
    print("\n[2] number anchors")
    for name, tok in anchors:
        hit = norm(tok) in all_norm
        print(f"    {'PASS' if hit else 'FAIL'} {name}: '{tok}'")
        if not hit:
            ok = False

    # ---- 4. citation keys ---------------------------------------------------
    bib = MDIR / "references_v2.bib"
    if bib.exists():
        bib_text = bib.read_text(encoding="utf-8")
        bib_keys = set(re.findall(r"@\w+\{\s*([A-Za-z0-9_]+)\s*,", bib_text))
        cited_keys: set[str] = set()
        for m in re.finditer(r"\[@([^\]]+)\]", all_text):
            for k in m.group(1).split(";"):
                cited_keys.add(k.strip().lstrip("@"))
        missing_keys = sorted(k for k in cited_keys if k not in bib_keys)
        unused = sorted(k for k in bib_keys if k not in cited_keys)
        print(f"\n[4] citation keys: bib={len(bib_keys)} cited={len(cited_keys)}")
        if missing_keys:
            print(f"    FAIL cited but absent from bib: {missing_keys}")
            ok = False
        if unused:
            print(f"    WARN in bib but never cited: {unused}")
        if not missing_keys:
            print("    PASS every cited key exists in references_v2.bib")

    # ---- 3. banned phrasing (advisory) -------------------------------------
    print("\n[3] banned phrasing (advisory)")
    low = all_text.lower()
    hits = []
    for ph in BANNED:
        for m in re.finditer(re.escape(ph), low):
            s = max(0, m.start() - 60)
            hits.append((ph, all_text[s:m.end() + 60].replace("\n", " ")))
    if not hits:
        print("    PASS no banned phrase found")
    else:
        for ph, ctx in hits[:12]:
            print(f"    REVIEW '{ph}': ...{ctx}...")

    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
