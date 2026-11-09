"""Programmatic citation audit for references_v2.bib.

For every entry, query OpenAlex (by DOI when present, otherwise by title) and compare
the returned title and year with the .bib.  This catches hallucinated authors/venues/
years and stale metadata without relying on an LLM.

Similarity is a difflib ratio on normalised titles (lowercase, alphanumerics only).
Year tolerance is +/-1 (online-first vs issue year).

Exit 0 iff every entry matches.  Network: api.openalex.org, no key required.

Usage: python scripts/citation_audit.py [--bib PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

TITLE_OK = 0.90


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def parse_bib(text: str) -> list[dict]:
    entries = []
    for m in re.finditer(r"@(\w+)\{\s*([A-Za-z0-9_]+)\s*,(.*?)\n\}", text, re.S):
        kind, key, body = m.group(1), m.group(2), m.group(3)
        fields = {}
        # a field value ends either at a comma or at the entry-closing brace
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}(?=\s*,|\s*$)", body, re.S):
            fields[fm.group(1).lower()] = " ".join(fm.group(2).split())
        entries.append(dict(kind=kind, key=key, **fields))
    return entries


def fetch(url: str, tries: int = 3) -> dict | None:
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": "citation-audit/1.0 (research)"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < tries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            print(f"    fetch failed: HTTP {e.code}")
            return None
        except Exception as e:
            print(f"    fetch failed: {type(e).__name__}: {e}")
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", default="docs/manuscripts/ijphm_v2/references_v2.bib")
    args = ap.parse_args()
    entries = parse_bib(Path(args.bib).read_text(encoding="utf-8"))
    print(f"entries: {len(entries)}\n")

    ok = True
    for e in entries:
        title = e.get("title", "")
        year = e.get("year", "")
        doi = e.get("doi", "")
        if doi:
            rec = fetch(f"https://api.openalex.org/works/https://doi.org/{doi}"
                        f"?select=title,publication_year,doi")
            src = f"doi:{doi}"
        else:
            q = urllib.parse.quote(title)
            res = fetch(f"https://api.openalex.org/works?search={q}&per-page=1"
                        f"&select=title,publication_year,doi")
            rec = (res or {}).get("results", [None])[0]
            src = "title search"
        time.sleep(0.3)

        if not rec:
            print(f"FAIL {e['key']:<22} no OpenAlex record ({src})")
            ok = False
            continue

        ratio = SequenceMatcher(None, norm(title), norm(rec.get("title", ""))).ratio()
        year_rec = rec.get("publication_year")
        try:
            year_ok = abs(int(year) - int(year_rec)) <= 1
        except (TypeError, ValueError):
            year_ok = False
        passed = ratio >= TITLE_OK and year_ok
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'} {e['key']:<22} "
              f"title_sim={ratio:.3f} year {year}->{year_rec}  [{src}]")
        if not passed:
            print(f"     bib : {title[:90]}")
            print(f"     oa  : {(rec.get('title') or '')[:90]}")

    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}  ({len(entries)} entries)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
