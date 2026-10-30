#!/usr/bin/env python
"""E1 reference verification (T6).

Phase 1: resolve each candidate DOI against OpenAlex, reconstruct the abstract
from its inverted index, and dump title/authors/year/venue/abstract to JSON.
Nothing is written into refs_verified.csv here.

Usage:
    python scripts/e1_verify_refs.py fetch
"""
import json
import sys
import urllib.request
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dsh" / "e1"
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    "zheng2017": "10.1109/icphm.2017.7998311",
    "li2018": "10.1016/j.ress.2017.11.021",
    "cohen1992": "10.1037/0033-2909.112.1.155",
    "lei2018": "10.1016/j.ymssp.2017.11.016",
    "li2014oc": "10.1109/icphm.2014.7036396",
    "freitas2026": "10.5220/0015076000004091",
    "dodge2019": "10.48550/arXiv.1909.03004",
    "raff2019": "10.48550/arXiv.1909.06674",
    "ress2019": "10.1016/j.ress.2018.11.027",
}


def fetch_one(doi):
    url = "https://api.openalex.org/works/https://doi.org/" + doi
    req = urllib.request.Request(url, headers={"User-Agent": "e1-verify/1.0 (mailto:dev@example.org)"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)


def reconstruct(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def _authors(display_names):
    """'Shuai Zheng' -> 'Zheng, S.'  (family = last token)."""
    out = []
    for n in display_names:
        parts = n.split()
        if len(parts) < 2:
            out.append(n)
            continue
        family = parts[-1]
        initials = "".join(p[0] + "." for p in parts[:-1])
        out.append(f"{family}, {initials}")
    return "; ".join(out)


# (key, verbatim abstract substring, role). The substring is asserted against the
# fetched abstract before the row is written - a retyped or paraphrased quote fails.
ROWS = [
    ("zheng2017",
     "In this work, we propose a Long Short-Term Memory (LSTM) approach for RUL estimation, which can make full use of the sensor sequence information",
     "method-side C-MAPSS baseline (LSTM line); cited so SS2 can describe the field without endorsing the reported gains"),
    ("ress2019",
     "The advantages of the proposed semi-supervised setup have been verified on the popular C-MAPSS dataset.",
     "second method-side C-MAPSS baseline (semi-supervised deep architecture); the comparison practice our resolution argument applies to"),
    ("li2014oc",
     "This paper presents a prognostics approach based on operating condition for estimating the Remaining Useful Life (RUL).",
     "peer-reviewed precedent treating operating condition as an explicit modelling variable; the confound our decomposition holds fixed"),
    ("cohen1992",
     "The sample sizes necessary for .80 power to detect effects at these levels are tabled for eight standard statistical tests",
     "source of the required-n / power relation behind the minimum detectable effect in SS3; keeps the MDE machinery from being self-invented"),
    ("dodge2019",
     "we demonstrate that test-set performance scores alone are insufficient for drawing accurate conclusions about which model performs best",
     "evidence that single test-set scores are insufficient for ranking models; our engine-level MDE is the C-MAPSS analogue"),
    ("raff2019",
     "We take the first step toward a quantifiable answer by manually attempting to implement 255 papers published from 1984 until 2017",
     "precedent that reproducibility is measured rather than asserted; motivates the artefact+sha256 evidence standard"),
]


def write_rows():
    import csv
    meta_path = OUT / "refs_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    csv_path = ROOT / "paper" / "refs_verified.csv"

    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    have = {r["key"] for r in rows}

    added = 0
    for key, quote, role in ROWS:
        m = meta[key]
        abs_ = m.get("abstract") or ""
        if quote not in abs_:
            raise SystemExit(f"QUOTE NOT VERBATIM for {key}: {quote[:60]!r}")
        if key in have:
            print(f"[SKIP] {key} already present")
            continue
        rows.append({
            "key": key,
            "title": m["title"].rstrip("."),
            "authors": _authors(m["authors"]),
            "year": m["year"],
            "venue": m["venue"] or "",
            "doi_or_url": "https://doi.org/" + m["doi"],
            "verified_on": "2026-10-30",
            "evidence_quote": quote,
            "role": role,
        })
        added += 1
        print(f"[ADD ] {key}")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)

    print(f"added={added} total_rows={len(rows)}")
    if not (12 <= len(rows) <= 15):
        raise SystemExit(f"ACCEPTANCE BREACH: total {len(rows)} outside [12,15]")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "write":
        write_rows()
        return
    recs = {}
    for key, doi in CANDIDATES.items():
        try:
            w = fetch_one(doi)
        except Exception as e:  # network / 404
            recs[key] = {"doi": doi, "error": repr(e)}
            print(f"[ERR ] {key} {doi} {e!r}")
            continue
        recs[key] = {
            "doi": doi,
            "title": w.get("title"),
            "year": w.get("publication_year"),
            "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "authors": [a["author"]["display_name"] for a in w.get("authorships", [])],
            "abstract": reconstruct(w.get("abstract_inverted_index")),
        }
        print(f"[OK  ] {key} {recs[key]['title']} ({recs[key]['year']}) abs_len={len(recs[key]['abstract'])}")

    (OUT / "refs_metadata.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")
    print("wrote", OUT / "refs_metadata.json")


if __name__ == "__main__":
    main()
