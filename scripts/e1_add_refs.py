#!/usr/bin/env python
"""E1 expansion — append the resolved candidates to paper/refs_verified.csv.

Each row carries a verbatim sentence from the abstract that OpenAlex returned. The
sentence is asserted as an exact substring before the row is written; a retyped or
paraphrased quote aborts the whole run. Rows whose abstract is empty are recorded
in results/dsh/e1/refs_metadata_only.json instead (they may be cited but never
quoted).

Usage:
    python scripts/e1_add_refs.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDS = ROOT / "results" / "dsh" / "e1" / "refs_candidates.json"
CSV_PATH = ROOT / "paper" / "refs_verified.csv"
VERIFIED_ON = "2026-10-31"

# key -> (verbatim abstract substring, role in the manuscript)
ROWS = [
    ("saxena2008cmapss",
     "response surfaces of all sensors are generated via a thermo-dynamical simulation model for the engine",
     "origin of the C-MAPSS run-to-failure simulator; anchors what the benchmark data actually are"),
    ("saxena2010metrics",
     "prognostics concepts lack standard definitions and suffer from ambiguous and inconsistent interpretations",
     "source of the offline prognostic metrics, including the asymmetric penalty behind the NASA score"),
    ("heimes2008",
     "The solution utilizes an advanced recurrent neural network architecture to estimate the remaining useful life of the system.",
     "the PHM08 challenge runner-up RNN; the earliest point on the RNN line the field keeps comparing against"),
    ("lakens2017",
     "In equivalence tests, such as the Two One-Sided Tests (TOST) procedure discussed in this article, an upper and lower equivalence bound is specified based on the smallest effect size of interest.",
     "the TOST procedure and the smallest-effect-size-of-interest rule used in the equivalence analysis"),
    ("holm1979",
     "It is shown that the test has a prescribed level of significance protection against error of the first kind for any combination of true hypotheses.",
     "the Holm step-down correction named when the multiplicity audit reports that nothing survives"),
    ("benjamini1995",
     "It calls for controlling the expected proportion of falsely rejected hypotheses",
     "the false-discovery-rate alternative discussed alongside the Bonferroni/Holm results"),
    ("cameron2008",
     "Standard asymptotic tests can over-reject, however, with few (five to thirty) clusters.",
     "justifies resampling engines as clusters rather than rows, and motivates the cluster bootstrap"),
    ("gelman2014",
     "In noisy, small-sample settings, statistically significant results can often be misleading.",
     "supports the design calculation (type S / type M) that replaces a bare power analysis"),
    ("ioannidis2005",
     "There is increasing concern that most current published research findings are false.",
     "precedent for reporting a non-replication as the result rather than hiding it"),
    ("reuel2024",
     "not all benchmarks are the same: their quality depends on their design and usability",
     "benchmark-hygiene precedent for the thesis that a benchmark has a resolution limit"),
    ("dietterich1998",
     "This article reviews five approximate statistical tests for determining whether one learning algorithm outperforms another on a particular learning task.",
     "paired-comparison practice on a single data set, and the type-I-error warning that motivates our paired design"),
    ("bouthillier2021",
     "variance due to data sampling, parameter initialization and hyperparameter choice impact markedly the results",
     "closest prior variance decomposition of benchmark scores; the comparison point for our engine-vs-method split"),
    ("guo2017",
     "the problem of predicting probability estimates representative of the true correctness likelihood",
     "calibration background for any predictive-interval claim made from MC dropout"),
    ("efron1994",
     "The bootstrap is a computer-based method of statistical inference that answers statistical questions without formulas",
     "the bootstrap itself; cited once, in the estimator description"),
    ("cohen1994",
     "After 4 decades of severe criticism, the ritual of null hypothesis significance testing",
     "cautionary source against reading a non-significant paired difference as evidence of no difference"),
    ("lakens2018justify",
     "The default use of an alpha level of 0.05 is suboptimal for two reasons.",
     "justifies fixing alpha and the equivalence margin before looking at the result"),
    ("zhang2019dl",
     "Extracting and modeling the engine symmetry characteristics is significant in improving remaining useful life (RUL) predictions for aircraft components",
     "representative recent deep-learning RUL paper on the same turbofan benchmark"),
    ("gal2016",
     "This mitigates the problem of representing uncertainty in deep learning without sacrificing either computational complexity or test accuracy.",
     "MC dropout cast as approximate Bayesian inference; the route to an interval without retraining"),
]

# abstract is empty in OpenAlex: may be cited, never quoted.
METADATA_ONLY = ["schuirmann1987", "button2013", "geirhos2020", "nadeau2003", "lei2018review"]


def _authors(display_names):
    out = []
    for n in display_names:
        parts = n.split()
        if len(parts) < 2:
            out.append(n)
            continue
        out.append(f"{parts[-1]}, " + "".join(p[0] + "." for p in parts[:-1]))
    return "; ".join(out)


def main() -> int:
    data = json.loads(CANDS.read_text(encoding="utf-8"))

    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    have = {r["key"] for r in rows}

    added, skipped = [], []
    for key, quote, role in ROWS:
        if key in have:
            skipped.append(key)
            print(f"[SKIP] {key} already present")
            continue
        hit = (data.get(key) or {}).get("hits", [None])[0]
        if not hit:
            raise SystemExit(f"NO CANDIDATE for {key}; run scripts/e1_expand_refs.py --key {key}")
        abs_ = hit.get("abstract") or ""
        if not abs_:
            raise SystemExit(f"NO ABSTRACT for {key}: cannot be added as a quotable row")
        if quote not in abs_:
            raise SystemExit(f"QUOTE NOT VERBATIM for {key}: {quote[:70]!r}")
        rows.append({
            "key": key,
            "title": (hit.get("title") or "").rstrip("."),
            "authors": _authors(hit.get("authors") or []),
            "year": hit.get("year"),
            "venue": hit.get("venue") or "",
            "doi_or_url": ("https://doi.org/" + hit["doi"]) if hit.get("doi") else "",
            "verified_on": VERIFIED_ON,
            "evidence_quote": quote,
            "role": role,
        })
        added.append(key)
        print(f"[ADD ] {key}")

    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)

    meta_only = {k: (data.get(k) or {}).get("hits", [None])[0] for k in METADATA_ONLY
                 if (data.get(k) or {}).get("hits")}
    (ROOT / "results" / "dsh" / "e1" / "refs_metadata_only.json").write_text(
        json.dumps({k: {"doi": v.get("doi"), "title": v.get("title"), "year": v.get("year"),
                        "venue": v.get("venue"), "authors": _authors(v.get("authors") or []),
                        "abstract_len": len(v.get("abstract") or ""),
                        "note": "no retrievable abstract; may be cited, never quoted"}
                    for k, v in meta_only.items()}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"added={len(added)} skipped={len(skipped)} total_rows={len(rows)}")
    print("metadata-only (cite, do not quote):", ", ".join(sorted(meta_only)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
