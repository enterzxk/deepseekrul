#!/usr/bin/env python
"""E1 expansion — resolve candidate references by TITLE search, never by memory.

Why: the project rule is "no unverified citation" and a wrong DOI is worse than a
missing reference. This script only searches OpenAlex by (approximate) title and
records whatever OpenAlex returns, including whether an abstract exists. Nothing
is written to paper/refs_verified.csv here; a human/agent picks from the dump.

Usage:
    python scripts/e1_expand_refs.py            # search all queries
    python scripts/e1_expand_refs.py --key X    # one query, verbose abstract
Output: results/dsh/e1/refs_candidates.json
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dsh" / "e1"
OUT.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "e1-verify/1.0 (mailto:dev@example.org)"}

# key -> (search string, why the paper might need it)
QUERIES = {
    "saxena2008cmapss": ("Damage propagation modeling for aircraft engine run-to-failure simulation",
                         "origin of the C-MAPSS simulator the whole study is about"),
    "saxena2010metrics": ("Metrics for offline evaluation of prognostics",
                          "origin of the asymmetric scoring function used as the NASA score"),
    "heimes2008": ("Recurrent neural networks for remaining useful life estimation",
                   "the first RNN line on C-MAPSS; historical anchor"),
    "lei2018review": ("Machinery health prognostics: A systematic review from data acquisition to RUL prediction",
                      "survey framing for the RUL field"),
    "zhang2019dl": ("Deep learning based remaining useful life prediction for turbofan engine",
                    "representative deep-learning RUL method line"),
    "schuirmann1987": ("A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability",
                       "source of the TOST procedure used for the equivalence claims"),
    "lakens2017": ("Equivalence tests: a practical primer for t-tests, correlations, and meta-analyses",
                   "modern equivalence-testing practice and reporting"),
    "holm1979": ("A simple sequentially rejective multiple test procedure",
                 "the multiplicity correction named in the statistics audit"),
    "benjamini1995": ("Controlling the false discovery rate: a practical and powerful approach to multiple testing",
                      "FDR alternative to Bonferroni, needed for the multiplicity discussion"),
    "cameron2008": ("Bootstrap-based improvements for inference with clustered errors",
                    "cluster-robust bootstrap that justifies resampling engines, not rows"),
    "gelman2014": ("Beyond power calculations: assessing type S (sign) and type M (magnitude) errors",
                   "why an underpowered comparison exaggerates the winning effect"),
    "button2013": ("Power failure: why small sample size undermines the reliability of neuroscience",
                   "field-level precedent for low-power fields producing inflated effects"),
    "ioannidis2005": ("Why most published research findings are false",
                      "precedent for reporting non-replication as a result"),
    "geirhos2020": ("Shortcut learning in deep neural networks",
                    "the shortcut-learning frame used for the time-channel argument"),
    "reuel2024": ("BetterBench: assessing AI benchmarks, uncovering issues, and establishing best practices",
                  "benchmark-hygiene precedent for the 'what a benchmark can measure' thesis"),
    "demsar2006": ("Statistical comparisons of classifiers over multiple data sets",
                   "standard practice for comparing learners across splits/benchmarks"),
    "dietterich1998": ("Approximate statistical tests for comparing supervised classification learning algorithms",
                       "paired-comparison practice on a single data set"),
    "bouthillier2021": ("Accounting for variance in machine learning benchmarks",
                        "variance decomposition of benchmark scores; closest analogue to our §4.1"),
    "nadeau2003": ("Inference for the generalization error",
                   "why resampled estimates of a paired difference are correlated"),
    "guo2017": ("On calibration of modern neural networks",
                "calibration background if uncertainty intervals are reported"),
    "lakens2018justify": ("Justify your alpha: a primer on two-sided tests and one-sided alternatives",
                          "pre-registration / alpha justification practice"),
    "simmons2011": ("False-positive psychology: undisclosed flexibility in data collection and analysis",
                    "researcher-degrees-of-freedom precedent behind the pre-registration contract"),
    "efron1994": ("An introduction to the bootstrap",
                  "the bootstrap itself"),
    "cohen1994": ("The earth is round (p < .05)",
                  "cautionary source on null-hypothesis significance testing"),
    "gal2016": ("Dropout as a Bayesian approximation: representing model uncertainty in deep learning",
                "MC-dropout as a cheap uncertainty estimate; the basis of any interval we report without retraining"),
}


def reconstruct(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def search(q: str, per_page: int = 3):
    url = ("https://api.openalex.org/works?per-page=%d&search=%s"
           % (per_page, urllib.parse.quote(q)))
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=None)
    a = ap.parse_args()

    dump = {}
    keys = [a.key] if a.key else list(QUERIES)
    for key in keys:
        q, why = QUERIES[key]
        try:
            d = search(q)
        except Exception as e:
            dump[key] = {"query": q, "why": why, "error": repr(e)}
            print(f"[ERR ] {key}: {e!r}")
            continue
        hits = []
        for w in d.get("results", []):
            hits.append({
                "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                "title": w.get("title"),
                "year": w.get("publication_year"),
                "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                "authors": [x["author"]["display_name"] for x in w.get("authorships", [])][:6],
                "abstract_len": len(reconstruct(w.get("abstract_inverted_index"))),
                "abstract": reconstruct(w.get("abstract_inverted_index")),
                "cited_by": w.get("cited_by_count"),
            })
        dump[key] = {"query": q, "why": why, "hits": hits}
        top = hits[0] if hits else {}
        print(f"[OK  ] {key:22s} {str(top.get('year')):>5s} abs={top.get('abstract_len'):>4d}  "
              f"{(top.get('title') or '')[:70]}")
        if a.key:
            print(json.dumps(hits, indent=2, ensure_ascii=False)[:4000])

    path = OUT / "refs_candidates.json"
    if a.key:  # merge instead of overwrite when probing a single key
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
        prev.update(dump)
        dump = prev
    path.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
