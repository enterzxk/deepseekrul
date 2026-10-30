# E1 — Literature position (working note, 2026-10-29)

Purpose: give every citation in the paper a verified home, and state the
difference from the three nearest works **before** any draft text exists.
Nothing below may be cited until it appears in `paper/refs_verified.csv`.

## 1. Search log (novelty record for G4)

| # | query | engine | date | outcome |
|---|---|---|---|---|
| 1 | C-MAPSS benchmark evaluation problems remaining useful life critical review | WebSearch | 2026-10-29 | found Ramasso & Saxena 2014 (verified), plus one low-quality venue (rejected, see §4) |
| 2 | remaining useful life prediction reproducibility benchmark evaluation pitfalls | WebSearch | 2026-10-29 | found a 2026 conference paper on FD001 "statistical compression" (candidate, metadata unverified) |
| 3 | Saxena "Metrics for offline evaluation of prognostic performance" DOI | WebSearch | 2026-10-29 | DOI confirmed 10.36001/ijphm.2010.v1i1.1336 |
| 4 | "Leakage and the reproducibility crisis..." Kapoor Narayanan | WebSearch | 2026-10-29 | DOI + abstract quote confirmed |
| 5 | "Damage propagation modeling for aircraft engine run-to-failure simulation" DOI | WebSearch | 2026-10-29 | DOI 10.1109/PHM.2008.4711414 confirmed |
| 6 | arXiv 1709.06560 / 2103.03098 | WebFetch | 2026-10-29 | title, authors, venue, abstract quote confirmed |
| 7 | Crossref `query.bibliographic` = "Long Short-Term Memory Network for RUL estimation Zheng" | Crossref API | 2026-10-30 | `10.1109/icphm.2017.7998311` resolved; used |
| 8 | Crossref `query.bibliographic` = "RUL estimation … deep convolution neural networks" | Crossref API | 2026-10-30 | `10.1016/j.ress.2017.11.021` resolved but abstract withheld → rejected |
| 9 | Crossref `query.bibliographic` = "operating condition normalization C-MAPSS clustering" | Crossref API | 2026-10-30 | `10.1109/icphm.2014.7036396` (peer-reviewed) used; `10.31224/8145` is a preprint → rejected; also surfaced the 2026 FD001 paper's DOI, still unquotable |
| 10 | Crossref `query.bibliographic` = "A power primer Cohen 1992" | Crossref API | 2026-10-30 | `10.1037/0033-2909.112.1.155` resolved; used |
| 11 | OpenAlex `search` = "C-MAPSS remaining useful life prediction", `has_abstract:true`, cited-by desc | OpenAlex API | 2026-10-30 | `10.1016/j.ress.2018.11.027` (525 citations) used as second baseline |
| 12 | Semantic Scholar / OpenAlex abstract fetch for `li2018`, `lei2018`, `freitas2026` | S2 + OpenAlex | 2026-10-30 | all three returned no abstract → withheld from the table |

## 2. Verified references (12)

See `paper/refs_verified.csv`. Each row carries a DOI/URL, an access date and a
verbatim sentence from the abstract that the paper is allowed to lean on. From
2026-10-30 the quote is asserted as an exact substring of the fetched abstract by
`scripts/e1_verify_refs.py` (a paraphrase cannot be written into the table).

Benchmark and metric provenance:

* `saxena2008` — the generator and the health-index→failure convention.
* `saxena2010` — the asymmetric score and the lack of standard definitions.
* `ramasso2014` — C-MAPSS benchmarking is inconsistent across papers.

Method line (described, not endorsed):

* `zheng2017` — LSTM baseline; the sequence-model line §2 must characterise without
  adopting its reported gains.
* `ress2019` — semi-supervised deep architecture on C-MAPSS; second baseline.
* `li2014oc` — operating-condition-classified prognosis; precedent for treating
  operating condition as an explicit modelling variable.

Statistics and reproducibility machinery:

* `cohen1992` — required-n/power relation behind the MDE in §3.
* `dodge2019` — test-set scores alone cannot rank models; the C-MAPSS analogue is
  our engine-level MDE.
* `raff2019` — reproducibility measured, not asserted (255 reimplementations).
* `kapoor2023` — leakage taxonomy; 294 papers across 17 fields.
* `henderson2018` — significance reporting / seed variance.
* `bouthillier2021` — variance components in benchmark comparisons.

## 3. Difference from the three nearest works

**(a) Ramasso & Saxena (2014).** They survey 70 C-MAPSS papers and show that
datasets, metrics and test usage are inconsistent; the remedy they propose is
reporting discipline (which subsets, which metrics, top-30 challenge scores).
*Our difference:* we do not audit what authors report, we measure what the
benchmark **can resolve** — an engine-level minimum detectable effect, a variance
decomposition with engine identity as a random factor, and two mechanism probes
(time shortcut, lifespan identifiability). Their unit of analysis is the paper;
ours is the engine.

**(b) Kapoor & Narayanan (2023).** They give an eight-type leakage taxonomy and
show that correcting leakage erases claimed ML advantages in other fields.
*Our difference:* we apply the taxonomy to **our own** pipeline and report the
result as a withdrawal (the `c/T̂` arm: constant fallback in training plus a
true-lifespan leak in validation, §J of `REVIEW_20261022_R3.md`). The paper's
contribution is not "leakage exists" but "here is how much a single leak was
worth in RMSE (2.78, CI [−4.72,−0.87]) in a published-style RUL comparison".

**(c) Seed/benchmark variance line (Henderson 2018; Bouthillier 2021).** They
establish that seed and split variance can swamp method differences.
*Our difference:* we quantify the split at the level that matters for C-MAPSS —
**which engines** are held out (31.5% of variance) versus which method is used
(2.6%) — and convert it into a concrete MDE and a stopping rule.

No claim of priority ("first") is made anywhere; the search log above is the
only novelty evidence we have, and it is thin by design at this stage.

## 4. Rejected / not citable

| item | reason |
|---|---|
| "Leakage-Resistant and Uncertainty-Aware RUL Prediction..." (sjphrt.com.ly, 2026) | venue not recognisable as a peer-reviewed PHM outlet; not used, even though the topic overlaps |
| blog posts on RUL pitfalls (iotdigitaltwinplm.com) | non-peer-reviewed; may inform intuition, never a citation |
| MDPI battery RUL review | different asset class; only usable as a general leakage remark, not for C-MAPSS claims |
| Li, Ding & Sun 2018, *DCNN for RUL*, RESS, `10.1016/j.ress.2017.11.021` | metadata resolved, but no abstract is exposed by Crossref, OpenAlex or Semantic Scholar; criterion 1c (verbatim quote) unmet, so the row is withheld. Use once a licensed PDF is read |
| Lei et al., *Machinery health prognostics* review, MSSP, `10.1016/j.ymssp.2017.11.016` | same failure mode: DOI valid, abstract not openly retrievable |
| Freitas & Lopes 2026, *Statistical Analysis of Transformer RUL Models on C-MAPSS FD001*, SciTePress, `10.5220/0015076000004091` | the parked 2026 item from §5.2: DOI, authors and venue now confirmed, but no abstract text is retrievable, so no quotable sentence exists. Keep parked, do not cite |
| Gupta 2026, engrXiv `10.31224/8145` (OC-normalisation audit) | preprint; `has_abstract: true` and tempting to cite, but §5.2 asked for a peer-reviewed OC source, which `li2014oc` now supplies |
| DOI `10.48550/arXiv.2201.08619` | resolving it returned an unrelated backdoor-attack paper, i.e. the DOI was wrong; discarded before use. Recorded here because it shows the fetch step catching a plausible-looking identifier |

## 5. Status — E1 closed (2026-10-30)

Target was 12–15 verified entries; `paper/refs_verified.csv` holds **12**, every
row with a DOI/URL and a verbatim abstract sentence (count produced by
`scripts/e1_verify_refs.py write`, which exits non-zero if the total leaves
[12,15]). Coverage required by `paper/E1_PREREGISTRATION.md` §3 is met: two
method-side C-MAPSS baselines (`zheng2017`, `ress2019`), one peer-reviewed
operating-condition source (`li2014oc`), one power/MDE source (`cohen1992`).

Deliberately at the lower bound: four further strong candidates
(`li2018`, `lei2018`, `freitas2026`, Gupta preprint) were withheld because their
abstracts are not openly retrievable, and widening the count with unquotable rows
would break the evidence standard rather than strengthen §2.

**Verdict: PASS** (12/12 rows meet 1a–1d; coverage satisfied; no priority claim made).
