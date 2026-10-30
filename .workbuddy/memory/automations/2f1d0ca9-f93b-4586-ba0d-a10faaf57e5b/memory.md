# Automation memory — overnight hop driver (2f1d0ca9)

One line per hop. Purpose: let the next hop know the queue position and whether
the guard state has changed, without re-reading the whole paper tree.

| local | hop | task | verdict | note |
|---|---|---|---|---|
| 2026-10-30T00:40 | hop-0040 | T6 | PASS | E1 closed at 12 verified refs (6 added: zheng2017, ress2019, li2014oc, cohen1992, dodge2019, raff2019). Head of queue is now **T7** (draft §2 from verified refs only). STATUS.json sha changed to `ab6d2b51`, so the two-hop circuit breaker did not trip. Lock released. |

## Standing notes for the next hop

* Do not re-add refs already in `paper/refs_verified.csv`; the write step skips them.
* Withheld refs (`li2018`, `lei2018`, `freitas2026`, engrXiv `10.31224/8145`) have no
  openly retrievable abstract — do not promote them without reading a licensed PDF.
* Existing rows store a bare DOI; the script normalises `https://doi.org/` away.
