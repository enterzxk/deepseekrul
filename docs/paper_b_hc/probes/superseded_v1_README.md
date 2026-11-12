# superseded_v1 — Stage D acceptance records from an earlier code version

These JSON records were produced by an intermediate `hi_d.py` that already had
the `b0` bias fix but **not** the two later numerical revisions:

  * the `k` basis was fed **raw**, `k in [1, K]` (not `k/K, log1p(k)/log1p(K)`);
  * the rate head's output layer kept its **random** init (not zero-init).

Code identity of the run that produced them (md5 taken on a40 at push time):

```
paper_b/hc/hi_d.py                      8f24b7b3733da82892b9d14a43ad80df
docs/paper_b_hc/probes/d_acceptance.py  5e899e12909613b4b035bc268a790775
```

What they are cited for (`STAGE_D_ACCEPTANCE.md` section 3.1, "version A"):

  the rate head output was `7.29e-8` instead of `4.8e-3`; the arrival was pinned
  at `K` for 82.5-95.6% of uncensored samples; and the backward produced
  **exactly zero** gradient on FD001/FD002/FD004 — identically at `K=200` and
  `K=600`, which is the evidence that `K` was not the cause of that zero.

Do **not** mix these with the current records. The current run has a different
head init (normalised `k` + zero-init output layer) and therefore different
numbers, including non-zero gradients at `K=200`.
