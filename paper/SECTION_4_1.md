# 4.1 How much of a C-MAPSS result is the method?

This section answers one question before any method comparison is attempted:
when two reported RMSE values differ, how much of that difference can be
attributed to the method, and how much to the engines that happened to be held
out? Three independent decompositions are reported. All are point estimates:
none of the designs replicates runs at the cell level, so no interval is
attached to a share.

**Design A — 27 runs on three splits (7/8/9).**
The engine partition explains **22.8%** of the variance in RMSE, method identity
**1.9%**, the method-by-split interaction **47.8%**, and the seed **27.6%**
(`results/dsh/review_r1/variance_decomposition.json`, rebuilt by
`scripts/paper_variance_decomp.py`).

**Design B — 3 methods × 3 validation-set families, balanced.**
A second, independent design uses three methods crossed with three different
held-out engine families:

| method | validation set A | B | C | row mean |
|---|---:|---:|---:|---:|
| M0_raw_tf | 15.78 | 16.99 | 13.99 | 15.59 |
| M1_global | 17.19 | 12.93 | 15.57 | 15.23 |
| M6_tfA | 16.79 | 14.66 | 15.80 | 15.75 |
| **column mean** | 16.59 | 14.86 | 15.12 | 15.52 |

Two-way decomposition of the same nine cells: method **2.57%**, validation set
**31.52%**, interaction plus residual **65.91%**
(`results/dsh/e10/variance_two_designs.json`).

| source | design A | design B |
|---|---:|---:|
| method identity | 1.86% | **2.57%** |
| held-out engines / validation set | 22.81% | **31.52%** |
| interaction (+ seed in A) | 47.78% + 27.55% | **65.91%** |

The two designs differ in construction — A is unbalanced with seed replication,
B is a balanced 3 × 3 without replication — and they agree on the quantity that
matters: **method identity explains under 3% of the variance in both**, while the
choice of held-out engines explains one to two orders of magnitude more. In
design B the column means span 1.73 RMSE (14.86 to 16.59) while the row means
span 0.52 (15.23 to 15.75). Swapping the validation family moves the score more
than three times as much as swapping the method.

**A third view, at the level of a single contrast.**
Variance decomposition across methods answers "how much does the method matter
in general". A paired contrast answers a sharper question: for two *trained*
models compared on the *same* engines, what dominates the uncertainty of the
difference? Using the PE-versus-A contrast on FD002 (52 held-out engines, 3
seeds, `scripts/dsh_e9_variance_decomp.py`):

| component | estimate | share of variance |
|---|---:|---:|
| held-out engine sampling | SE **0.733** | **97.6%** |
| seed (initialisation) | SE **0.116** (3 seeds) | **2.4%** |
| seed at 10 seeds (extrapolated) | SE 0.063 | — |

This is the number that governs every comparison in the rest of the paper. It
says two things at once:

1. **Replication over seeds cannot rescue a comparison.** Going from 3 to 10
   seeds shrinks the seed term from 0.116 to 0.063 while the engine term stays at
   0.733; the total standard error barely moves. A study that reports "3 seeds,
   mean ± sd" is reporting the smaller component.
2. **The only lever is the number of independently held-out engines.** Standard
   error scales as 1/√n_engines, which is why section 4.2 converts engine count
   directly into a minimum detectable effect, and why section 4.3.3 re-runs the
   time contrast with every engine held out exactly once (5-fold
   cross-validation, `paper/E10_CV_PREREGISTRATION.md`).

The same asymmetry appears on FD001, where only 20 engines are held out per
split and the seed term is relatively larger (engine SE 0.894, seed SE 0.458,
seed share 20.8%). The engine term still dominates, but with 20 engines neither
component is small enough to resolve a sub-RMSE difference.

## What this section establishes, and what it does not

* **Supported.** In this protocol the method label is a minor source of variance
  (1.86% and 2.57% in two independent designs), and the held-out engine set is a
  major one (22.8% and 31.5%). For a single paired contrast, engine sampling
  accounts for 97.6% of the variance of the difference.
* **Supported.** Seed replication is the wrong lever: it addresses the 2.4%
  component, not the 97.6% one.
* **Not supported.** Any claim that methods are interchangeable in general. The
  decompositions cover the methods that were actually run under this protocol,
  on FD001/FD002 training engines. A method with a qualitatively different error
  profile could still separate cleanly.
* **Not supported.** That 2.6% is a universal constant. It is a property of this
  benchmark, this metric and this family of models; the paper reports it as a
  measured quantity for C-MAPSS, not as a law.
