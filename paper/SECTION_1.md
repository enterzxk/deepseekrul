# 1 Introduction

Draft. Author block and funding are placeholders pending the human input recorded in
`paper/STATUS.json`.

## 1.1 The problem with a comparison table

A C-MAPSS paper typically ends with a table in which one method is a few tenths of an
RMSE better than another, and a sentence that calls the difference an improvement.
The sentence is only meaningful if the benchmark can resolve a difference of that
size. Nobody reports whether it can. The question is not rhetorical: with 20 held-out
engines, the smallest difference our own pipeline can detect is about 2.4 RMSE, and
with 52 it is about 1.9 — both larger than the differences that motivate most published
architectural changes.

This paper takes that question seriously and follows it wherever it goes, including
into our own results. Three things follow from measuring resolution first.

**Most method differences are not decidable at this scale.** Decomposing RMSE variance
across methods, engine sets and seeds attributes a few percent to the method and
roughly a third to which engines were held out. Converting that into a minimum
detectable effect gives ≈1.5 RMSE at 260 engines — an order of magnitude larger than
the 5% target that our own project brief specified, and larger than most reported
gains.

**What does move the metric is not architecture.** Removing the cycle index from the
input costs 6.9 RMSE in a linear probe over 360 engines and 2.7 in the frozen neural
pipeline; the frozen-pipeline contrast replicates on three of the four subsets (pooled
+0.98 [+0.35, +1.61] over 709 engines), reverses sign on the fourth, and transfers to
the official test sets at half the magnitude (pooled +0.48 [+0.04, +0.93] over 653
engines). A model's accuracy on this benchmark is, to first approximation, the accuracy
of its time representation — and a per-subset estimate of even that effect is not a
measurement: the same evaluation protocol that shows the effect on three subsets
returns the opposite sign on FD003 in development and a significant positive effect on
FD003 at test.

**The remaining headroom is real but not reachable by a better time feature.** If the
true lifespan is known and used to normalise time, RMSE falls by 8.2. A deployable
prefix estimator recovers none of it: its estimate is 1.9× too large and correlates
0.20 with the truth, and forcing the health index to be monotone does not fix it. The
bottleneck is identifiability, not encoding.

## 1.2 Contributions

1. **A resolution-first protocol for C-MAPSS evaluation.** Engine-level splits, one
   matched prediction per engine, engine-clustered intervals with a reported
   Monte-Carlo error, one sign convention, and a minimum-detectable-effect calculation
   with an explicit power convention. The protocol is cheap — no retraining — and it
   changes verdicts: two of our own comparisons pass or fail depending on whether the
   bootstrap has converged.
2. **A quantitative account of what the benchmark actually measures.** Time is the
   dominant signal and is partially a shortcut; lifespan scale is the dominant
   *unexploited* signal; architecture is a distant third. Each claim carries an
   interval and a stated boundary, including the subset on which it fails to replicate.
3. **A reproducibility record that includes the corrections.** A from-scratch rerun
   reproduces the reported metrics exactly (ΔRMSE = 0.00000); three claims were
   withdrawn during the work and are reported as withdrawals with dates, artefacts and
   the rule that killed them. We treat those withdrawals as evidence for the thesis
   rather than as embarrassments to be edited out.

## 1.3 What this paper does not claim

No state of the art, no new architecture, no claim of being first. FD003 and FD004 were
initially sealed; they were unsealed (train files for the five-fold designs, and — only
after every model was frozen — the official test files, with sha256 recorded and
mtimes untouched) so that the resolution analysis now covers **all four subsets**, on
both the development pool (709 engines) and the official test sets (653 engines).
Even so, the four subsets are four operating regimes of one simulator; the claims are
about what this benchmark family can resolve, not about turbofan engines in general.
The bounded time encoding we test is *not* shown to be better than the linear one —
its advantage never reaches significance on any subset or protocol, and it does not
transfer to the official test sets — and we report it as a promising but unresolved
design change. Where a number in an earlier internal report turned out to have no
artefact behind it, the claim is withdrawn rather than restated.

## 1.4 Roadmap

§2 reviews what is already known about C-MAPSS usage, leakage and variance, and
identifies the missing question. §3 fixes the design: units, metrics, statistics,
pre-registration and provenance. §4 reports results in four parts — variance and
resolution (§4.1–4.2), the time contrast, its cross-subset replication and its
official-test generalisation (§4.3), the lifespan ceiling and the estimator that fails
to reach it (§4.4), and the metric and protocol traps that change conclusions (§4.5).
§5 turns the protocol into a checklist and a calculator. §6 states the limitations,
including the ones the protocol cannot fix. §7 documents reproduction.
