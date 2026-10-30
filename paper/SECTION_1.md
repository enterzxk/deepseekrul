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
pipeline; the same contrast is directionally identical on a second subset but below
that subset's detection threshold. A model's accuracy on this benchmark is, to first
approximation, the accuracy of its time representation.

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

No state of the art, no new architecture, no claim of being first. FD003 and FD004 are
sealed and are not used; the two subsets we do use cannot support a claim about
"C-MAPSS" as a family, only about those subsets. The bounded time encoding we test is
*not* shown to be better than the linear one — the pre-registered margin is not met on
either subset — and we report it as a promising but unresolved design change. Where a
number in an earlier internal report turned out to have no artefact behind it, the
claim is withdrawn rather than restated.

## 1.4 Roadmap

§2 reviews what is already known about C-MAPSS usage, leakage and variance, and
identifies the missing question. §3 fixes the design: units, metrics, statistics,
pre-registration and provenance. §4 reports results in four parts — variance and
resolution (§4.1–4.2), the time contrast and its cross-subset replication (§4.3), the
lifespan ceiling and the estimator that fails to reach it (§4.4), and the metric and
protocol traps that change conclusions (§4.5). §5 turns the protocol into a checklist
and a calculator. §6 states the limitations, including the ones the protocol cannot
fix. §7 documents reproduction.
