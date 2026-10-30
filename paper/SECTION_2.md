# 2 Background

Draft. Every citation below appears in `paper/refs_verified.csv` with
`quote_status = verbatim`; nothing with `metadata_only` is cited here.

## 2.1 The benchmark and its ground truth

C-MAPSS is a thermodynamic simulation of a turbofan engine in which degradation is
injected as a loss of flow and efficiency in selected modules, propagated until a
failure criterion is met [saxena2008]. The published challenge data (FD001–FD004)
consist of run-to-failure trajectories, each labelled with the number of cycles
remaining before that criterion. Two properties of the generator matter for
evaluation and are easy to forget. First, "failure" is not a physical event but the
moment a simulated health index reaches zero, where that index is defined as the
minimum of several superimposed operational margins [saxena2008]; the label is
therefore a property of the simulation contract, not of the asset. Second, the
degradation rate is drawn per engine, so the distribution of lifespans is a design
parameter of the dataset rather than an empirical fact.

These two properties are why the label is usually capped (commonly at 125 cycles):
early in life the remaining life is not identifiable from the sensors, and a linear
target would ask the model to extrapolate a quantity that the data do not contain.
The cap is a modelling convention, not part of the benchmark definition, and it
changes reported error substantially — a point we return to in §4.5.

## 2.2 The metric

The de-facto score is the asymmetric function of Saxena et al. [saxena2010], which
penalises late predictions (predicted life larger than the truth) more than early
ones, on the argument that a late prediction is the safety-relevant error. The same
paper states the problem this paper inherits: prognostic evaluation "lack[s] standard
definitions and suffer[s] from ambiguous and inconsistent interpretations"
[saxena2010]. Three conventions remain unsettled in current practice and are
routinely conflated in comparison tables:

1. **Aggregation.** The score can be summed over units or averaged; the two differ by
   the number of units and are not comparable across subsets of different size.
2. **The "late" indicator.** Whether a late prediction is defined by the sign of the
   error at a single cutoff, over a window, or beyond a tolerance, changes the
   reported fraction by a factor of several (§4.5).
3. **The unit of resampling.** Windows from the same engine are not independent;
   intervals computed over windows are therefore narrower than the evidence warrants.

## 2.3 The unit of evaluation

Most deep models on this benchmark are trained on sliding windows and evaluated on
windows [li2018], which is a reasonable choice for learning and a poor one for
inference about methods: a single engine contributes tens of windows, so a
window-level comparison is dominated by a handful of trajectories. The alternative —
one prediction per engine at a matched cutoff, with engines as the resampling unit —
is what we use throughout, and it is the reason our intervals are wider than those
usually reported. The difference is not cosmetic: with 20 held-out engines the paired
minimum detectable difference is around 2.4 RMSE (§4.3.3), which is larger than most
of the effects the literature reports as improvements.

## 2.4 What is already known to be wrong

Three lines of critique are relevant, and none of them is ours.

**Usage.** Ramasso and Saxena surveyed the first six years of C-MAPSS papers and
found that datasets, metrics and evaluation protocols were mixed freely, that only a
few of the surveyed works actually used the provided test sets, and that "any
meaningful comparison between approaches was not trivial" [ramasso2014]. Their remedy
is reporting discipline. Our question is different: even with perfect reporting, how
large a difference can this benchmark resolve?

**Leakage.** Kapoor and Narayanan document data leakage across 17 fields and 294
papers, and show that correcting it can erase the reported advantage of complex
models over simple baselines [kapoor2023]. We apply their taxonomy to our own
pipeline in §4.4, where a lifespan estimator turned out to be reading the true
lifespan at validation time; the episode is reported as a withdrawal, not as a
footnote.

**Variance.** In reinforcement learning and in benchmarking generally, seed and
split variance can exceed the differences being claimed, and significance-aware
reporting changes conclusions [henderson2018]; variance from data sampling,
initialisation and hyper-parameter choice "impact[s] markedly the results"
[bouthillier2021]. These results motivate the design in §3: engine identity is a
random factor, seeds are a random factor, and the quantity of interest is the
detectable effect at a given number of engines.

## 2.5 The gap this paper addresses

The three critiques above are about *how results are reported* (usage), *whether the
evaluation is valid* (leakage), and *whether a difference is stable across runs*
(variance). None of them answers the question a reader actually asks of a comparison
table: **given this benchmark, this split and this metric, what is the smallest
difference that could have been detected, and is the reported difference larger than
that?** That is a resolution question, it has a standard answer in experimental
design (the minimum detectable effect at a stated power [cohen1988]), and — as far as
our search in §E1 goes — it has not been computed for C-MAPSS. §3 turns it into a
protocol; §4 reports what the protocol says about our own experiments.
