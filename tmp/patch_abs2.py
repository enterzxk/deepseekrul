from pathlib import Path
p=Path('PAPER_DRAFT.md'); s=p.read_text()
abstract = """## Abstract

Remaining useful life (RUL) predictors for aero-engines are compared by pooled regression
error, a convention that hides whether the observed sensor history identifies the individual
engine's position on its own degradation trajectory. We quantify that question on the
C-MAPSS benchmark with a frozen protocol: five engine-disjoint outer folds, a full training
budget with no epoch selection, a 42.3k-parameter backbone, and four metrics with
engine-clustered intervals. The lifetime scale is essentially unobservable early in life:
its rank correlation with the true lifetime is 0.26 at 50 observed cycles and rises to 0.92
at 150, while most training windows lie in the uninformative region. An additive lifetime
prior improves pooled RMSE by 3.3% on FD002 (95% CI 0.07-0.82 cycles) and 6.5% on FD003,
but not significantly on FD001 or FD004; on FD004 the same prior worsens the asymmetric
score by 25%. Using the identical estimate to re-base the cycle coordinate costs 7.5% and to
rescale the target costs 28%. No observable we tested predicts which engines benefit
(|rho| < 0.08), although a per-window oracle over the variants would gain 15%, and every
gate we built is worse than a fixed fusion. Under 20% missing samples the additive prior's
advantage grows to 6.2%; under a one-standard-deviation offset on three channels the pooled
error is unchanged but the asymmetric score doubles. We conclude that individualisation on
this benchmark is observability-limited rather than model-limited, and that pooled RMSE
alone cannot support claims about it.

"""
anchor = "## I. INTRODUCTION"
if anchor in s and '## Abstract' not in s:
    s = s.replace(anchor, abstract + anchor, 1); p.write_text(s); print('patched')
else:
    print('skip')
