from pathlib import Path
p=Path('PAPER_DRAFT.md'); s=p.read_text()
start=s.find('## Abstract')
end=s.find('## I. INTRODUCTION')
assert start!=-1 and end!=-1
new = '''## Abstract

Remaining useful life (RUL) predictors for aero-engines are compared by pooled regression error,
a convention that hides two measurement questions: how long the observation should be, and whether
the observed history identifies the individual engine's position on its own degradation trajectory.
We answer both on C-MAPSS with a frozen protocol: five engine-disjoint outer folds, a full training
budget with no epoch selection, a 42.3k-parameter backbone, and four metrics with engine-clustered
intervals.

First, the observation span is a first-order design variable that the 50-cycle convention leaves
untouched. On the windows where a longer history exists, a 100-cycle window lowers the pooled error
from 11.03 to 7.50 cycles. Because a long window is only available once the engine has been observed
long enough, we schedule the span to the available history (50 cycles below 100 observed cycles,
100 cycles above). Relative to the locked reference this reduces the pooled RMSE by 6.64% on FD002
(95% CI 0.65-1.09 cycles) and 4.10% on FD004, improves all four metrics on both fleets, and improves
the individual engine on 73-83% of FD002 rather than on a tail of engines; the stratified interval is
5.37% with a lower bound of 0.53 cycles. Three controls attribute the gain to the span: a late-window
specialist at the same span is worth 3.5%, an equal-parameter span control 12.8%, and an equal-compute
three-seed ensemble leaves the gain intact.

Second, the lifetime prior that individualisation methods inject is only partially observable: its
rank correlation with the true lifetime is 0.26 at 50 observed cycles and 0.92 at 150, and no
observable predicts which engines benefit (|rho| < 0.08). Additive injection improves the pooled RMSE
by 3.3% on FD002 but not significantly on FD001 or FD004, re-basing the cycle coordinate costs 7.5%,
and rescaling the target costs 28%.

We conclude that RUL accuracy on this benchmark is limited first by how long the measurement observes
and only second by how the lifetime prior is used, and that pooled RMSE alone cannot support either
claim.

'''
s = s[:start] + new + s[end:]
p.write_text(s); print('patched')