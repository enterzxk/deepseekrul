# 摘要修订草案（待 FD004 验证后应用）

## 新版摘要（英文，草稿）

Remaining useful life (RUL) predictors for aero-engines are compared by pooled regression
error, a convention that hides two measurement questions: how long the observation should
be, and whether the observed history identifies the individual engine's position on its own
degradation trajectory. We answer both on C-MAPSS with a frozen protocol: five
engine-disjoint outer folds, a full training budget with no epoch selection, a
42.3k-parameter backbone, and four metrics with engine-clustered intervals.

First, the observation span is a first-order design variable. The 50-cycle window used by
convention leaves accuracy on the table: on the windows where a longer history is
available, a 100-cycle window reduces the pooled error from 11.03 to 7.50 cycles, and a
150-cycle window to 4.97. Matching the span to the available history (50 cycles below 100
observed cycles, 100 cycles above) lowers the pooled RMSE by X% on FD002 and Y% on FD004
relative to the locked reference, and improves the individual engine on 79% of the fleet
rather than on a tail of engines. Specialisation alone (same span, trained only on late
windows) is worth 3.5%, an equal-parameter span control 12.8%, and an equal-compute
three-seed ensemble leaves the gain intact, so the effect is the span itself.

Second, the lifetime prior that individualisation methods inject is only partially
observable: its rank correlation with the true lifetime is 0.26 at 50 observed cycles and
0.92 at 150, and no observable predicts which engines benefit (|rho| < 0.08). Additive
injection improves the pooled RMSE by 3.3% on FD002 but not on FD001 or FD004, re-basing
the cycle coordinate costs 7.5%, and rescaling the target costs 28%.

We conclude that RUL accuracy on this benchmark is limited first by how long the
measurement observes and only second by how the lifetime prior is used, and that pooled
RMSE alone cannot support either claim.

## 待填数字

- X% = FD002 的 FULL 调度相对参考（预计 −6 到 −7%）
- Y% = FD004 的 FULL 调度相对参考（待测）
