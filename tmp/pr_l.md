
## L. What does not work: nine model-side interventions

The span schedule changes how the measurement is taken. A natural question is whether
model-side changes could deliver more. We tested nine, all with the same frozen recipe,
the same folds, and the same four metrics. Table L1 reports the single-fold screen
(FD002 fold 2 and FD004 fold 2) against the 42k baseline on the same fold.

| Intervention | Params | FD002 M0 | FD002 FULL | FD004 M0 | FD004 FULL |
|---|---:|---:|---:|---:|---:|
| Multi-scale conv + extra layer + attention pooling | +74k to +677k | +7.4% to +42% | +24.9% | - | - |
| A: trajectory monotonicity loss | 0 | +4.3% | not run | - | - |
| B: metric-matched pinball loss | 0 | -3.9% | +0.1% | +5.3% | +5.8% |
| C: lifetime reparameterisation | 0 | +13.8% | +9.9% | - | - |
| D: engine-level mixup (all channels) | 0 | **-9.1%** | +0.8% | - | - |
| DX: mixup on inputs only | 0 | +3.5% | +4.5% | - | - |
| E: cross-sensor graph attention | ~200 | see protocol 9.7 | | | |

Only one cell is clearly positive, and it is on the arm that does not carry the prior.
Three structural conclusions follow.

**The prior arm is saturated.** Every intervention that helps the no-prior arm (B at -3.9%,
D at -9.1%) returns to zero or reverses on the prior arm. The zero-initialised linear
injection of the lifetime statistics is itself a strong prior regulariser: it supplies
information the model would otherwise have to estimate from data, so additional variance
reduction has no marginal value.

**Reducing degrees of freedom works on the no-prior arm; adding expressiveness never works.**
Mixup, which constrains the model through interpolated samples, is the only large gain.
A hard monotonicity constraint and an output reparameterisation both fail, and every
capacity increase fails. This is consistent with an effective sample size of 208 engines
rather than 33k windows: the no-prior arm suffers from variance, and the prior arm does not.

**Mixup only works when every input channel is mixed consistently.** Mixing the window and
the label but keeping the original prior turns -9.1% into +3.5% on the no-prior arm and
+0.8% into +4.5% on the prior arm. A mixed window with an unmixed prior is a self-contradictory
training sample. We did not predict this; it is a reproducible observation.

The negative results are half of the paper's argument. Reporting only the -5.55% would leave
open the explanation that the right model-side change had simply not been found. Nine
interventions, four of them with zero additional parameters, close that explanation: on this
benchmark the return is in how the measurement is taken, not in what is added to the model.
