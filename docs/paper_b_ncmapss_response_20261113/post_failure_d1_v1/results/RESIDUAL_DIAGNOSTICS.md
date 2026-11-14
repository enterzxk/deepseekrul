# Post-failure D1 — training-side residual diagnostic

Exploratory mechanism diagnostic after observing the closed common-route failure. No RUL/TEST labels were decoded; no new model was trained.

Execution: **COMPLETE**, 120/120 fixed crossfits; CPU elapsed 84.12 s.

|Subset|Crossfit engine-sensor rows|Fraction residual/raw std <0.1|Median residual/raw std|Median dynamic error reduction|
|---|---:|---:|---:|---:|
|DS02|420|0.928571|0.013833|0.074315|
|DS05|420|0.928571|0.013638|0.103694|
|DS06|420|0.938095|0.013288|0.098626|
|DS07|420|0.964286|0.014839|0.092489|

Fixed scale-suppression gate: **MET**.
Fixed dynamic-omission gate: **NOT_MET** (['DS05']).

Empirical residual scale is a fitted descriptive scale, not a noise estimate. Dividing by a fixed positive scale cannot change a within-engine/sensor Pearson age correlation; this equality was checked.

The dynamic predictor includes the original quadratic W terms, an intercept, position, and differences; its total gain cannot isolate difference terms from recalibration or position effects. Error reduction is versus a zero residual predictor and is not evidence of health or RUL prediction.

All120 crossfits overlap in engines/training sets. They are not120 independent samples; no p-value, independence-based CI, or generalization claim is supplied. No feature, ridge, threshold, outer evaluation, or frozen model was selected using this diagnostic.
