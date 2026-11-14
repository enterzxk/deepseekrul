# DEVELOPMENT_LOEO — full BayesRUL-HNN migration

Status: **PASS**; full 192-training gate: **True**.

Validated: 120/120 inner and 72/72 outer; 24/24 nested selections.

No training, model forward, or official TEST was executed by this collector.

|Subset|View|Target|Status|Engine-equal RMSE mean ± seed SD|
|---|---|---|---|---|
|DS02|clean|raw|COMPLETE|14.796958 ± 3.361847|
|DS02|clean|clip125|COMPLETE|14.796958 ± 3.361847|
|DS02|density|raw|COMPLETE|16.453093 ± 3.974208|
|DS02|density|clip125|COMPLETE|16.453093 ± 3.974208|
|DS02|unpaired|raw|COMPLETE|23.257620 ± 1.146985|
|DS02|unpaired|clip125|COMPLETE|23.257620 ± 1.146985|
|DS05|clean|raw|COMPLETE|14.740798 ± 0.751353|
|DS05|clean|clip125|COMPLETE|14.740798 ± 0.751353|
|DS05|density|raw|COMPLETE|17.604227 ± 0.749212|
|DS05|density|clip125|COMPLETE|17.604227 ± 0.749212|
|DS05|unpaired|raw|COMPLETE|30.436809 ± 4.902764|
|DS05|unpaired|clip125|COMPLETE|30.436809 ± 4.902764|
|DS06|clean|raw|COMPLETE|18.500019 ± 6.998012|
|DS06|clean|clip125|COMPLETE|18.500019 ± 6.998012|
|DS06|density|raw|COMPLETE|19.402304 ± 7.134218|
|DS06|density|clip125|COMPLETE|19.402304 ± 7.134218|
|DS06|unpaired|raw|COMPLETE|27.631520 ± 6.105695|
|DS06|unpaired|clip125|COMPLETE|27.631520 ± 6.105695|
|DS07|clean|raw|COMPLETE|10.778149 ± 3.410131|
|DS07|clean|clip125|COMPLETE|10.778149 ± 3.410131|
|DS07|density|raw|COMPLETE|12.227165 ± 4.085288|
|DS07|density|clip125|COMPLETE|12.227165 ± 4.085288|
|DS07|unpaired|raw|COMPLETE|22.639433 ± 8.672805|
|DS07|unpaired|clip125|COMPLETE|22.639433 ± 8.672805|

- Development engines only; no official TEST or paper-native protocol reproduction.
- Inner HPO scores and 72 outer evaluations are never pooled.
- Seed first averages six heldout-engine RMSE values; SD is across three seed means.
- Six folds share training engines; seeds are not independent engines; no significance claim.
- Training seconds exclude inference/IO; incomplete snapshots are not final performance claims.
- Collector checks CPU model construction/state, not forward; replay is a separate explicit operation.

Errors: 0; see COLLECTION.json.
