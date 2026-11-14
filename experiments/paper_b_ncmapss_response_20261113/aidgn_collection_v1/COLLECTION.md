# DEVELOPMENT_LOEO — full AIDGN migration

Status: **PASS**; full 192-training gate: **True**.

Validated: 120/120 inner and 72/72 outer; 24/24 nested selections.

No training, model forward, or official TEST was executed by this collector.

|Subset|View|Target|Status|Engine-equal RMSE mean ± seed SD|
|---|---|---|---|---|
|DS02|clean|raw|COMPLETE|9.027269 ± 0.088845|
|DS02|clean|clip125|COMPLETE|9.027269 ± 0.088845|
|DS02|density|raw|COMPLETE|11.005732 ± 0.675222|
|DS02|density|clip125|COMPLETE|11.005732 ± 0.675222|
|DS02|unpaired|raw|COMPLETE|16.001179 ± 1.829456|
|DS02|unpaired|clip125|COMPLETE|16.001179 ± 1.829456|
|DS05|clean|raw|COMPLETE|12.706486 ± 0.448734|
|DS05|clean|clip125|COMPLETE|12.706486 ± 0.448734|
|DS05|density|raw|COMPLETE|15.299465 ± 0.715237|
|DS05|density|clip125|COMPLETE|15.299465 ± 0.715237|
|DS05|unpaired|raw|COMPLETE|14.079935 ± 1.833255|
|DS05|unpaired|clip125|COMPLETE|14.079935 ± 1.833255|
|DS06|clean|raw|COMPLETE|10.625214 ± 0.349056|
|DS06|clean|clip125|COMPLETE|10.625214 ± 0.349056|
|DS06|density|raw|COMPLETE|12.026289 ± 1.735476|
|DS06|density|clip125|COMPLETE|12.026289 ± 1.735476|
|DS06|unpaired|raw|COMPLETE|12.582193 ± 1.461535|
|DS06|unpaired|clip125|COMPLETE|12.582193 ± 1.461535|
|DS07|clean|raw|COMPLETE|8.423422 ± 0.186463|
|DS07|clean|clip125|COMPLETE|8.423422 ± 0.186463|
|DS07|density|raw|COMPLETE|8.577576 ± 0.556397|
|DS07|density|clip125|COMPLETE|8.577576 ± 0.556397|
|DS07|unpaired|raw|COMPLETE|8.435539 ± 0.439675|
|DS07|unpaired|clip125|COMPLETE|8.435539 ± 0.439675|

- Development engines only; no official TEST or paper-native protocol reproduction.
- Inner HPO scores and 72 outer evaluations are never pooled.
- Seed first averages six heldout-engine RMSE values; SD is across three seed means.
- Six folds share training engines; seeds are not independent engines; no significance claim.
- Training seconds exclude inference/IO; incomplete snapshots are not final performance claims.
- Collector checks CPU model construction/state, not forward; replay is a separate explicit operation.

Errors: 0; see COLLECTION.json.
