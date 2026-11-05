"""Open final labels once, after both prospectively declared budgets are sealed."""
import hashlib
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from score_final import metrics
from evidence import HERE,ROOT,FLEETS,delta


def summarize(predictions,y):
    result={"engines":len(y),"single_seed":{},"ensemble":{},"ensemble_contrasts":{}}
    ensembles={}
    for arm,models in predictions.items():
        assert all(len(f)==len(y) for f in models.values())
        result["single_seed"][arm]={str(s):metrics(y,f.y_pred.to_numpy()) for s,f in models.items()}
        ensembles[arm]=np.mean([f.y_pred.to_numpy() for f in models.values()],axis=0)
        result["ensemble"][arm]=metrics(y,ensembles[arm])
    for a,b in [("FULL","M0"),("FULL","OPS"),("FULL","CET")]:
        result["ensemble_contrasts"][f"{a}-{b}"]=delta(pd.Series((ensembles[a]-y)**2),pd.Series((ensembles[b]-y)**2))
    for seed in [101,202,303]:
        result.setdefault("single_seed_contrasts",{})[str(seed)]={}
        for a,b in [("FULL","M0"),("FULL","OPS")]:
            aa=predictions[a][seed].y_pred.to_numpy()
            bb=predictions[b][seed].y_pred.to_numpy()
            result["single_seed_contrasts"][str(seed)][f"{a}-{b}"]=delta(pd.Series((aa-y)**2),pd.Series((bb-y)**2))
    return result


def main():
    if (HERE/"test_opening_completed.json").exists():
        return
    assert not (HERE/"official_results.json").exists()
    assert not (HERE/"official_fullbudget_results.json").exists()
    data={}
    manifest={}
    for budget in ["official","official_fullbudget"]:
        data[budget]={}
        for s in FLEETS:
            assert (HERE/budget/s/"done.json").is_file()
            data[budget][s]={}
            for arm in ["M0","OPS","FULL","CET"]:
                models={}
                for seed in [101,202,303]:
                    path=HERE/budget/s/f"{arm}_{seed}.csv"
                    frame=pd.read_csv(path)
                    assert np.array_equal(frame.unit,np.arange(1,len(frame)+1))
                    assert np.isfinite(frame.y_pred).all() and frame.y_pred.between(0,125).all()
                    models[seed]=frame
                    manifest[str(path.relative_to(HERE))]=hashlib.sha256(path.read_bytes()).hexdigest()
                data[budget][s][arm]=models
    (HERE/"official_predictions_sealed.json").write_text(json.dumps(manifest,indent=2))
    opened=datetime.now(timezone.utc).isoformat()
    results={b:{"opened_at_utc":opened,"subsets":{}} for b in data}
    for s in FLEETS:
        y=np.loadtxt(ROOT/"data/CMAPSS"/f"RUL_{s}.txt").reshape(-1).clip(0,125)
        for budget in data:
            results[budget]["subsets"][s]=summarize(data[budget][s],y)
    for budget,result in results.items():
        (HERE/f"{budget}_results.json").write_text(json.dumps(result,indent=2))
    (HERE/"test_opening_completed.json").write_text(json.dumps(dict(opened_at_utc=opened,
        predictions_sealed=len(manifest),subsets=FLEETS,budgets=list(data)),indent=2))
    print("Scored both fixed budgets. No further test-driven fitting is permitted.")

if __name__=="__main__":
    main()
