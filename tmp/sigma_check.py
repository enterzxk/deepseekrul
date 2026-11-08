import sys, torch, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "."); sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import run_e2e_cqr as E
import run_ida as R
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for sub in ["FD001", "FD002", "FD003", "FD004"]:
    folds, df = R.outer_folds(sub)
    tr_u, va_u = folds[0]
    prior_dir = Path("experiments/ida/prior_boot")
    import pandas as pd, json
    table = pd.read_csv(prior_dir / (sub + "_prior_cutoffs.csv"))
    meta_p = json.loads((prior_dir / (sub + "_prior_meta.json")).read_text())
    median_T = float(meta_p["folds"]["0"]["median_T_train"])
    sp = R.StaircasePrior(table, {int(k): v for k, v in ((fk, fm["median_T_train"]) for fk, fm in meta_p["folds"].items())}, sub)
    (xtr, ytr, etr, mtr, ctr), (xva, yva, eva, mva, cva) = R.prep_fold(df, sub, 0, tr_u, va_u, window=50)
    ptr, (mu_tr, sg_tr) = R.prior_vectors(mtr, sp, 0, "M1", median_T)
    pva, (mu_va, sg_va) = R.prior_vectors(mva, sp, 0, "M1", median_T)
    ck = Path("experiments/punet_e2e") / sub / "f0_s42" / "e2e_cqr_none.json"
    print(sub, "sigma available:", sg_va is not None, "n:", len(sg_va))
