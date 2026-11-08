"""Delivery verification: required files, table completeness, figure presence."""
from pathlib import Path
import pandas as pd, json
HERE = Path('/home/user/rulb/tim_20261105/full_20261105')
required = ["PROTOCOL.md","NOVELTY_MATRIX.md","IDENTIFIABILITY.md","ENDPOINT_DEPENDENCE.md",
            "REPRODUCIBILITY.md","LEAKAGE_AUDIT.md","FINAL_DECISION.md","KEY_FINDINGS.md",
            "TIM_PAPER_PACKAGE.md","PAPER_DRAFT.md","INDEX.md","RESULTS.csv","ABLATIONS.csv",
            "MEASUREMENT_CONDITIONS.csv","PER_ENGINE.csv","REFERENCE_LOCK.json","runtime.json"]
missing=[f for f in required if not (HERE/f).is_file()]
print("required files:", "ALL PRESENT" if not missing else "MISSING "+str(missing))
r=pd.read_csv(HERE/'RESULTS.csv')
print("RESULTS.csv rows:", len(r), "subsets:", sorted(r.subset.unique()), "arms:", sorted(r.arm.unique()))
print(r.groupby('subset').agg(arms=('arm','nunique'), rows=('arm','size')).to_string())
figs=sorted(p.name for p in (HERE/'paper_figures').glob('*.png'))
print("figures:", len(figs), figs)
tabs=sorted(p.name for p in (HERE/'paper_tables').glob('*'))
print("tables:", tabs)
lock=json.loads((HERE/'REFERENCE_LOCK.json').read_text())
print("reference lock:", {k:(v['reference'],round(v['reference_rmse'],3)) for k,v in lock['subsets'].items()})
n_pred=sum(1 for _ in (HERE/'runs').rglob('*.csv'))
print("prediction CSVs:", n_pred)
