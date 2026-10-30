import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS

df = load_cmapss_train(Path("data/CMAPSS"), "FD002")
print("selected:", SELECTED_SENSOR_COLUMNS)
cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
print("cols found:", cols)
print("nan per col:", {c: int(df[c].isna().sum()) for c in cols if df[c].isna().sum() > 0})
print("dtypes nonzero:", {c: str(df[c].dtype) for c in cols[:3]})
print(df[cols].head(2).to_string())
