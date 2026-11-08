import pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
r=pd.read_csv(H/'RESULTS.csv')
print('cols:', list(r.columns))
print(r.head(2).to_string())