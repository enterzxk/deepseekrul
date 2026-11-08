import pandas as pd
r=pd.read_csv('RESULTS.csv')
print(r[['subset','arm','rmse_mean','nasa','reference','gain']].to_string(index=False))