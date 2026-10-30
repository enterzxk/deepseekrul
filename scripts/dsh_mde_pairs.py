import numpy as np
pairs = {
  'M1 vs M6 (r=0.72)': (238.9, 0.718, 12.93),
  'M1 vs M0 (r=-0.01)': (589.9, -0.014, 12.93),
}
print('pair                  SD(diff)   n=52 MDE   n=260 MDE   n=709 MDE')
for name, (sd, r, rmse) in pairs.items():
    row = [name.ljust(20), f'{sd:8.1f}']
    for n in (52, 260, 709):
        se = sd/np.sqrt(n)/(2*rmse)
        row.append(f'{2.8*se:10.2f}')
    print('  '.join(row))
print()
print('interpretation: a method that is BEHAVIOURALLY DIFFERENT (low r) has a LARGER')
print('difference SD, hence a WORSE detection floor - the two goals conflict.')
