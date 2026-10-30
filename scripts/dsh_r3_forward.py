t = open('src/candidates/model.py', encoding='utf-8', errors='ignore').read().splitlines()
for i in range(285, 320):
    print(f'{i+1:4d} {t[i]}')
print('=== forward signature ===')
import re
for i, l in enumerate(t):
    if 'def forward' in l:
        print(f'{i+1:4d} {l}')
