import re
lines = open('scripts/run_candidate.py', encoding='utf-8', errors='ignore').read().splitlines()
for i, l in enumerate(lines):
    if re.match(r'^def build_windows\(', l):
        for j in range(i, min(i+22, len(lines))): print(f'{j+1:5d} {lines[j]}')
        break
print('=== callers ===')
for i, l in enumerate(lines):
    if 'build_windows(' in l and 'def ' not in l: print(f'{i+1:5d} {l.strip()[:120]}')
