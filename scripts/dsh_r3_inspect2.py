import re
t = open('scripts/run_candidate.py', encoding='utf-8', errors='ignore').read()
lines = t.splitlines()
def show(name, n=26):
    for i, l in enumerate(lines):
        if re.match(rf'^def {name}', l):
            print(f'--- {name} (line {i+1}) ---')
            for j in range(i, min(i+n, len(lines))): print(f'{j+1:5d} {lines[j]}')
            return
show('build_windows_matched')
show('build_windows', 18)
