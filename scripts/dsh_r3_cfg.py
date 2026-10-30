lines = open('scripts/run_candidate.py', encoding='utf-8', errors='ignore').read().splitlines()
for i, l in enumerate(lines[50:105], start=51):
    print(f'{i:4d} {l}')
