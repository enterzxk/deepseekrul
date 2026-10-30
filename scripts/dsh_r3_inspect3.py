lines = open('scripts/run_candidate.py', encoding='utf-8', errors='ignore').read().splitlines()
for i in range(405, 432):
    print(f'{i+1:5d} {lines[i]}')
