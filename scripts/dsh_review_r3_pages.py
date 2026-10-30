import re
p = 'results/dsh/review_r3/sources/p2607_24809.txt'
pages = open(p, encoding='utf-8', errors='ignore').read().split('\f')
print('pages:', len(pages))
for n in (16, 20, 29):
    t = re.sub(r'\s+', ' ', pages[n-1]) if n-1 < len(pages) else ''
    print(f'\n===== p{n} ({len(t)} chars) =====')
    print(t[:1200])
