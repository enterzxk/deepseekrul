import re
t = open('results/dsh/review_r3/sources/p2607_24809.txt', encoding='utf-8', errors='ignore').read()
pages = t.split('\f')
def find(term, ctx=260, maxhits=4):
    out = []
    for i, pg in enumerate(pages, 1):
        s = re.sub(r'\s+', ' ', pg)
        for m in re.finditer(term, s, re.I):
            out.append((i, s[max(0,m.start()-ctx):m.start()+ctx]))
            if len(out) >= maxhits: return out
    return out
for term in [r'tEOL\s*,?\s*known', r'known total', r'multipl', r'test unit']:
    print(f'\n### {term}')
    for pg, snip in find(term):
        print(f'  [p{pg}] {snip[:420]}')
