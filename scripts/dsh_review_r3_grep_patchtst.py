import re
t = open('results/dsh/review_r3/sources/patchtst.txt', encoding='utf-8', errors='ignore').read()
pages = t.split('\f')
print('pages:', len(pages))
def hits(term, ctx=300, maxhits=3):
    out=[]
    for i,pg in enumerate(pages,1):
        s=re.sub(r'\s+',' ',pg)
        for m in re.finditer(term, s, re.I):
            out.append((i, s[max(0,m.start()-ctx):m.start()+ctx]))
            if len(out)>=maxhits: return out
    return out
for term in [r'channel.independen', r'patch length P\s*=', r'supervised', r'instance normali']:
    print(f'\n### {term}')
    for pg, sn in hits(term):
        print(f'  [p{pg}] {sn[:380]}')
