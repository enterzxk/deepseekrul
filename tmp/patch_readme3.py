import io
p='README_LOCAL.md'
s=io.open(p,encoding='utf-8').read()
a=s.find('## 4. A 门槛逐条')
assert a!=-1
n=io.open('readme3b.md',encoding='utf-8').read()
io.open(p,'w',encoding='utf-8').write(s[:a]+n+s[a:])
print('ok',len(s))