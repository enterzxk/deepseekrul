import io
p='README_LOCAL.md'
s=io.open(p,encoding='utf-8').read()
a=s.find('## 6.'); b=s.find('## 7.')
n=io.open('readme6.md',encoding='utf-8').read()
io.open(p,'w',encoding='utf-8').write(s[:a]+n+s[b:])
print('ok',a,b,len(s))