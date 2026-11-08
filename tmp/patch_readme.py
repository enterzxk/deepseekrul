from pathlib import Path
p=Path('README.md'); s=p.read_text()
start=s.find('## 6. 负结果')
end=s.find('## 7. 已排除的泄漏陷阱')
assert start!=-1 and end!=-1 and end>start
new=open('readme6.md').read()
s=s[:start]+new+s[end:]
p.write_text(s); print('patched', len(s))