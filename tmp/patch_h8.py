from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
a=s.find('### H.8 适用范围检查')
assert a!=-1
s=s[:a]+open('h8b.md').read().lstrip()
p.write_text(s); print('replaced')