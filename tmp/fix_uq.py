from pathlib import Path
p = Path('/home/user/rulb/d5_uq.py')
s = p.read_text()
old = "        p = V / ('M_' + tag) / subset / ('dspt_s' + tag.split('_s')[1] + '.csv')"
new = "        seed = tag.split('_s')[1].split('_')[0]" + chr(10) + "        p = V / ('M_' + tag) / subset / ('dspt_s' + seed + '.csv')"
assert old in s
s = s.replace(old, new)
p.write_text(s)
print('fixed')