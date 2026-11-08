from pathlib import Path
p = Path('/home/user/rulb/tim_20261105/full_20261105/manuscript_v4.tex')
s = p.read_text()
import re
for cmd in ['emph', 'textbf', 'texttt']:
    opens = len(re.findall(r'\\\\' + cmd + r'\\\\{', s))
    print(cmd, 'opens:', opens)
# brace balance per line for emph
depth = 0; start = None
for i, line in enumerate(s.split('\n')):
    j = 0
    while j < len(line):
        if line.startswith(r'\emph{', j):
            if depth == 0: start = (i+1, line[:70])
            depth += 1; j += 6; continue
        if line[j] == '}' and depth > 0: depth -= 1
        j += 1
if depth != 0: print('UNCLOSED emph starting at', start, 'depth', depth)
else: print('emph balanced')