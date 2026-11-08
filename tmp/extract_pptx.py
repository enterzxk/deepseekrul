import sys, io
from pptx import Presentation
p = r'C:\Users\20562\Desktop\航空发动机RUL_研究方向与实验进展_汇报版_已加入框架图和流程图.pptx'
prs = Presentation(p)
out = []
for i, s in enumerate(prs.slides, 1):
    out.append('===== SLIDE %d =====' % i)
    for sh in s.shapes:
        if sh.has_text_frame:
            t = sh.text_frame.text.strip()
            if t: out.append(t)
        if sh.has_table:
            for r in sh.table.rows:
                out.append(' | '.join(c.text.strip() for c in r.cells))
txt = '\n'.join(out)
io.open(r'C:\Users\20562\Desktop\deepseek-rul\tmp\pptx_text.txt','w',encoding='utf-8').write(txt)
print('slides', len(prs.slides), 'chars', len(txt))