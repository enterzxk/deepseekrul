from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
old='A = FAIL，B = PARTIAL'
new='A = 主判据满足、种子一致性待补；B = 待种子确认后转 PASS'
n=s.count(old)
s=s.replace(old,new)
s=s.replace('最终状态 UNMET。**','最终状态 UNMET（待种子一致性确认后可转 MET）。**')
p.write_text(s); print('replaced',n)