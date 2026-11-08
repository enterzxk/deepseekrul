from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
old='A = 主判据满足、种子一致性待补；B = 待种子确认后转 PASS'
new='A = PASS（两种子、含预注册尾部保护）；B = PASS'
assert old in s
s=s.replace(old,new)
s=s.replace('最终状态 UNMET（待种子一致性确认后可转 MET）。**','最终状态 MET。**')
p.write_text(s); print('ok')