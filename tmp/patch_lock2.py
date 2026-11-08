from pathlib import Path
p=Path('lock_reference.py'); s=p.read_text()
old = """    if not vals:
        continue"""
new = """    if len(vals) < 3:
        print(subset, "incomplete: only", list(vals), "-> not locked")
        continue"""
assert old in s
s = s.replace(old, new)
p.write_text(s); print('patched')
