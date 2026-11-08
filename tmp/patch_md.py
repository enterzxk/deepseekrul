from pathlib import Path
helper = '''

def to_md(frame):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\\n".join(lines)
'''
for f in ['make_ablation_table.py', 'make_paper_assets.py']:
    p = Path(f); s = p.read_text()
    if 'def to_md(' in s:
        print(f, 'already'); continue
    s = s.replace('HERE = Path(__file__).resolve().parent', 'HERE = Path(__file__).resolve().parent' + helper, 1)
    s = s.replace('.to_markdown(index=False)', '')
    s = s.replace('md = frame.round(4)', 'md = to_md(frame.round(4))')
    s = s.replace('(OUT_T / "observability.md").write_text(pd.DataFrame(rows).round(3))',
                  '(OUT_T / "observability.md").write_text(to_md(pd.DataFrame(rows).round(3)))')
    s = s.replace('(HERE / "paper_tables" / "ablations.md").write_text(frame.round(3))',
                  '(HERE / "paper_tables" / "ablations.md").write_text(to_md(frame.round(3)))')
    p.write_text(s); print(f, 'patched')
