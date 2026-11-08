from pathlib import Path
p=Path('manuscript.tex'); s=p.read_text()
s=s.replace('We tested nine model-side interventions','We tested seven families of model-side intervention')
s=s.replace('nine model-side interventions, four of them parameter-free','seven families of model-side intervention, five of them parameter-free')
s=s.replace('nine model-side interventions, four of them with zero additional parameters','seven families of model-side intervention, five of them with zero additional parameters')
p.write_text(s); print('nine left:', s.count('nine'))