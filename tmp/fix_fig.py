from pathlib import Path
p=Path('manuscript.tex'); s=p.read_text()
for f in ['fig_error_by_age','fig_per_engine_scatter','fig_observability']:
    s=s.replace('{'+f+'.png}','{paper_figures/'+f+'.png}')
p.write_text(s); print(s.count('paper_figures/fig_'))