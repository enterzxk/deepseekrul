from pathlib import Path
p=Path('manuscript.tex'); s=p.read_text()

# 1) shrink the negative-results table
s=s.replace('\\label{tab:neg}\n\\small','\\label{tab:neg}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}')

# 2) add the error-by-age figure after the span subsection
anchor='\\subsection{The guard removes a tail'
fig1=('\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\columnwidth]{fig_error_by_age.png}\n'
      '\\caption{Pooled error against the age of the engine at the prediction time. The gain of the span schedule is concentrated where a long history exists.}\n'
      '\\label{fig:age}\n\\end{figure}\n\n')
assert anchor in s
s=s.replace(anchor, fig1+anchor, 1)

# 3) add the per-engine scatter after the main results paragraph
anchor2='\\subsection{The span is worth more'
fig2=('\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\columnwidth]{fig_per_engine_scatter.png}\n'
      '\\caption{Per-engine RMSE of the plain backbone against the prior arm. The pooled improvement comes from a few large gains outweighing a few large losses.}\n'
      '\\label{fig:eng}\n\\end{figure}\n\n')
s=s.replace(anchor2, fig2+anchor2, 1)

# 4) add the observability figure in the discussion
anchor3='\\section{Conclusion}'
fig3=('\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\columnwidth]{fig_observability.png}\n'
      '\\caption{Rank correlation between the individual lifetime estimate and the true lifetime as a function of the observation cutoff. The scale is essentially unobservable before 100 cycles.}\n'
      '\\label{fig:obs}\n\\end{figure}\n\n')
s=s.replace(anchor3, fig3+anchor3, 1)

p.write_text(s)
print('figures:', s.count('includegraphics'))