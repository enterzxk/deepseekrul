from pathlib import Path
p = Path('/home/user/rulb/tim_20261105/full_20261105/manuscript_v4.tex')
L = p.read_text().split('\n')
bs = chr(92)
def fig(name, cap):
    return [bs+'begin{figure}[t]', bs+'centering',
            bs+'includegraphics[width='+bs+'columnwidth]{paper_figures_v4/'+name+'}',
            bs+'caption{'+cap+'}', bs+'end{figure}']
# 1) compare figure after the main table's table env
i = next(k for k, s in enumerate(L) if s.startswith(bs+'label{tab:main}'))
j = next(k for k, s in enumerate(L) if k > i and s.startswith(bs+'end{table}'))
L[j+1:j+1] = [''] + fig('fig_compare.pdf', 'Official-test RMSE per fleet. Grey: the 2026 baseline; blue: our ensemble.')
# 2) scatter after 'Table~\\ref{tab:main} reports' paragraph block
k2 = next(k for k, s in enumerate(L) if 'worst single models on FD003' in s)
L[k2+1:k2+1] = [''] + fig('fig_scatter.pdf', 'Ensemble prediction against true RUL on the official test engines. Dashed: y = x.')
# 3) bias figure before the 'Where the residual error sits' subsection
k3 = next(k for k, s in enumerate(L) if s.startswith(bs+'subsection{Where the residual error sits}'))
L[k3:k3] = fig('fig_bias.pdf', 'Mean signed error of the ensemble against true RUL, 25-cycle bins. Positive means over-estimation of remaining life.') + ['']
p.write_text('\n'.join(L))
print('figures inserted:', sum(1 for s in L if 'includegraphics' in s))