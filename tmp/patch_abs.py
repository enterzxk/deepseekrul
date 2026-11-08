from pathlib import Path
p=Path('PAPER_DRAFT.md'); s=p.read_text()
s=s.replace("Additive injection reduces the\npooled RMSE by 3.1%, but replacing the cycle coordinate by c/mu_T increases it by 7.7% and\nmultiplying a predicted life fraction by mu_T increases it by 28%.",
            "Additive injection reduces the\npooled RMSE by 3.3%, but replacing the cycle coordinate by c/mu_T increases it by 7.5% and\nmultiplying a predicted life fraction by mu_T increases it by 28%.")
p.write_text(s); print('patched')
