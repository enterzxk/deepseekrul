from pathlib import Path
p=Path('d6_stats.py'); s=p.read_text()
s=s.replace("'%-6s %5s %8s %8s %8s %8s %8s' % ('subset','n_eng','RMSE','MAE','NASA','bias','CI_lo','CI_hi')",
            "'%-6s %5s %8s %8s %8s %8s %8s %8s' % ('subset','n_eng','RMSE','MAE','NASA','bias','CI_lo','CI_hi')")
p.write_text(s); print('fixed')