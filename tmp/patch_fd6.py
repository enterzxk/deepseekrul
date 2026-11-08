from pathlib import Path
p=Path('FINAL_DECISION.md'); s=p.read_text()
old = "| FD003 | 待补 | | | | | |"
new = "| FD003（100 台） | M0 | 10.863 | 7.412 | 2.00 | −1.72 | — |\n| FD003 | FULL | 10.158 | 6.995 | 1.73 | −1.13 | RMSE −6.49% / MAE −5.63% / NASA −13.58% |"
assert old in s
s = s.replace(old, new)
s = s.replace("FD001 判定：**PASS**（三项均在允许退化范围内，实际略优）。",
              "FD001 判定：**PASS**（三项均在允许退化范围内，实际略优）。\nFD003 判定：**PASS**（三项均显著优于 M0）。")
p.write_text(s); print('patched')
