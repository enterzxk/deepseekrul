
import json, re, os
from pypdf import PdfReader
files = {
 "split_cp": "docs/papers/Valid and Efficient Split Conformal Regression for Time Series.pdf",
 "rccp": "docs/papers/Retrieval-Corrected Conformal Prediction for Time Series.pdf",
 "localized": "docs/papers/Conformal Prediction for Time Series with Deep Sequence Models.pdf",
 "qr_ssm": "docs/papers/A Quantile Regression Approach for Remaining Useful Life Estimation with State Space Models.pdf",
 "qr_biz": "docs/papers/Uncertainty and Business-Aware Remaining Useful Life Estimation for Semiconductor Manufacturing.pdf",
}
res = {}
def grab(t, kw, n=2, w=260):
    out=[]
    for m in re.finditer(kw, t, re.I):
        out.append(" ".join(t[max(0,m.start()-w):m.start()+w].split()))
        if len(out)>=n: break
    return out
for k,f in files.items():
    try:
        r = PdfReader(f)
        full = "\n".join((r.pages[i].extract_text() or "") for i in range(len(r.pages)))
        t = " ".join(full.split())
        res[k] = {
          "pages": len(r.pages),
          "abstract": t[:600],
          "code": grab(t, r"github|code is available|publicly available", 2, 200),
          "coverage": grab(t, r"coverage level|nominal coverage|90% coverage", 2, 220),
          "dataset": grab(t, r"dataset", 1, 200),
        }
    except Exception as e:
        res[k] = {"err": str(e)}
open("tmp/_c1.json","w",encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
print("ok")

