
import json, re
from pypdf import PdfReader
out = {}
r = PdfReader("docs/papers/Conformal prediction beyond exchangeability.pdf")
for pi in range(min(6, len(r.pages))):
    t = r.pages[pi].extract_text() or ""
    for kw in ["weighted", "coverage gap", "Theorem 1", "distribution shift"]:
        for m in re.finditer(kw, t, re.I):
            seg = " ".join(t[max(0,m.start()-180):m.start()+220].split())
            out.setdefault(kw, []).append(f"p{pi+1}: {seg}")
out = {k: v[:3] for k,v in out.items()}
r2 = PdfReader("docs/papers/Conformal Coverage of Time Series Validity and Inference.pdf")
t2 = " ".join((r2.pages[i].extract_text() or "") for i in range(min(3,len(r2.pages))))
m = re.search(r"first central limit theorem", t2, re.I)
out["clt"] = [" ".join(t2[max(0,m.start()-200):m.start()+200].split())] if m else ["NOT FOUND"]
open("tmp/_verify.json","w",encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
print("ok")

