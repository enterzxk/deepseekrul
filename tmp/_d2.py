
import json, re
from pypdf import PdfReader
files = {
 "scc": "docs/papers/Similarity-calibrated conformal prediction data-free coverage guarantees for remaining-useful-life intervals under operating-regime transfer.pdf",
 "rccp": "docs/papers/Retrieval-Corrected Conformal Prediction for Time Series.pdf",
 "query": "docs/papers/Query based learning via conformal uncertainty for RUL prediction.pdf",
 "turbofan": "docs/papers/Turbofan Engine Remaining Useful Life Prediction with Reliable Prediction Intervals via LSTM-Based Quantile Regression and Conformal Calibration.pdf",
}
HEAD = re.compile(r"(^|\s)(\d\.?\s*)?(Method|Methodology|Proposed|Approach|Experiments?|Results?|Evaluation)\b", re.I)
out = {}
for k, f in files.items():
    r = PdfReader(f)
    pages = [" ".join((p.extract_text() or "").split()) for p in r.pages]
    hits = []
    for i, t in enumerate(pages):
        for m in HEAD.finditer(t[:600]):
            hits.append((i, m.group(0).strip(), t[m.start():m.start()+560]))
            break
    out[k] = {"pages": len(pages), "hits": hits[:6]}
open("tmp/_deep2.json","w",encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
print("ok")

