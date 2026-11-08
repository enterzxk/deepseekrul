
import json
from pypdf import PdfReader
files = {
 "similarity": "docs/papers/Similarity-calibrated conformal prediction data-free coverage guarantees for remaining-useful-life intervals under operating-regime transfer.pdf",
 "query": "docs/papers/Query based learning via conformal uncertainty for RUL prediction.pdf",
 "turbofan": "docs/papers/Turbofan Engine Remaining Useful Life Prediction with Reliable Prediction Intervals via LSTM-Based Quantile Regression and Conformal Calibration.pdf",
}
out = {}
for k, f in files.items():
    try:
        r = PdfReader(f)
        t = " ".join((r.pages[0].extract_text() or "").split())
        t2 = " ".join((r.pages[1].extract_text() or "").split()) if len(r.pages) > 1 else ""
        out[k] = {"pages": len(r.pages), "p1": t[:1400], "p2": t2[:700]}
    except Exception as e:
        out[k] = {"err": str(e)[:100]}
open("tmp/_newpapers.json","w",encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
print("ok")

