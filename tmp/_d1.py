
import json, re
from pypdf import PdfReader
files = {
 "scc": "docs/papers/Similarity-calibrated conformal prediction data-free coverage guarantees for remaining-useful-life intervals under operating-regime transfer.pdf",
 "rccp": "docs/papers/Retrieval-Corrected Conformal Prediction for Time Series.pdf",
 "query": "docs/papers/Query based learning via conformal uncertainty for RUL prediction.pdf",
 "turbofan": "docs/papers/Turbofan Engine Remaining Useful Life Prediction with Reliable Prediction Intervals via LSTM-Based Quantile Regression and Conformal Calibration.pdf",
}
KEYS = ["method", "experiment", "result", "conclusion", "limitation"]
out = {}
for k, f in files.items():
    try:
        r = PdfReader(f)
        pages = [" ".join((p.extract_text() or "").split()) for p in r.pages]
        full = " ".join(pages)
        info = {"pages": len(pages), "len": len(full)}
        for kw in KEYS:
            m = re.search(r"\b" + kw + r"s?\b", full, re.I)
            info[kw] = full[m.start():m.start()+700] if m else ""
        # capture numbers-heavy results sentence
        nums = re.findall(r"[^.]{0,160}(?:coverage|PICP|Winkler|RMSE)[^.]{0,160}\.", full, re.I)[:3]
        info["numbers"] = nums
        out[k] = info
    except Exception as e:
        out[k] = {"err": str(e)[:120]}
open("tmp/_deep1.json","w",encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
print("ok")

