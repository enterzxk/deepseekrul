
import os, json
from pypdf import PdfReader
files = [
 "docs/papers/Empirical Calibration and Conditional-Reliability Diagnostics for Bearing RUL Prediction under Operating-Regime Shift.pdf",
 "docs/papers/Retrieval-Corrected Conformal Prediction for Time Series.pdf",
 "docs/papers/Turbofan Engine RUL Prediction with Reliable Prediction Intervals via LSTM Quantile Regression and Conformal Calibration.pdf",
 "docs/papers/Conformal Prediction for Time Series with Deep Sequence Models.pdf",
]
res = {}
for f in files:
    try:
        r = PdfReader(f)
        txt = " ".join((r.pages[i].extract_text() or "") for i in range(min(2, len(r.pages))))
        res[os.path.basename(f)[:60]] = " ".join(txt.split())[:1500]
    except Exception as e:
        res[os.path.basename(f)[:60]] = "ERR " + str(e)
open("tmp/_ex.json","w",encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
print("done")

