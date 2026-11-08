
import sys, glob, os
from pypdf import PdfReader
files = [
 "docs/papers/Empirical Calibration and Conditional-Reliability Diagnostics for Bearing RUL Prediction under Operating-Regime Shift.pdf",
 "docs/papers/Retrieval-Corrected Conformal Prediction for Time Series.pdf",
 "docs/papers/Conformal Coverage of Time Series Validity and Inference.pdf",
 "docs/papers/Turbofan Engine RUL Prediction with Reliable Prediction Intervals via LSTM Quantile Regression and Conformal Calibration.pdf",
 "docs/papers/Conformal Prediction for Time Series with Deep Sequence Models.pdf",
]
for f in files:
    try:
        r = PdfReader(f)
        txt = " ".join((r.pages[i].extract_text() or "") for i in range(min(2, len(r.pages))))
        txt = " ".join(txt.split())
        print("###", os.path.basename(f)[:70], "| pages:", len(r.pages))
        print(txt[:1500])
        print()
    except Exception as e:
        print("ERR", f, e)

