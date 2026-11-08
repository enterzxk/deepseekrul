
import json, os
from pypdf import PdfReader

PLAN = {
 "1-s2.0-S0360835226001804-main.pdf": ("Deep evidential transformer with Monte Carlo dropout for uncertainty-aware remaining useful life prediction", "Deep evidential"),
 "1-s2.0-S0951832025003783-main.pdf": ("Robust uncertainty quantification for online remaining useful life prediction with randomly missing and partially faulty sensor data", "Robust uncertainty quantification"),
 "1-s2.0-S0951832025009780-main.pdf": ("A physics-constrained Bayesian neural network for machinery remaining useful life prediction and uncertainty quantification", "physics-constrained Bayesian"),
 "1-s2.0-S0951832026014365-main.pdf": ("Similarity-calibrated conformal prediction data-free coverage guarantees for remaining-useful-life intervals under operating-regime transfer", "Similarity-calibrated"),
 "1-s2.0-S2090447926000195-main.pdf": ("Uncertainty aware predictive maintenance using a hybrid Transformer with Monte Carlo Dropout and conformal prediction", "Uncertainty aware predictive maintenance"),
 "Query_based_learning_via_conformal_uncertainty_for_RUL_prediction.pdf": ("Query based learning via conformal uncertainty for RUL prediction", "Query based learning"),
 "Remaining_Useful_Life_Prediction_and_Uncertainty_Quantification_with_Multi-Branch_Deep_Networks_and_Probabilistic_Calibration.pdf": ("Remaining Useful Life Prediction and Uncertainty Quantification with Multi-Branch Deep Networks and Probabilistic Calibration", None),
 "sensors-26-02249.pdf": ("Turbofan Engine Remaining Useful Life Prediction with Reliable Prediction Intervals via LSTM-Based Quantile Regression and Conformal Calibration", "Turbofan"),
}
res = []
for src, (title, must) in PLAN.items():
    p = os.path.join("docs/papers", src)
    if not os.path.exists(p):
        res.append({"src": src, "status": "missing"}); continue
    ok = True
    if must:
        try:
            r = PdfReader(p)
            txt = " ".join((r.pages[0].extract_text() or "").split())
            ok = must.lower() in txt.lower()
        except Exception as e:
            ok = False
    dst = os.path.join("docs/papers", title + ".pdf")
    if ok:
        os.replace(p, dst)
        res.append({"src": src, "status": "renamed", "dst": os.path.basename(dst)})
    else:
        res.append({"src": src, "status": "CHECK_FAILED_kept"})
open("tmp/_rename.json","w",encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
print("done")

