"""
plot_calibration.py — Render the calibration curve from the SAVED benefit-risk model.

Reuses retrieval_agent's features + the SAME patient-level split as build_classifier.py,
loads the calibrated winning model, predicts on the held-out test set, and saves a
calibration-curve figure (reliability diagram + prediction histogram).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))   # repo folder, resolved at runtime (portable; no hard-coded path)
sys.path.insert(0, HERE)
import retrieval_agent as ra
from joblib import load
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACT = os.path.join(HERE, "agent_artifacts", "benefit_risk_model.joblib")
OUT = os.path.join(HERE, "calibration_curve.png")   # written into the repo folder (no hard-coded user path)

# --- rebuild the exact held-out test set (same as build_classifier.py) ---
art = load(ARTIFACT)
model, FEATURES, name = art["model"], art["features"], art["name"]

df = ra.build_patient_table()
if "cpt_oncotree_code" in df.columns:
    df = df.rename(columns={"cpt_oncotree_code": "oncotree"})
_, test = ra.split(df)
Xte, yte = test[FEATURES], test["benefited"].values

p = model.predict_proba(Xte)[:, 1]
brier = brier_score_loss(yte, p)
auc = roc_auc_score(yte, p)

# 5-bin quantile curve (matches build_classifier.py); 10-bin uniform for finer view
fp5, mp5 = calibration_curve(yte, p, n_bins=5, strategy="quantile")
fp10, mp10 = calibration_curve(yte, p, n_bins=10, strategy="uniform")

print(f"Model: {name} | n_test={len(yte)} | AUC={auc:.3f} | Brier={brier:.3f}")
print("5-bin quantile (pred -> actual): " +
      "  ".join(f"{a:.2f}->{b:.2f}" for a, b in zip(mp5, fp5)))

# --- figure: reliability diagram (top) + histogram of predictions (bottom) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7.5),
                               gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

ax1.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
ax1.plot(mp10, fp10, "o-", color="#2c6fbb", lw=1.5, ms=5, alpha=0.7,
         label="10-bin (uniform)")
ax1.plot(mp5, fp5, "s-", color="#c0392b", lw=2, ms=8,
         label="5-bin (quantile, as reported)")
ax1.set_ylabel("Observed benefit rate (actual)")
ax1.set_title(f"Calibration Curve — Benefit-Risk Classifier ({name})\n"
              f"Held-out test (n={len(yte)})   ROC-AUC={auc:.3f}   Brier={brier:.3f}",
              fontsize=11)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

ax2.hist(p, bins=20, range=(0, 1), color="#2c6fbb", alpha=0.7, edgecolor="white")
ax2.set_xlabel("Predicted probability of benefit")
ax2.set_ylabel("Count")
ax2.grid(alpha=0.3)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print("Saved:", OUT)
