"""
build_classifier.py  —  Offline: the benefit-risk classifier (Agent 3's model)

Trains TWO calibrated models to predict `benefited` from leakage-safe patient
features at the decision point T, then PICKS THE WINNER on a held-out test set:
  - Logistic Regression (interpretable, naturally calibrated)
  - HistGradientBoosting  (stronger on tabular; calibrated via CalibratedClassifierCV)

Reports ROC-AUC, PR-AUC, Brier (calibration), confusion matrix, baselines,
calibration curve, and subgroup AUCs. Saves the winning model for Agent 3.

Run:  python build_classifier.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\Prashant\claude-test\Capstone")
import retrieval_agent as ra                      # reuse features/label/split (consistency)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_score, recall_score, f1_score, confusion_matrix)
from joblib import dump

ARTIFACT = r"C:\Users\Prashant\claude-test\Capstone\agent_artifacts\benefit_risk_model.joblib"

# ---- leakage-safe FEATURES only (NO label-derived cols: churn/B/C/survival/death) ----
NUM = ["age_dx", "T", "prog_events_before_T", "scans_before_T", "regimens_before_T"]
CAT = ["sex", "race", "stage", "grade", "oncotree", "sample_type", "pre_status"]
FEATURES = NUM + CAT

def preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])

def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                brier=brier_score_loss(y, p), precision=precision_score(y, yhat, zero_division=0),
                recall=recall_score(y, yhat, zero_division=0), f1=f1_score(y, yhat, zero_division=0))

def show(name, m):
    print(f"  {name:24s} AUC={m['auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  Brier={m['brier']:.3f}  "
          f"P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")

if __name__ == "__main__":
    print("[1/5] Building features + label (leakage-safe) ...")
    df = ra.build_patient_table()
    if "cpt_oncotree_code" in df.columns:      # normalize histology column name
        df = df.rename(columns={"cpt_oncotree_code": "oncotree"})
    # add institution for subgroup reporting (from index file)
    inst = ra.L("cancer_level_dataset_index.csv").groupby("record_id")["institution"].first()
    df = df.merge(inst.rename("institution"), on="record_id", how="left")

    train, test = ra.split(df)                       # same patient-level stratified split as retrieval
    Xtr, ytr = train[FEATURES], train["benefited"].values
    Xte, yte = test[FEATURES], test["benefited"].values
    print(f"      train={len(train)} (pos {ytr.mean()*100:.1f}%) | test={len(test)} (pos {yte.mean()*100:.1f}%)")

    models = {
        "LogisticRegression": Pipeline([("pre", preprocessor()),
                                        ("clf", LogisticRegression(max_iter=2000, C=1.0))]),
        "HistGradientBoosting": Pipeline([("pre", preprocessor()),
                                          ("clf", HistGradientBoostingClassifier(
                                              max_depth=3, learning_rate=0.05, max_iter=300,
                                              l2_regularization=1.0, random_state=ra.SEED))]),
    }

    print("\n[2/5] 5-fold CV ROC-AUC on TRAIN (discrimination, model selection) ...")
    cv = StratifiedKFold(5, shuffle=True, random_state=ra.SEED)
    for name, pipe in models.items():
        s = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc")
        print(f"  {name:24s} CV AUC = {s.mean():.3f} +/- {s.std():.3f}")

    print("\n[3/5] Calibrate (Platt/sigmoid) + evaluate on HELD-OUT TEST ...")
    fitted, test_metrics = {}, {}
    for name, pipe in models.items():
        cal = CalibratedClassifierCV(pipe, method="sigmoid", cv=5)
        cal.fit(Xtr, ytr)
        p = cal.predict_proba(Xte)[:, 1]
        fitted[name] = cal; test_metrics[name] = metrics(yte, p)
        show(name + " (calibrated)", test_metrics[name])

    print("\n  baselines:")
    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    show("majority-class", metrics(yte, dummy.predict_proba(Xte)[:, 1]))
    stage_only = Pipeline([("pre", ColumnTransformer([("c",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["stage"])])),
                    ("clf", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
    show("stage-only LogReg", metrics(yte, stage_only.predict_proba(Xte)[:, 1]))

    print("\n[4/5] Pick winner (by test ROC-AUC, tie-break Brier) ...")
    winner = max(test_metrics, key=lambda k: (round(test_metrics[k]["auc"], 4),
                                              -test_metrics[k]["brier"]))
    print(f"      WINNER: {winner}")
    wm = test_metrics[winner]; pw = fitted[winner].predict_proba(Xte)[:, 1]
    cm = confusion_matrix(yte, (pw >= 0.5).astype(int))
    print(f"      confusion matrix [tn fp / fn tp]: {cm.ravel().tolist()}")
    # calibration curve
    frac_pos, mean_pred = calibration_curve(yte, pw, n_bins=5, strategy="quantile")
    print("      calibration (pred -> actual):  " +
          "  ".join(f"{mp:.2f}->{fp:.2f}" for mp, fp in zip(mean_pred, frac_pos)))

    print("\n[5/5] Subgroup ROC-AUC (winner) + save model ...")
    te = test.copy(); te["p"] = pw
    te["age_band"] = pd.cut(te["age_dx"], [0, 60, 70, 200], labels=["<60", "60-70", "70+"])
    for col in ["institution", "stage", "sex", "age_band"]:
        print(f"  by {col}:")
        for gv, g in te.groupby(col, observed=True):
            if len(g) >= 20 and g["benefited"].nunique() == 2:
                print(f"      {str(gv):<28} n={len(g):>3}  AUC={roc_auc_score(g['benefited'], g['p']):.3f}")

    os.makedirs(os.path.dirname(ARTIFACT), exist_ok=True)
    dump({"model": fitted[winner], "name": winner, "features": FEATURES,
          "test_auc": wm["auc"], "test_brier": wm["brier"]}, ARTIFACT)
    print(f"\nSaved winning model -> {ARTIFACT}")
    print(f"Summary: {winner}  AUC={wm['auc']:.3f}  PR-AUC={wm['pr_auc']:.3f}  Brier={wm['brier']:.3f}  "
          f"(target was AUC>=0.70, well-calibrated)")
