"""
evaluate.py  —  4-tier evaluation harness for the agentic system

  Tier 1  Classifier   : ROC-AUC / PR-AUC / Brier on held-out test (recap from saved model)
  Tier 2  Retrieval    : neighbour label-agreement + "what-worked" hit-rate@k
  Tier 3  Proactivity  : decision-point -> first-progression lead time (proxy)
  Tier 4  End-to-end   : groundedness rate (no hallucinated therapy), deferral correctness,
                         critic revision rate, latency  (LLM, run on a small sample)

Run:  python evaluate.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys
HERE = os.path.dirname(os.path.abspath(__file__))   # repo folder, resolved at runtime (portable; no hard-coded path)
sys.path.insert(0, HERE)
import retrieval_agent as ra
import agent3_timing as a3
import agent4_critic as a4
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

K = 20
PROG = "Progressing/Worsening/Enlarging"

def num(s): return pd.to_numeric(s, errors="coerce")

def drug_tokens(s):
    if not isinstance(s, str): return set()
    out = set()
    for part in s.replace(";", ",").split(","):
        p = part.strip()
        if p: out.add(p)
    return out

# ============================================================ Tier 1
def tier1(C, test):
    art = C["art"]
    p = art["model"].predict_proba(test[art["features"]])[:, 1]
    y = test["benefited"].values
    print("TIER 1  Classifier (held-out test):")
    print(f"  ROC-AUC={roc_auc_score(y,p):.3f}  PR-AUC={average_precision_score(y,p):.3f}  "
          f"Brier={brier_score_loss(y,p):.3f}   (model: {art['name']})")

# ============================================================ Tier 2
def tier2(C, test):
    db, emb, df = C["db"], C["emb"], C["df"]
    base_rate = df["benefited"].mean()
    ben_q, non_q, hits, hit_denom = [], [], 0, 0
    for _, r in test.iterrows():
        card = ra.make_card(r)
        scored = db.similarity_search_with_score(card, k=40)
        labels = [d.metadata.get("benefited") for d, _ in scored][:K]
        frac_ben = np.mean(labels)                       # fraction of top-K neighbours that benefited
        (ben_q if r["benefited"] == 1 else non_q).append(frac_ben)
        # what-worked hit-rate: only for benefited test patients who actually had a regimen
        if r["benefited"] == 1 and isinstance(r.get("regimen_drugs"), str):
            qdrugs = drug_tokens(r["regimen_drugs"])
            rec = set()
            for d, _ in scored:
                if d.metadata.get("benefited") == 1:
                    rec |= drug_tokens(d.metadata.get("regimen_drugs", ""))
            if qdrugs:
                hit_denom += 1
                hits += 1 if (qdrugs & rec) else 0
    print("\nTIER 2  Retrieval quality:")
    print(f"  mean %benefited among top-{K}: benefited-query={np.mean(ben_q)*100:.1f}%  "
          f"non-benefited-query={np.mean(non_q)*100:.1f}%  (cohort base rate {base_rate*100:.1f}%)")
    print(f"  'what-worked' hit-rate@{K}: {hits}/{hit_denom} = "
          f"{(hits/hit_denom*100 if hit_denom else 0):.1f}%  "
          f"(actual effective regimen appears in recommended plans)")

# ============================================================ Tier 3
def tier3(C, test):
    dfs = C.get("dfs") or __import__("journey_builder").load_all()
    cpt = dfs["cpt"]; T = cpt.groupby("record_id")["dx_cpt_rep_days"].min()
    mon = dfs["mon"][["record_id", "dx_md_visit_days", "md_ca_status"]].rename(
        columns={"dx_md_visit_days": "day", "md_ca_status": "st"})
    img = dfs["img"][["record_id", "dx_scan_days", "image_overall"]].rename(
        columns={"dx_scan_days": "day", "image_overall": "st"})
    ev = pd.concat([mon, img], ignore_index=True)
    ev["day"] = num(ev["day"]); ev = ev[ev["st"] == PROG].dropna(subset=["day"])
    ev = ev.join(T.rename("T"), on="record_id")
    post = ev[ev["day"] > ev["T"]]
    first_prog = post.groupby("record_id")["day"].min()
    leads = (first_prog - T).dropna()
    tids = set(test["record_id"])
    leads = leads[leads.index.isin(tids)]
    print("\nTIER 3  Proactivity (decision-point -> first post-T progression; proxy):")
    print(f"  test patients with a post-T progression: {len(leads)}/{len(test)} "
          f"({len(leads)/len(test)*100:.0f}%)")
    print(f"  median lead time T -> first progression: {leads.median():.0f} days "
          f"(IQR {leads.quantile(.25):.0f}-{leads.quantile(.75):.0f})")
    print("  [proxy: shows the proactive window; full visit-by-visit flagging simulation = future work]")

# ============================================================ Tier 4
def narrative_lite(r):
    return (f"{r.get('stage')} {r.get('oncotree') or 'NSCLC'}, age {r.get('age_dx')}; genomic test at "
            f"day {int(r['T'])}; status before test: {r.get('pre_status')}; "
            f"{int(r.get('regimens_before_T',0))} prior regimen(s).")

def tier4(C, test, n=12):
    llm, art, db, emb, df = C["llm"], C["art"], C["db"], C["emb"], C["df"]
    vocab = set()
    for s in df["regimen_drugs"].dropna():
        vocab |= drug_tokens(s)
    CLASS_WORDS = ["chemotherapy", "immunotherapy", "targeted therapy", "radiation", "adjuvant"]
    sample = test.sort_values("p").iloc[:: max(1, len(test)//n)][:n]   # spread across the risk range
    grounded_ok = defer_ok = revised = 0; latencies = []
    for _, r in sample.iterrows():
        t0 = time.perf_counter()
        p = float(art["model"].predict_proba(pd.DataFrame([r[art["features"]]]))[0, 1])
        card = ra.make_card(r)
        _, top, diag = ra.retrieve_cohort(db, emb, card, stage=str(r.get("stage")))  # single pass (no LLM)
        has_ev = len(top) > 0
        plans = ("\n".join(f"  - {d}" for d in top.index) if has_ev
                 else "  (no specific systemic regimen; conservative management)")
        timing = ("INSUFFICIENT EVIDENCE" if diag.get("abstain")
                  else "ACT NOW" if p >= a3.ACT_THRESHOLD else "MONITOR")
        a = {"narrative": narrative_lite(r), "p": p, "timing": timing,
             "unc": a3.assess_uncertainty(p, diag), "plans_txt": plans, "has_evidence": has_ev}
        draft = a3.synthesize(llm, a)
        v, _, _ = a4.safety_review(llm, a, draft)
        latencies.append(time.perf_counter() - t0)
        # groundedness check (rule-based)
        named = {d for d in vocab if d.lower() in draft.lower()}
        allowed = drug_tokens("; ".join(top.index)) if has_ev else set()
        ungrounded = named - allowed
        if not ungrounded:
            grounded_ok += 1
        if not has_ev:                                   # deferral correctness: empty evidence -> no specifics
            class_hit = any(w in draft.lower() for w in CLASS_WORDS)
            if not named and not class_hit:
                defer_ok += 1
        if v == "REVISE":
            revised += 1
    n_empty = int((sample.apply(lambda r: len(ra.retrieve_cohort(db, emb, ra.make_card(r),
                    stage=str(r.get('stage')))[1]) == 0, axis=1)).sum())
    print(f"\nTIER 4  End-to-end (sample of {len(sample)} patients):")
    print(f"  groundedness rate (no therapy outside evidence): {grounded_ok}/{len(sample)} "
          f"= {grounded_ok/len(sample)*100:.0f}%")
    print(f"  deferral correctness (empty-evidence cases naming no therapy): {defer_ok}/{n_empty}")
    print(f"  critic revision rate: {revised}/{len(sample)} = {revised/len(sample)*100:.0f}%")
    print(f"  mean latency/patient: {np.mean(latencies):.1f}s")

if __name__ == "__main__":
    print("[setup] loading components ...\n")
    C = a3.load_components()
    _, test = ra.split(C["df"])
    test = test.copy()
    test["p"] = C["art"]["model"].predict_proba(test[C["art"]["features"]])[:, 1]
    print("=" * 70)
    tier1(C, test)
    tier2(C, test)
    tier3(C, test)
    tier4(C, test)
    print("=" * 70)
    print("Evaluation complete.")
