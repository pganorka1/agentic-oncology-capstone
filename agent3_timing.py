"""
agent3_timing.py  —  Agent 3: Timing & Recommendation

Converges the three upstream components for one patient at decision point T:
  1. Journey-Builder (Agent 1)   -> leakage-safe timeline + narrative
  2. Benefit-risk classifier      -> calibrated benefit-likelihood score
  3. Cohort-Retrieval (Agent 2)   -> "what worked" for similar benefited patients

Then it decides "is NOW the optimal juncture?" and drafts an evidence-grounded
recommendation with an uncertainty estimate. Decision-support, clinician decides.

Run:  python agent3_timing.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings, joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, r"C:\Users\Prashant\claude-test\Capstone")
import retrieval_agent as ra
import journey_builder as jb
import build_classifier as bc
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

ACT_THRESHOLD = 0.50          # benefit-likelihood needed to call "now" a good juncture
MODEL_PATH = r"C:\Users\Prashant\claude-test\Capstone\agent_artifacts\benefit_risk_model.joblib"

def load_components():
    art = joblib.load(MODEL_PATH)
    emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=ra.API_KEY)
    db = FAISS.load_local(ra.INDEX_DIR, emb, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=ra.API_KEY)
    dfs = jb.load_all()
    df = ra.build_patient_table()
    return dict(art=art, emb=emb, db=db, llm=llm, dfs=dfs, df=df)

def benefit_likelihood(art, row):
    X = pd.DataFrame([row[art["features"]]])
    return float(art["model"].predict_proba(X)[0, 1])

def assess_uncertainty(p, diag):
    """Confidence from (a) how decisive the score is and (b) cohort match quality.
    NOTE: treatment churn is a post-T LABEL component (future) -> NOT a runtime input."""
    if diag.get("abstain"):
        return "Low — insufficient similar-patient evidence"
    decisive = abs(p - 0.5) > 0.15
    strong_evidence = diag.get("n_used", 0) >= 10 and (diag.get("mean_dist") or 1) < 0.035
    if decisive and strong_evidence:
        return "High"
    if (not decisive) or diag.get("n_used", 0) < ra.MIN_MATCHES:
        return "Low"
    return "Moderate"

def assemble(record_id, C):
    """Gather timeline + score + cohort evidence into one case object (no LLM synthesis yet)."""
    df, dfs, art, db, emb, llm = C["df"], C["dfs"], C["art"], C["db"], C["emb"], C["llm"]
    row = df[df.record_id == record_id].iloc[0]
    header, before, decision, after = jb.build_timeline(record_id, dfs)
    narrative = jb.summarize(llm, header, before, decision)
    p = benefit_likelihood(art, row)
    card = ra.make_card(row)
    ben, top, diag, verdict = ra.retrieve_cohort_reflexive(db, emb, card,
                                                           str(row.get("stage")), llm, verbose=False)
    if diag.get("abstain"):
        timing = "INSUFFICIENT EVIDENCE — defer to clinician"
    elif p >= ACT_THRESHOLD:
        timing = "ACT NOW — this juncture resembles patients who benefited"
    else:
        timing = "NOT YET / MONITOR — benefit-likelihood below threshold"
    unc = assess_uncertainty(p, diag)
    if len(top):
        plans_txt = "\n".join(f"  - {d} (worked for {n} of {diag.get('n_used','?')})" for d, n in top.items())
    else:
        plans_txt = ("  (no specific systemic regimen among the matched benefited patients; they were "
                     "largely managed conservatively, e.g., surgery/surveillance)")
    return dict(record_id=record_id, header=header, narrative=narrative, p=p, timing=timing,
                unc=unc, plans_txt=plans_txt, has_evidence=bool(len(top)), top=top, diag=diag)

def synthesize(llm, a, feedback=""):
    """Draft the recommendation. `feedback` (from the Safety/Critic) drives Reflexion revision."""
    fb = f"\n\nSAFETY-REVIEWER FEEDBACK you MUST address: {feedback}" if feedback else ""
    return llm.invoke(
        "You are the Timing & Recommendation agent of a clinical decision-support system "
        "(decision-support, NOT diagnosis; the oncologist decides). Write a concise recommendation with "
        "labelled parts: TIMING VERDICT, BENEFIT-LIKELIHOOD, SUGGESTED NEXT ACTIONS, CONFIDENCE, CAVEAT. "
        "Name ONLY treatments that appear in the EVIDENCE below. If the evidence lists NO specific regimen, "
        "DO NOT name any drug or therapy class — instead recommend a genomic-guided discussion and note the "
        "evidence is limited. Do not invent therapies or numbers." + fb + "\n\n"
        f"PATIENT JOURNEY (up to decision point):\n{a['narrative']}\n\n"
        f"MODEL BENEFIT-LIKELIHOOD: {a['p']:.0%}  | TIMING SIGNAL: {a['timing']}  | CONFIDENCE: {a['unc']}\n"
        f"EVIDENCE — treatment plans that worked for similar benefited patients:\n{a['plans_txt']}"
    ).content

def run(record_id, C, show_journey=False):
    a = assemble(record_id, C)
    rec = synthesize(C["llm"], a)
    print("\n" + "=" * 80)
    print(f"PATIENT {record_id} | {a['header'].get('ca_type') or 'NSCLC'}, {a['header'].get('stage_dx')}, "
          f"age {a['header'].get('age_dx')} | decision point T = day {int(a['header']['T'])}")
    if show_journey:
        print("\n-- Journey narrative --\n" + a["narrative"])
    print(f"\n  Benefit-likelihood (calibrated): {a['p']:.0%}")
    print(f"  Timing verdict: {a['timing']}")
    print(f"  Confidence: {a['unc']}  | cohort matches: {a['diag'].get('n_used', 0)} "
          f"(attempts: {a['diag'].get('attempts', 1)}, mean_dist: {a['diag'].get('mean_dist')})")
    print(f"  Evidence — what worked:\n{a['plans_txt']}")
    print("\n-- Agent 3 recommendation --\n" + rec)

if __name__ == "__main__":
    print("[setup] loading model, FAISS index, data, LLM ...")
    C = load_components()
    df = C["df"]
    _, test = ra.split(df)

    # score all test patients; demonstrate a high-likelihood ('act now') and a low ('monitor') case
    test = test.copy()
    test["p"] = [benefit_likelihood(C["art"], r) for _, r in test.iterrows()]
    hi = test.sort_values("p", ascending=False).iloc[0]
    lo = test.sort_values("p", ascending=True).iloc[0]

    print("\n############## CASE A: highest benefit-likelihood (expect ACT NOW) ##############")
    run(hi["record_id"], C, show_journey=True)
    print("\n############## CASE B: lowest benefit-likelihood (expect MONITOR) ##############")
    run(lo["record_id"], C)
