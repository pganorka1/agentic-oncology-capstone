"""
demo.py  —  narrated end-to-end demo for the recorded presentation

Runs the full 4-agent pipeline on two contrasting held-out patients, printing a
clean, narrated walkthrough of each agent. Designed to be screen-recorded.

Run all:                 python demo.py
Run one chosen patient:  python demo.py GENIE-MSK-P-0006640
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))   # repo folder, resolved at runtime (portable; no hard-coded path)
sys.path.insert(0, HERE)
import retrieval_agent as ra
import journey_builder as jb
import agent3_timing as a3
import agent4_critic as a4

def banner(t):  print("\n" + "=" * 80 + "\n  " + t + "\n" + "=" * 80)
def step(t):    print("\n" + "-" * 80 + "\n  " + t + "\n" + "-" * 80)

def intro():
    banner("AGENTIC CLINICAL DECISION-SUPPORT — for genomic-test timing (NSCLC)")
    print(
        "  PROBLEM : a patient's cancer journey is scattered across 9 data sources. We want to flag,\n"
        "            proactively, the earliest moment a patient resembles prior patients who BENEFITED\n"
        "            from genomic testing — and surface the treatment plans that worked — for the\n"
        "            oncologist to consider. Decision-support, NOT diagnosis.\n"
        "  SYSTEM  : 4 cooperating agents, orchestrated end-to-end:\n"
        "            1) Journey-Builder   - assembles a leakage-safe timeline up to the decision point\n"
        "            2) Cohort-Retrieval  - finds similar patients who benefited (RAG + MMR + Reflexion)\n"
        "            3) Timing & Recommend- calibrated benefit-risk model -> 'act now?' + next actions\n"
        "            4) Safety / Critic   - blocks any therapy not backed by evidence; adds caveat\n"
        "  LLM     : ChatGPT (OpenAI).  DATA: AACR GENIE-BPC NSCLC (~1,846 patients).")

def narrated_run(rid, C, case_label=""):
    df, dfs, art, db, emb, llm = C["df"], C["dfs"], C["art"], C["db"], C["emb"], C["llm"]
    rid = str(rid).strip()                                     # tolerate stray whitespace from copy-paste
    match = df[df.record_id == rid]                            # look up the patient by full record_id
    if match.empty:                                            # not found -> friendly message instead of an IndexError
        print(f"\n  [error] record_id '{rid}' not found.")
        print("  Pass the FULL id (institution prefix + number), e.g.:  python demo.py GENIE-MSK-P-0016319")
        print("  Valid examples from the data:", ", ".join(df["record_id"].head(5).tolist()))
        return
    row = match.iloc[0]
    banner(f"{case_label}PATIENT {rid}  |  {row.get('stage')}, {row.get('oncotree') or 'NSCLC'}, age {row.get('age_dx')}")

    # ---- Agent 1 ----
    step("AGENT 1 — Journey-Builder: assemble timeline + enforce leakage firewall")
    header, before, decision, after = jb.build_timeline(rid, dfs)
    print(f"  decision point T = day {int(header['T'])} (first genomic test). Recent events before T:")
    for e in before[-8:]:
        print(f"    day {e['day']:>5}  [{e['category']:<10}] {e['description']}")
    print(f"    day {decision['day']:>5}  [{decision['category']}] {decision['description']}  <== NOW")
    print(f"  FIREWALL: {len(before)} events kept (day < T); {after} future events EXCLUDED (no leakage).")
    narrative = jb.summarize(llm, header, before, decision)
    print("\n  Journey narrative:\n  " + narrative.replace("\n", "\n  "))

    # ---- Agent 2 ----
    step("AGENT 2 — Cohort-Retrieval: similar patients who benefited (Reflexion if evidence weak)")
    card = ra.make_card(row)
    ben, top, diag, verdict = ra.retrieve_cohort_reflexive(db, emb, card, str(row.get("stage")), llm, verbose=True)
    if diag.get("abstain"):
        plans_txt = "  (insufficient similar evidence — agent abstains)"
        has_ev = False
    elif len(top):
        plans_txt = "\n".join(f"  - {d} (worked for {n} of {diag['n_used']})" for d, n in top.items())
        has_ev = True
    else:
        plans_txt = "  (matched benefited patients managed conservatively; no systemic regimen)"
        has_ev = False
    print(f"  match quality: {diag.get('n_used', 0)} benefited patients, "
          f"{diag.get('attempts', 1)} attempt(s), mean_dist={diag.get('mean_dist')}")
    print("  TREATMENT PLANS THAT WORKED for similar patients:\n" + plans_txt)

    # ---- Agent 3 ----
    step("AGENT 3 — Timing & Recommendation: calibrated benefit-likelihood -> decision")
    p = a3.benefit_likelihood(art, row)
    timing = ("INSUFFICIENT EVIDENCE — defer to clinician" if diag.get("abstain")
              else "ACT NOW — resembles patients who benefited" if p >= a3.ACT_THRESHOLD
              else "NOT YET / MONITOR — below threshold")
    unc = a3.assess_uncertainty(p, diag)
    a = {"narrative": narrative, "p": p, "timing": timing, "unc": unc,
         "plans_txt": plans_txt, "has_evidence": has_ev}
    print(f"  benefit-likelihood (calibrated): {p:.0%}   ->   {timing}   (confidence: {unc})")
    draft = a3.synthesize(llm, a)
    print("\n  DRAFT recommendation:\n  " + draft.replace("\n", "\n  "))

    # ---- Agent 4 ----
    step("AGENT 4 — Safety / Critic: groundedness check + (Reflexion) revision + mandatory caveat")
    v, fb, detail = a4.safety_review(llm, a, draft)
    print(f"  groundedness verdict: {v}  ({detail})")
    if v == "REVISE":
        print(f"  feedback -> Agent 3: {fb}")
        draft = a3.synthesize(llm, a, feedback=fb)
        v2, _, _ = a4.safety_review(llm, a, draft)
        print(f"  re-check after revision: {v2}")
    final = draft + "\n\nMANDATORY CAVEAT: " + a4.CAVEAT
    banner("FINAL VETTED RECOMMENDATION (to the oncologist)")
    print("  " + final.replace("\n", "\n  "))

def closing():
    banner("VALIDATION (held-out test set) — see evaluate.py")
    print("  Tier 1 Classifier : ROC-AUC 0.78, well-calibrated (Brier 0.16)\n"
          "  Tier 2 Retrieval  : the ACTUAL effective regimen is surfaced 78% of the time (hit-rate@20)\n"
          "  Tier 3 Proactivity: decision point precedes progression by a median ~110 days\n"
          "  Tier 4 End-to-end : 100% groundedness (no hallucinated therapy); ~2s/patient\n"
          "  => proactive, evidence-grounded, clinician-supervised decision support.")

if __name__ == "__main__":
    print("[setup] loading model, FAISS index, data, LLM ...")
    C = a3.load_components()
    df = C["df"]; _, test = ra.split(df)
    intro()
    if len(sys.argv) > 1:                                   # presenter-chosen patient
        narrated_run(sys.argv[1], C)
    else:
        test = test.copy()
        test["p"] = C["art"]["model"].predict_proba(test[C["art"]["features"]])[:, 1]
        treated = test[test["regimens_before_T"] >= 1]
        caseA = (treated if len(treated) else test).sort_values("p", ascending=False).iloc[0]["record_id"]
        caseB = test.sort_values("p", ascending=True).iloc[0]["record_id"]
        narrated_run(caseA, C, case_label="CASE A (high benefit-likelihood)  ")
        narrated_run(caseB, C, case_label="CASE B (low / insufficient)  ")
    closing()
