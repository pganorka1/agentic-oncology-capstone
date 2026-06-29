"""
agent4_critic.py  —  Agent 4: Safety / Critic

An INDEPENDENT reviewer of Agent 3's draft recommendation. It:
  - checks groundedness: every therapy named must appear in the cohort EVIDENCE;
  - enforces deferral: if evidence is empty / confidence Low, the recommendation
    must NOT name specific therapies (catches ungrounded generic advice);
  - runs a capped REFLEXION revision: on REVISE, it returns verbal feedback to
    Agent 3 (synthesize) for one revision (Actor <-> Evaluator loop);
  - appends a mandatory association-not-causation + clinician-decides caveat.

Run:  python agent4_critic.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re, warnings
warnings.filterwarnings("ignore")
import sys
HERE = os.path.dirname(os.path.abspath(__file__))   # repo folder, resolved at runtime (portable; no hard-coded path)
sys.path.insert(0, HERE)
import retrieval_agent as ra
import agent3_timing as a3

CAVEAT = ("Association, not causation — this is based on outcomes of similar prior patients, not a "
          "controlled trial. Decision-support only; the oncologist makes the final decision.")

def _drug_tokens(s):
    if not isinstance(s, str):
        return set()
    return {p.strip() for p in s.replace(";", ",").split(",") if p.strip()}

_DRUG_VOCAB = None
def drug_vocab():
    """Set of real drug/regimen names (index-cancer regimens), built once and cached."""
    global _DRUG_VOCAB
    if _DRUG_VOCAB is None:
        reg = ra.L("regimen_cancer_level_dataset.csv")
        reg = reg[reg["redcap_ca_index"] == "Yes"]
        v = set()
        for s in reg["regimen_drugs"].dropna():
            v |= _drug_tokens(s)
        _DRUG_VOCAB = {d for d in v if len(d) >= 4}   # drop tiny tokens that could false-match
    return _DRUG_VOCAB

# generic therapy-class phrases that count as "naming a specific therapy" only when there is NO evidence
CLASS_WORDS = ["chemotherapy", "immunotherapy", "targeted therapy", "radiation therapy", "radiotherapy"]

def safety_review(llm, a, rec):
    """Independent, DETERMINISTIC groundedness check. REVISE only on real violations:
       (1) a therapy named that is NOT in the cohort evidence, or
       (2) any specific drug/therapy-class named when the evidence is empty.
       Cautious deferrals ('genomic-guided discussion') correctly PASS.
       Returns (verdict 'PASS'/'REVISE', feedback, detail)."""
    low = rec.lower()
    vocab = drug_vocab()
    named = {d for d in vocab if d.lower() in low}                      # specific drugs mentioned
    # allowed = clean drug names from the evidence (strip list bullets and "(worked for N)" suffixes)
    allowed = set()
    if a.get("has_evidence"):
        for line in str(a.get("plans_txt", "")).splitlines():
            t = line.strip().lstrip("-").strip().split(" (")[0].strip()
            allowed |= _drug_tokens(t)
    issues = []

    ungrounded = named - allowed
    if ungrounded:
        issues.append(f"names therapies not supported by the evidence: {sorted(ungrounded)}")

    if not a.get("has_evidence"):
        classes = [w for w in CLASS_WORDS if w in low]
        if named:
            issues.append(f"names specific drugs despite no supporting evidence: {sorted(named)}")
        if classes:
            issues.append(f"names therapy classes despite no supporting evidence: {classes}")

    if issues:
        feedback = ("Remove unsupported therapies and instead defer to a genomic-guided discussion "
                    "without naming specific treatments. Issues: " + "; ".join(issues))
        return "REVISE", feedback, "; ".join(issues)
    return "PASS", "", "no groundedness violations"

def vetted(record_id, C, verbose=True):
    llm = C["llm"]
    a = a3.assemble(record_id, C)
    draft = a3.synthesize(llm, a)
    v1, fb1, review1 = safety_review(llm, a, draft)

    revised = None
    if v1 == "REVISE":
        revised = a3.synthesize(llm, a, feedback=fb1)        # Reflexion: Agent 3 revises with critic feedback
        v2, fb2, review2 = safety_review(llm, a, revised)
    final_rec = (revised if revised is not None else draft) + "\n\nMANDATORY CAVEAT: " + CAVEAT
    final_verdict = (v2 if revised is not None else v1)

    if verbose:
        h = a["header"]
        print("\n" + "=" * 84)
        print(f"PATIENT {record_id} | {h.get('ca_type') or 'NSCLC'}, {h.get('stage_dx')}, age {h.get('age_dx')}"
              f" | benefit-likelihood {a['p']:.0%} | {a['unc']} | evidence: "
              f"{'present' if a['has_evidence'] else 'NONE'}")
        print("\n--- Agent 3 DRAFT ---\n" + draft)
        print(f"\n--- Agent 4 SAFETY REVIEW ---\nVERDICT: {v1}")
        if v1 == "REVISE":
            print(f"FEEDBACK to Agent 3: {fb1}")
            print("\n--- Agent 3 REVISED (after reflexion) ---\n" + revised)
            print(f"\n--- Agent 4 re-review verdict: {final_verdict} ---")
        print("\n===== FINAL VETTED RECOMMENDATION =====\n" + final_rec)
    return final_rec, final_verdict

if __name__ == "__main__":
    print("[setup] loading components ...")
    C = a3.load_components()
    _, test = ra.split(C["df"])
    test = test.copy()
    test["p"] = [a3.benefit_likelihood(C["art"], r) for _, r in test.iterrows()]
    hi = test.sort_values("p", ascending=False).iloc[0]   # high-likelihood (was: ungrounded advice on empty evidence)
    lo = test.sort_values("p", ascending=True).iloc[0]     # low-likelihood / abstain case

    print("\n############## CASE A: high benefit-likelihood ##############")
    vetted(hi["record_id"], C)
    print("\n############## CASE B: low benefit-likelihood / abstain ##############")
    vetted(lo["record_id"], C)
