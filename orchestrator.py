"""
orchestrator.py  —  the full agentic pipeline (LangGraph)

Wires the four agents into ONE system for a single patient:

    journey  ->  cohort  ->  timing  ->  critic  --(PASS)-->  finalize -> END
                                ^                   |
                                |---- REVISE -------|   (capped Reflexion loop)

Each node calls the agent code we already built. Run end-to-end with:
    python orchestrator.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, r"C:\Users\Prashant\claude-test\Capstone")
from typing import TypedDict
from langgraph.graph import StateGraph, END

import retrieval_agent as ra          # Agent 2 + features/label/split
import journey_builder as jb          # Agent 1
import agent3_timing as a3            # Agent 3 (score/uncertainty)
import agent3_tot as tot             # Agent 3 reasoning: Tree-of-Thoughts beam search
import agent4_critic as a4            # Agent 4 (safety_review, CAVEAT)

MAX_REVISIONS = 1                      # cap on the critic->timing Reflexion loop

# ----- the shared "case" state passed between nodes -----
class CaseState(TypedDict, total=False):
    record_id: str
    header: dict
    narrative: str
    p: float
    timing: str
    unc: str
    plans_txt: str
    has_evidence: bool
    diag: dict
    top: object          # ranked "what worked" plans (pandas Series) for the ToT thought-generator
    draft: str
    critic_verdict: str
    critic_feedback: str
    revisions: int
    final: str

def build_graph(C):
    df, dfs, art, db, emb, llm = C["df"], C["dfs"], C["art"], C["db"], C["emb"], C["llm"]

    def journey_node(state):                                   # Agent 1
        rid = state["record_id"]
        header, before, decision, _ = jb.build_timeline(rid, dfs)
        narrative = jb.summarize(llm, header, before, decision)
        print(f"  [journey ] built timeline ({len(before)} pre-T events), T=day {int(header['T'])}")
        return {"header": header, "narrative": narrative}

    def cohort_node(state):                                    # Agent 2
        row = df[df.record_id == state["record_id"]].iloc[0]
        card = ra.make_card(row)
        ben, top, diag, _ = ra.retrieve_cohort_reflexive(db, emb, card, str(row.get("stage")), llm, verbose=False)
        if len(top):
            plans = "\n".join(f"  - {d} (worked for {n} of {diag.get('n_used','?')})" for d, n in top.items())
        else:
            plans = ("  (no specific systemic regimen among matched benefited patients; managed "
                     "conservatively, e.g., surgery/surveillance)")
        print(f"  [cohort  ] {diag.get('n_used',0)} matches, {diag.get('attempts',1)} attempt(s), "
              f"evidence={'yes' if len(top) else 'NONE'}")
        return {"plans_txt": plans, "has_evidence": bool(len(top)), "diag": diag, "top": top}

    def timing_node(state):                                    # Agent 3 (Tree-of-Thoughts)
        row = df[df.record_id == state["record_id"]].iloc[0]
        p = a3.benefit_likelihood(art, row)
        diag = state["diag"]
        # ToT beam search over candidate action-strategies (deterministic; no extra LLM calls)
        ctx = {"p": p, "diag": diag, "top": state.get("top"), "row": row,
               "has_evidence": state.get("has_evidence", False)}
        decision = tot.tot_decide(ctx, verbose=False)
        timing = ("INSUFFICIENT EVIDENCE — defer to clinician" if decision["abstain"]
                  else decision["best"][0]["label"])
        unc = a3.assess_uncertainty(p, diag)
        a = {**state, "p": p, "timing": timing, "unc": unc}
        feedback = state.get("critic_feedback", "") if state.get("revisions", 0) > 0 else ""
        draft = tot.tot_synthesize(llm, a, decision, feedback=feedback)
        tag = "revised" if feedback else "draft"
        print(f"  [timing  ] benefit-likelihood {p:.0%} | ToT selected: {timing} ({tag})")
        return {"p": p, "timing": timing, "unc": unc, "draft": draft}

    def critic_node(state):                                    # Agent 4
        v, fb, _ = a4.safety_review(llm, state, state["draft"])
        rev = state.get("revisions", 0) + (1 if v == "REVISE" else 0)
        print(f"  [critic  ] verdict={v}" + (f"  feedback: {fb[:80]}..." if v == "REVISE" else ""))
        return {"critic_verdict": v, "critic_feedback": fb, "revisions": rev}

    def route_after_critic(state):                             # conditional Reflexion edge
        if state["critic_verdict"] == "REVISE" and state.get("revisions", 0) <= MAX_REVISIONS:
            return "timing"
        return "finalize"

    def finalize_node(state):
        return {"final": state["draft"] + "\n\nMANDATORY CAVEAT: " + a4.CAVEAT}

    g = StateGraph(CaseState)
    for name, fn in [("journey", journey_node), ("cohort", cohort_node), ("timing", timing_node),
                     ("critic", critic_node), ("finalize", finalize_node)]:
        g.add_node(name, fn)
    g.set_entry_point("journey")
    g.add_edge("journey", "cohort")
    g.add_edge("cohort", "timing")
    g.add_edge("timing", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"timing": "timing", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()

def run_patient(record_id, C, app):
    print(f"\n================ RUNNING PIPELINE for {record_id} ================")
    out = app.invoke({"record_id": record_id})
    h = out["header"]
    print(f"\nPATIENT {record_id} | {h.get('ca_type') or 'NSCLC'}, {h.get('stage_dx')}, age {h.get('age_dx')}")
    print("\n===== FINAL VETTED RECOMMENDATION =====\n" + out["final"])
    return out

if __name__ == "__main__":
    print("[setup] loading components + compiling graph ...")
    C = a3.load_components()
    app = build_graph(C)
    _, test = ra.split(C["df"])
    rid = test.iloc[0]["record_id"]                            # one held-out patient end-to-end
    run_patient(rid, C, app)
