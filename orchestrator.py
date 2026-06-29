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
# FAISS + Intel MKL can both load the OpenMP runtime on Windows and crash ("libiomp5md
# already initialized"). This env flag tells the runtime to tolerate the double-load.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings("ignore")          # silence sklearn/pandas deprecation noise for a clean demo
import sys
# Add THIS file's own folder to the import path so the sibling agent modules import correctly.
# Derived from __file__ at runtime -> portable (works on any clone) and exposes no hard-coded local path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from typing import TypedDict                # typed dict for the shared state object
from langgraph.graph import StateGraph, END # LangGraph: the graph builder and the terminal node marker

# ----- the four agents + shared helpers, each imported as its own module -----
import retrieval_agent as ra          # Agent 2 + features/label/split helpers (make_card, retrieve, build_patient_table)
import journey_builder as jb          # Agent 1 (build_timeline, summarize)
import agent3_timing as a3            # Agent 3 helpers (benefit_likelihood score, assess_uncertainty, load_components)
import agent3_tot as tot             # Agent 3 reasoning: Tree-of-Thoughts beam search (tot_decide, tot_synthesize)
import agent4_critic as a4            # Agent 4 (safety_review groundedness check, the mandatory CAVEAT text)

MAX_REVISIONS = 1                      # cap on the critic->timing Reflexion loop (at most one rewrite)

# ----- the shared "case" state passed between nodes -----
# Every node receives this dict and returns a partial dict; LangGraph merges the returns back in.
# total=False means none of the fields are required up front (they get filled in as the pipeline runs).
class CaseState(TypedDict, total=False):
    record_id: str        # the patient ID being processed (the only input we start with)
    header: dict          # static patient facts (stage, age, cancer type, decision point T) from Agent 1
    narrative: str        # Agent 1's plain-language timeline summary
    p: float              # Agent 3's calibrated benefit-likelihood (0-1) from the classifier
    timing: str           # the timing verdict (ACT NOW / MONITOR / INSUFFICIENT EVIDENCE)
    unc: str              # confidence level (High / Moderate / Low)
    plans_txt: str        # formatted "what worked" treatment plans from Agent 2
    has_evidence: bool    # True if Agent 2 found a usable cohort with regimens
    diag: dict            # Agent 2 retrieval diagnostics (match count, distances, abstain flag)
    top: object           # ranked "what worked" plans (pandas Series) for the ToT thought-generator
    draft: str            # Agent 3's drafted recommendation text
    critic_verdict: str   # Agent 4's verdict: "PASS" or "REVISE"
    critic_feedback: str  # Agent 4's verbal feedback when it asks for a revision
    revisions: int        # how many times the critic has sent the draft back so far
    final: str            # the final vetted recommendation + caveat (the pipeline's output)

def build_graph(C):
    # C is the bundle of pre-loaded heavy components (data, model, FAISS index, LLM) from load_components().
    # Unpack them once so the inner node functions can close over them (avoids reloading per call).
    df, dfs, art, db, emb, llm = C["df"], C["dfs"], C["art"], C["db"], C["emb"], C["llm"]

    def journey_node(state):                                   # Agent 1 — Journey-Builder
        rid = state["record_id"]                               # read the patient ID from the shared state
        # Build the leakage-safe timeline: events before T, the decision point itself, and (ignored) future events.
        header, before, decision, _ = jb.build_timeline(rid, dfs)
        narrative = jb.summarize(llm, header, before, decision) # LLM writes a 4-6 sentence journey summary
        print(f"  [journey ] built timeline ({len(before)} pre-T events), T=day {int(header['T'])}")  # progress log
        return {"header": header, "narrative": narrative}      # write these two fields back into the state

    def cohort_node(state):                                    # Agent 2 — Cohort-Retrieval
        row = df[df.record_id == state["record_id"]].iloc[0]   # grab this patient's as-of-T feature row
        card = ra.make_card(row)                               # turn the row into the structured text "card" to embed
        # Reflexive retrieval: embed the card, FAISS-search similar BENEFITED patients, relax filters if sparse.
        ben, top, diag, _ = ra.retrieve_cohort_reflexive(db, emb, card, str(row.get("stage")), llm, verbose=False)
        if len(top):                                           # if any regimens were mined from the cohort...
            # format each "what worked" regimen with how many matched patients used it
            plans = "\n".join(f"  - {d} (worked for {n} of {diag.get('n_used','?')})" for d, n in top.items())
        else:                                                  # otherwise the cohort had no systemic regimen
            plans = ("  (no specific systemic regimen among matched benefited patients; managed "
                     "conservatively, e.g., surgery/surveillance)")
        print(f"  [cohort  ] {diag.get('n_used',0)} matches, {diag.get('attempts',1)} attempt(s), "
              f"evidence={'yes' if len(top) else 'NONE'}")     # progress log: match count + attempts
        # write the formatted plans, the has-evidence flag, the diagnostics, and the raw ranked plans (top) into state
        return {"plans_txt": plans, "has_evidence": bool(len(top)), "diag": diag, "top": top}

    def timing_node(state):                                    # Agent 3 — Timing & Recommendation (Tree-of-Thoughts)
        row = df[df.record_id == state["record_id"]].iloc[0]   # this patient's feature row again
        p = a3.benefit_likelihood(art, row)                    # run the calibrated classifier -> probability p
        diag = state["diag"]                                   # reuse Agent 2's retrieval diagnostics
        # Assemble the context the ToT search needs: score, retrieval quality, ranked plans, raw features, evidence flag.
        ctx = {"p": p, "diag": diag, "top": state.get("top"), "row": row,
               "has_evidence": state.get("has_evidence", False)}
        decision = tot.tot_decide(ctx, verbose=False)          # ToT beam search picks the best action-strategy (deterministic)
        timing = ("INSUFFICIENT EVIDENCE — defer to clinician" if decision["abstain"]  # abstain -> defer
                  else decision["best"][0]["label"])           # otherwise use the chosen strategy's label as the verdict
        unc = a3.assess_uncertainty(p, diag)                   # derive confidence from score margin + match quality
        a = {**state, "p": p, "timing": timing, "unc": unc}    # merge current state with the new fields for synthesis
        # If this is a re-run triggered by the critic (revisions>0), pass the critic's feedback into the rewrite.
        feedback = state.get("critic_feedback", "") if state.get("revisions", 0) > 0 else ""
        draft = tot.tot_synthesize(llm, a, decision, feedback=feedback)  # LLM writes the recommendation text (grounded)
        tag = "revised" if feedback else "draft"               # label for the log (first pass vs. Reflexion rewrite)
        print(f"  [timing  ] benefit-likelihood {p:.0%} | ToT selected: {timing} ({tag})")  # progress log
        return {"p": p, "timing": timing, "unc": unc, "draft": draft}   # write score, verdict, confidence, draft into state

    def critic_node(state):                                    # Agent 4 — Safety / Critic
        # Deterministic groundedness check: returns PASS/REVISE + verbal feedback (no LLM call).
        v, fb, _ = a4.safety_review(llm, state, state["draft"])
        rev = state.get("revisions", 0) + (1 if v == "REVISE" else 0)  # bump the revision counter only on REVISE
        print(f"  [critic  ] verdict={v}" + (f"  feedback: {fb[:80]}..." if v == "REVISE" else ""))  # progress log
        return {"critic_verdict": v, "critic_feedback": fb, "revisions": rev}  # write verdict/feedback/count into state

    def route_after_critic(state):                             # conditional Reflexion edge (decides the next node)
        # If the critic wants a revision AND we haven't exceeded the cap, loop back to the timing node...
        if state["critic_verdict"] == "REVISE" and state.get("revisions", 0) <= MAX_REVISIONS:
            return "timing"
        return "finalize"                                      # ...otherwise proceed to finalize

    def finalize_node(state):                                  # terminal node: attach the mandatory caveat
        return {"final": state["draft"] + "\n\nMANDATORY CAVEAT: " + a4.CAVEAT}  # the pipeline's final output

    g = StateGraph(CaseState)                                  # create the graph, typed by our CaseState schema
    # register each node function under a name LangGraph will use for routing
    for name, fn in [("journey", journey_node), ("cohort", cohort_node), ("timing", timing_node),
                     ("critic", critic_node), ("finalize", finalize_node)]:
        g.add_node(name, fn)
    g.set_entry_point("journey")                               # the pipeline starts at Agent 1
    g.add_edge("journey", "cohort")                            # Agent 1 -> Agent 2
    g.add_edge("cohort", "timing")                             # Agent 2 -> Agent 3
    g.add_edge("timing", "critic")                             # Agent 3 -> Agent 4
    # after the critic, route dynamically: back to "timing" (REVISE) or on to "finalize" (PASS)
    g.add_conditional_edges("critic", route_after_critic, {"timing": "timing", "finalize": "finalize"})
    g.add_edge("finalize", END)                                # finalize -> END (stop)
    return g.compile()                                         # compile into a runnable app

def run_patient(record_id, C, app):
    print(f"\n================ RUNNING PIPELINE for {record_id} ================")  # header banner
    out = app.invoke({"record_id": record_id})                 # run the whole graph; seed state with just the patient ID
    h = out["header"]                                          # pull the static patient facts from the final state
    print(f"\nPATIENT {record_id} | {h.get('ca_type') or 'NSCLC'}, {h.get('stage_dx')}, age {h.get('age_dx')}")  # summary line
    print("\n===== FINAL VETTED RECOMMENDATION =====\n" + out["final"])  # print the final recommendation + caveat
    return out                                                 # return the full final state for any downstream use

if __name__ == "__main__":                                     # only runs when executed directly (not on import)
    print("[setup] loading components + compiling graph ...")
    C = a3.load_components()                                   # load data, the trained model, FAISS index, and the LLM
    app = build_graph(C)                                       # build + compile the LangGraph pipeline
    _, test = ra.split(C["df"])                                # recreate the same train/test split; keep the test set
    rid = test.iloc[0]["record_id"]                            # one held-out patient end-to-end
    run_patient(rid, C, app)                                   # run the full pipeline on that patient
