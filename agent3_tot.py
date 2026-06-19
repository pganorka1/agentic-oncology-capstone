"""
agent3_tot.py  —  Tree-of-Thoughts reasoning for Agent 3 (Timing & Recommendation)

Replaces Agent 3's single-pass recommendation with a BOUNDED BEAM SEARCH over
candidate action-strategies, so the agent explores alternatives instead of
committing prematurely.

  thought  = a candidate next-action strategy
  node     = a (partial) strategy + its score
  branch   = expanding a strategy class into concrete instantiations
  depth 1  = strategy CLASS (act / monitor / escalate / conservative)
  depth 2  = concrete INSTANTIATION (which regimen / interval / referral)
  search   = BEAM  (width W=2, depth D=2)  -> ~6 evaluations, predictable cost
  evaluator= DETERMINISTIC composite rubric (groundedness gate + alignment +
             evidence strength + constraints + safety); LLM only writes the final text
  selection= best grounded leaf; if its score < floor -> ABSTAIN (defer to clinician)

Run:  python agent3_tot.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, r"C:\Users\Prashant\claude-test\Capstone")
import retrieval_agent as ra
import agent3_timing as a3
import agent4_critic as a4

ACT, MON, ESC, CON = "ACT_NOW", "MONITOR", "ESCALATE", "CONSERVATIVE"
BEAM_W = 2
SCORE_FLOOR = 0.40           # leaves below this are not actionable -> abstain
def clamp(x, lo=0.0, hi=1.0): return max(lo, min(hi, x))

# ---------------------------------------------------------------- thought generation
def gen_roots():
    return [
        {"kind": ACT, "regimen": None, "label": "Act now: initiate/adjust systemic therapy"},
        {"kind": MON, "label": "Monitor: short-interval reassessment, defer change"},
        {"kind": ESC, "label": "Escalate: molecular tumor board / specialist review"},
        {"kind": CON, "label": "Conservative: surgery / surveillance, no systemic change"},
    ]

def expand(node, ctx):
    k = node["kind"]
    if k == ACT:
        regs = list(ctx["top"].index[:2])
        if regs:
            return [{"kind": ACT, "regimen": r, "label": f"Act now with {r}"} for r in regs]
        return [{"kind": ACT, "regimen": "__none__", "label": "Act now (no supported regimen)"}]
    if k == MON:
        return [{"kind": MON, "label": "Re-image in ~8 weeks, then reassess"},
                {"kind": MON, "label": "Re-assess in ~12 weeks"}]
    if k == ESC:
        return [{"kind": ESC, "label": "Refer to molecular tumor board"}]
    return [{"kind": CON, "label": "Surgery / surveillance, no systemic change"}]

# ---------------------------------------------------------------- evaluator (rubric)
def score(node, ctx):
    p, diag, top, row = ctx["p"], ctx["diag"], ctx["top"], ctx["row"]
    has_ev = ctx["has_evidence"]
    n_used = diag.get("n_used", 0); mean_dist = diag.get("mean_dist") or 0.05
    weak_ev = diag.get("abstain") or n_used < ra.MIN_MATCHES
    churn = float(row.get("regimens_before_T", 0) or 0)
    stage = str(row.get("stage") or "")
    age = float(row.get("age_dx") or 65)
    k = node["kind"]; reg = node.get("regimen")

    # (1) groundedness GATE
    if k == ACT:
        grounded = has_ev if reg is None else (reg in set(top.index))
    else:
        grounded = True                      # non-pharmacologic strategies name no therapy
    g = 1.0 if grounded else 0.0

    # (2) benefit alignment
    align = p if k == ACT else (1 - p) if k in (MON, CON) else 0.5

    # (3) evidence strength
    if k == ACT:
        e = 0.0 if weak_ev else clamp(n_used / 20.0) * clamp(1 - mean_dist / 0.05)
    else:
        e = 0.6                              # not regimen-dependent

    # (4) constraint satisfaction
    c = 0.5
    if k == ACT:
        if churn >= 3: c -= 0.2              # high churn = guesswork -> caution acting
        if stage.endswith("IV"): c += 0.2
        if age >= 80: c -= 0.2
    if k == ESC and weak_ev: c += 0.3        # escalate when uncertain
    if k == CON and stage in ("Stage I", "Stage II"): c += 0.2
    if k == MON and p < 0.4: c += 0.2
    c = clamp(c)

    # (5) safety
    s = 0.3 if (k == ACT and weak_ev) else 1.0

    composite = g * (0.30 * align + 0.30 * e + 0.25 * c + 0.15 * s)
    return composite, dict(g=g, align=round(align, 2), e=round(e, 2), c=round(c, 2), s=s)

# ---------------------------------------------------------------- beam controller
def tot_decide(ctx, verbose=True):
    trace = {}
    roots = [(n, *score(n, ctx)) for n in gen_roots()]
    roots.sort(key=lambda x: -x[1])
    trace["depth1"] = roots
    beam = [r[0] for r in roots[:BEAM_W]]                     # prune to top-W classes

    leaves = []
    for node in beam:
        for child in expand(node, ctx):
            comp, comps = score(child, ctx)
            leaves.append((child, comp, comps))
    leaves.sort(key=lambda x: -x[1])
    trace["depth2"] = leaves
    kept = leaves[:BEAM_W]
    best = kept[0] if kept else None
    abstain = (best is None) or (best[1] < SCORE_FLOOR)

    if verbose:
        print("  -- ToT beam search (W=2, D=2) --")
        print("  depth 1 (strategy classes):")
        for n, comp, cm in roots:
            mark = "KEEP" if n in beam else "prune"
            print(f"     [{mark:>5}] {comp:.2f}  {n['label']}   {cm}")
        print("  depth 2 (instantiations of kept classes):")
        for n, comp, cm in leaves:
            mark = "BEST" if (best and n is best[0]) else ("keep" if (n, comp, cm) in kept else "prune")
            print(f"     [{mark:>5}] {comp:.2f}  {n['label']}")
        print(f"  -> selected: {'ABSTAIN/defer' if abstain else best[0]['label']} "
              f"(score {best[1]:.2f})" if best else "  -> no candidates")
    return {"best": best, "abstain": abstain, "trace": trace}

# ---------------------------------------------------------------- final text (1 LLM call)
def tot_synthesize(llm, a, decision, feedback=""):
    if decision["abstain"]:
        chosen = "Defer to the clinician — no strategy met the evidence/score threshold."
        rule = "Do NOT name any specific therapy."
    else:
        chosen = decision["best"][0]["label"]
        rule = ("Name a specific regimen ONLY if the chosen strategy is an 'Act now with <regimen>' that "
                "appears in the EVIDENCE; otherwise name no drug.")
    fb = f"\n\nSAFETY-REVIEWER FEEDBACK you MUST address: {feedback}" if feedback else ""
    return llm.invoke(
        "You are the Timing & Recommendation agent (decision-support, not diagnosis). A Tree-of-Thoughts "
        "reasoning step evaluated several strategies and SELECTED the one below. Write the recommendation "
        "with labelled parts: TIMING VERDICT, BENEFIT-LIKELIHOOD, SUGGESTED NEXT ACTIONS, CONFIDENCE, "
        f"CAVEAT. {rule} Do not invent therapies or numbers." + fb + "\n\n"
        f"SELECTED STRATEGY: {chosen}\n"
        f"PATIENT: {a['narrative']}\n"
        f"BENEFIT-LIKELIHOOD: {a['p']:.0%} | CONFIDENCE: {a['unc']}\n"
        f"EVIDENCE (treatment plans that worked for similar benefited patients):\n{a['plans_txt']}"
    ).content

def narrative_lite(r):
    return (f"{r.get('stage')} {r.get('oncotree') or 'NSCLC'}, age {r.get('age_dx')}; genomic test day "
            f"{int(r['T'])}; status before test: {r.get('pre_status')}; "
            f"{int(r.get('regimens_before_T',0) or 0)} prior regimen(s).")

def run(rid, C):
    df, art, db, emb, llm = C["df"], C["art"], C["db"], C["emb"], C["llm"]
    row = df[df.record_id == rid].iloc[0]
    p = a3.benefit_likelihood(art, row)
    card = ra.make_card(row)
    _, top, diag = ra.retrieve_cohort(db, emb, card, stage=str(row.get("stage")))
    has_ev = len(top) > 0 and not diag.get("abstain")
    plans = ("\n".join(f"  - {d} (worked for {n})" for d, n in top.items()) if has_ev
             else "  (no specific systemic regimen among matched benefited patients)")
    ctx = {"p": p, "diag": diag, "top": top, "row": row, "has_evidence": has_ev}
    a = {"narrative": narrative_lite(row), "p": p, "unc": a3.assess_uncertainty(p, diag),
         "plans_txt": plans, "has_evidence": has_ev}

    print("\n" + "=" * 84)
    print(f"PATIENT {rid} | {row.get('stage')}, {row.get('oncotree') or 'NSCLC'}, age {row.get('age_dx')}"
          f" | benefit-likelihood {p:.0%} | evidence: {'yes' if has_ev else 'NONE'}")
    decision = tot_decide(ctx)
    draft = tot_synthesize(llm, a, decision)
    # Agent 4 safety review (unchanged, deterministic)
    v, fb, _ = a4.safety_review(llm, a, draft)
    if v == "REVISE":
        draft = tot_synthesize(llm, a, decision, feedback=fb)
    print("\n--- FINAL VETTED RECOMMENDATION ---\n" + draft + "\n\nMANDATORY CAVEAT: " + a4.CAVEAT)

if __name__ == "__main__":
    print("[setup] loading components ...")
    C = a3.load_components()
    df = C["df"]; _, test = ra.split(df)
    test = test.copy()
    test["p"] = C["art"]["model"].predict_proba(test[C["art"]["features"]])[:, 1]

    # Case A: find a high-likelihood patient WITH cohort evidence (exercises the ACT_NOW branch)
    caseA = None
    for _, r in test.sort_values("p", ascending=False).head(40).iterrows():
        _, top, diag = ra.retrieve_cohort(C["db"], C["emb"], ra.make_card(r), stage=str(r.get("stage")))
        if len(top) and not diag.get("abstain"):
            caseA = r["record_id"]; break
    caseB = test.sort_values("p", ascending=True).iloc[0]["record_id"]   # low-likelihood / likely defer

    if caseA:
        print("\n############## CASE A: high benefit-likelihood + evidence ##############")
        run(caseA, C)
    print("\n############## CASE B: low benefit-likelihood ##############")
    run(caseB, C)
