"""
retrieval_agent.py  —  Checkpoint 3.1 "working agent update"

Demonstrates that semantic retrieval MEANINGFULLY changes the agent's output.

Pipeline:
  1. Build leakage-safe patient "cards" at decision point T (first genomic test).
  2. Compute the `benefited` label (B: durable favorable outcome AND C: low churn).
  3. Train/test split (patient-level, stratified). Index is built from TRAIN only.
  4. Embed TRAIN cards -> FAISS (OpenAI embeddings).
  5. Cohort-Retrieval: for a test patient, find similar TRAIN patients, keep the
     ones who benefited, and surface "the treatment plans that worked".
  6. Show prompt-only vs retrieval-grounded LLM output (observable difference).

Run:  python retrieval_agent.py
"""
import os, re
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid Windows OpenMP (libiomp5md) double-load crash with FAISS+MKL
import numpy as np, pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv(r"C:\Users\Prashant\claude-test\.env")
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

DATA = r"C:\Users\Prashant\claude-test\Capstone\data\PJ_Data"
INDEX_DIR = r"C:\Users\Prashant\claude-test\Capstone\agent_artifacts\faiss_cohort"
WIN = 140          # benefit observation window (days after T)
K_RAW = 60         # raw neighbors to pull before filtering
K_KEEP = 20        # similar patients to keep
SEED = 42
DIST_MAX = 0.045   # FAISS L2 distance cutoff (lower=more similar); drop weaker matches
MIN_MATCHES = 5    # abstain if fewer than this many qualifying benefited matches
MMR_LAMBDA = 0.5   # MMR: 1.0=pure relevance, 0.0=max diversity

def L(f): return pd.read_csv(os.path.join(DATA, f), low_memory=False)

# ----------------------------------------------------------------------------
# 1-2. Build per-patient features at T, the card text, and the benefited label
# ----------------------------------------------------------------------------
def build_patient_table():
    cpt = L("cancer_panel_test_level_dataset.csv")
    T = cpt.groupby("record_id")["dx_cpt_rep_days"].min().rename("T")
    sample_type = (cpt.sort_values("dx_cpt_rep_days").groupby("record_id")["sample_type"].first())
    oncotree = (cpt.sort_values("dx_cpt_rep_days").groupby("record_id")["cpt_oncotree_code"].first()
                .rename("oncotree"))

    idx = L("cancer_level_dataset_index.csv")
    base = idx.groupby("record_id").agg(
        age_dx=("age_dx", "min"),
        stage=("stage_dx", "first"),
        grade=("ca_grade", "first"),
    )
    dob_dx = idx.groupby("record_id")["dob_ca_dx_days"].min().rename("dob_dx")

    pt = L("patient_level_dataset.csv").set_index("record_id")
    sex = pt["naaccr_sex_code"] if "naaccr_sex_code" in pt else pd.Series(dtype=object)
    race = pt["race_ethnicity"] if "race_ethnicity" in pt else pd.Series(dtype=object)

    df = pd.DataFrame(T).join([base, dob_dx, sample_type, oncotree])
    df["sex"] = sex; df["race"] = race
    df = df.join(pt[["hybrid_death_int", "last_alive_int"]])

    # --- pre-T disease state (events strictly before T) ---
    mon = L("med_onc_note_level_dataset.csv")[["record_id", "dx_md_visit_days", "md_ca_status"]].copy()
    mon["dx_md_visit_days"] = pd.to_numeric(mon["dx_md_visit_days"], errors="coerce")
    mon = mon.join(T, on="record_id")
    pre_status = mon[mon["dx_md_visit_days"] < mon["T"]].sort_values("dx_md_visit_days")
    last_status = pre_status.groupby("record_id")["md_ca_status"].last().rename("pre_status")
    prog_before = (pre_status.assign(p=pre_status["md_ca_status"].eq("Progressing/Worsening/Enlarging"))
                   .groupby("record_id")["p"].sum().rename("prog_events_before_T"))
    df = df.join([last_status, prog_before])

    img = L("imaging_level_dataset.csv")[["record_id", "dx_scan_days"]].copy()
    img["dx_scan_days"] = pd.to_numeric(img["dx_scan_days"], errors="coerce")
    img = img.join(T, on="record_id")
    scans_before = img[img["dx_scan_days"] < img["T"]].groupby("record_id").size().rename("scans_before_T")
    df = df.join(scans_before)

    # --- treatment history before T + regimen drugs (whole list, for "what worked") ---
    reg = L("regimen_cancer_level_dataset.csv")[
        ["record_id", "dx_drug_start_int_1", "regimen_drugs", "redcap_ca_index"]].copy()
    reg = reg[reg["redcap_ca_index"] == "Yes"]   # index (lung) cancer regimens ONLY -- avoid contamination from other cancers
    reg["dx_drug_start_int_1"] = pd.to_numeric(reg["dx_drug_start_int_1"], errors="coerce")
    reg = reg.join(T, on="record_id")
    reg_before = reg[reg["dx_drug_start_int_1"] < reg["T"]].groupby("record_id").size().rename("regimens_before_T")
    df = df.join(reg_before)
    drugs_all = reg.dropna(subset=["regimen_drugs"]).groupby("record_id")["regimen_drugs"].apply(
        lambda s: "; ".join(dict.fromkeys(s)))  # de-duped, order-preserving
    df = df.join(drugs_all.rename("regimen_drugs"))

    df = df.fillna({"prog_events_before_T": 0, "scans_before_T": 0, "regimens_before_T": 0})

    # ---------------- benefited label (B AND C) ----------------
    df["dead"] = df["hybrid_death_int"].notna()
    df["surv_dx"] = np.where(df["dead"], df["hybrid_death_int"] - df["dob_dx"],
                             df["last_alive_int"] - df["dob_dx"])
    df["post_surv"] = (df["surv_dx"] - df["T"]).clip(lower=0)
    surv_fav = df["post_surv"] > df["post_surv"].median()

    GOOD = {"Improving/Responding", "Stable/No change"}
    PROG = "Progressing/Worsening/Enlarging"
    monW = mon[(mon["dx_md_visit_days"] > mon["T"]) & (mon["dx_md_visit_days"] <= mon["T"] + WIN)]
    img2 = L("imaging_level_dataset.csv")[["record_id", "dx_scan_days", "image_overall"]].rename(
        columns={"dx_scan_days": "day", "image_overall": "status"})
    monX = monW.rename(columns={"dx_md_visit_days": "day", "md_ca_status": "status"})[["record_id", "day", "status"]]
    img2["day"] = pd.to_numeric(img2["day"], errors="coerce"); img2 = img2.join(T, on="record_id")
    img2 = img2[(img2["day"] > img2["T"]) & (img2["day"] <= img2["T"] + WIN)][["record_id", "day", "status"]]
    evW = pd.concat([monX, img2], ignore_index=True).dropna(subset=["day", "status"])
    def status_fav(g):
        g = g.sort_values("day"); return bool(g["status"].isin(GOOD).any() and g.iloc[-1]["status"] != PROG)
    sfav = evW.groupby("record_id").apply(status_fav, include_groups=False).rename("status_fav")
    df = df.join(sfav); df["status_fav"] = df["status_fav"].fillna(False)
    df["B"] = df["status_fav"] | surv_fav

    post = reg[reg["dx_drug_start_int_1"] > reg["T"]].groupby("record_id").size().rename("churn")
    df = df.join(post); df["churn"] = df["churn"].fillna(0)
    df["C"] = df["churn"] <= df["churn"].median()
    df["benefited"] = (df["B"] & df["C"]).astype(int)
    return df.reset_index()

def make_card(r):
    return (
        f"Lung cancer patient at genomic-test decision point (test ~{int(r['T'])} days after diagnosis).\n"
        f"Age at diagnosis: {r.get('age_dx','?')}; Sex: {r.get('sex','?')}.\n"
        f"Histology: {r.get('oncotree','?') if pd.notna(r.get('oncotree')) else 'NSCLC'}; "
        f"Stage at diagnosis: {r.get('stage','?')}; Grade: {r.get('grade','unknown') if pd.notna(r.get('grade')) else 'unknown'}.\n"
        f"Tumor sample at test: {r.get('sample_type','?')}.\n"
        f"Disease status before test: {r.get('pre_status','not documented') if pd.notna(r.get('pre_status')) else 'not documented'}; "
        f"progression events so far: {int(r.get('prog_events_before_T',0))}; scans so far: {int(r.get('scans_before_T',0))}.\n"
        f"Treatment before test: {int(r.get('regimens_before_T',0))} regimen(s)."
    )

# ----------------------------------------------------------------------------
# 3-4. Train/test split + build FAISS from TRAIN cards only
# ----------------------------------------------------------------------------
def split(df):
    rng = np.random.RandomState(SEED)
    test_ids = []
    for lab, g in df.groupby("benefited"):
        ids = g["record_id"].values.copy(); rng.shuffle(ids)
        test_ids.extend(ids[: int(0.18 * len(ids))])
    test_mask = df["record_id"].isin(test_ids)
    return df[~test_mask].copy(), df[test_mask].copy()

def build_index(train):
    emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=API_KEY)
    docs = []
    for _, r in train.iterrows():
        docs.append(Document(page_content=make_card(r), metadata={
            "record_id": r["record_id"], "benefited": int(r["benefited"]),
            "stage": str(r.get("stage")), "oncotree": str(r.get("oncotree")),
            "regimen_drugs": str(r.get("regimen_drugs")) if pd.notna(r.get("regimen_drugs")) else "",
        }))
    db = FAISS.from_documents(docs, emb)
    os.makedirs(os.path.dirname(INDEX_DIR), exist_ok=True)
    db.save_local(INDEX_DIR)
    print(f"  FAISS index built from {len(docs)} TRAIN patients -> {INDEX_DIR}")
    return db, emb

# ----------------------------------------------------------------------------
# 5. Cohort retrieval: similar -> keep benefited -> what worked
# ----------------------------------------------------------------------------
def retrieve_cohort(db, emb, card, stage=None, dist_max=DIST_MAX, use_stage=True):
    """Returns (selected_docs, top_plans, diag). Applies a distance threshold,
    an abstention rule, and MMR for diverse selection. `dist_max`/`use_stage`
    are relaxable by the Reflexion loop."""
    # 1) scored neighbors (FAISS L2 distance; lower = more similar)
    scored = db.similarity_search_with_score(card, k=K_RAW)
    keep = [(d, s) for d, s in scored
            if d.metadata.get("benefited") == 1 and s <= dist_max]
    # prefer same-stage matches when enough remain (skipped if relaxed)
    if use_stage and stage and stage != "nan":
        same = [(d, s) for d, s in keep if d.metadata.get("stage") == stage]
        if len(same) >= MIN_MATCHES:
            keep = same
    diag = {"n_matches": len(keep),
            "mean_dist": (float(np.mean([s for _, s in keep])) if keep else None),
            "best_dist": (float(min(s for _, s in keep)) if keep else None)}

    # 2) abstention rule: too little good evidence -> do not force an answer
    if len(keep) < MIN_MATCHES:
        diag.update(abstain=True,
                    reason=f"only {len(keep)} similar benefited patients within distance {dist_max}")
        return [], pd.Series(dtype=int), diag
    diag["abstain"] = False

    # 3) MMR: pick a DIVERSE subset of the qualifying matches (avoids near-duplicates)
    keep_ids = {d.metadata.get("record_id") for d, _ in keep}
    qvec = emb.embed_query(card)
    mmr = db.max_marginal_relevance_search_by_vector(
        qvec, k=min(K_KEEP * 2, K_RAW), fetch_k=K_RAW, lambda_mult=MMR_LAMBDA)
    diverse = [d for d in mmr if d.metadata.get("record_id") in keep_ids][:K_KEEP]
    if not diverse:                                   # fallback: nearest qualifying
        diverse = [d for d, _ in keep[:K_KEEP]]
    diag["n_used"] = len(diverse)

    # 4) mine the treatment plans that worked
    plans = []
    for d in diverse:
        txt = d.metadata.get("regimen_drugs", "")
        plans += [p.strip() for p in txt.split(";") if p.strip()]
    top = pd.Series(plans).value_counts().head(5) if plans else pd.Series(dtype=int)
    return diverse, top, diag

# ----------------------------------------------------------------------------
# 5b. REFLEXION loop for retrieval: Actor -> Evaluator -> Self-Reflection -> retry
# ----------------------------------------------------------------------------
TARGET_MATCHES = 10     # evidence is "strong" at/above this many diversified matches
WEAK_MEAN_DIST = 0.040  # mean distance above this = matches are clinically far/weak

# escalating relaxation plan; each step loosens one constraint
RELAX_PLAN = [
    dict(use_stage=True,  dist_max=DIST_MAX, desc="strict: same-stage filter, tight distance"),
    dict(use_stage=False, dist_max=DIST_MAX, desc="drop same-stage filter (search any stage)"),
    dict(use_stage=False, dist_max=0.060,    desc="widen distance threshold to 0.060"),
]

def evaluate_evidence(diag):
    """Evaluator: rate the retrieval result as strong / weak / abstain."""
    if diag.get("abstain"):
        return "abstain"
    if diag.get("n_used", 0) < TARGET_MATCHES or (diag.get("mean_dist") or 1) > WEAK_MEAN_DIST:
        return "weak"
    return "strong"

def reflect(llm, card, diag, next_desc):
    """Self-Reflection: short verbal note on WHY evidence was weak and what to relax."""
    return llm.invoke(
        "You are the retrieval self-reflection step of a clinical agent. The similar-patient search "
        "returned weak or insufficient evidence. In ONE sentence, say why and why the next relaxation is "
        "reasonable.\n"
        f"PATIENT:\n{card}\nDIAGNOSTICS: n_used={diag.get('n_used',0)}, "
        f"mean_dist={diag.get('mean_dist')}, abstain={diag.get('abstain')}\n"
        f"NEXT RELAXATION: {next_desc}"
    ).content.strip()

def retrieve_cohort_reflexive(db, emb, card, stage, llm, verbose=True):
    """Run retrieval; if evidence is weak/abstaining, reflect and retry with a
    relaxed search, up to len(RELAX_PLAN) attempts. Returns (docs, top, diag, verdict)."""
    reflections, docs, top, diag, verdict = [], [], None, {}, "abstain"
    for i, p in enumerate(RELAX_PLAN):
        docs, top, diag = retrieve_cohort(db, emb, card, stage=stage,
                                          dist_max=p["dist_max"], use_stage=p["use_stage"])
        verdict = evaluate_evidence(diag)
        if verbose:
            print(f"  [attempt {i+1}] {p['desc']} -> {verdict.upper()} "
                  f"(n={diag.get('n_used',0)}, mean_dist={diag.get('mean_dist')})")
        if verdict == "strong" or i == len(RELAX_PLAN) - 1:
            break
        note = reflect(llm, card, diag, RELAX_PLAN[i + 1]["desc"])  # Self-Reflection
        reflections.append(note)
        if verbose:
            print(f"      reflection: {note}")
    diag["attempts"] = i + 1
    diag["reflections"] = reflections
    return docs, top, diag, verdict

# ----------------------------------------------------------------------------
# 6. Observable demo: prompt-only vs retrieval-grounded
# ----------------------------------------------------------------------------
def demo(db, emb, test):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=API_KEY)
    # pick a non-trivial example: progressing before test if possible
    cand = test[test["pre_status"].eq("Progressing/Worsening/Enlarging")]
    row = (cand.iloc[0] if len(cand) else test.iloc[0])
    card = make_card(row)

    print("\n================= SAMPLE TEST PATIENT (held out, not in index) =================")
    print(card)

    print("\n----------------- (A) PROMPT-ONLY (no retrieval) -----------------")
    a = llm.invoke("You are assisting an oncologist. Given this patient profile, what should be "
                   "considered regarding treatment? Be specific.\n\n" + card).content
    print(a)

    print("\n----------------- (B) RETRIEVAL-GROUNDED (Reflexion-enabled) -----------------")
    ben, top, diag, verdict = retrieve_cohort_reflexive(db, emb, card, str(row.get("stage")), llm)
    if diag["abstain"]:
        print("Agent recommendation (grounded): Insufficient similar-patient evidence even after "
              "reflexive relaxation; defer to the clinician's own evaluation.")
    else:
        quality = (f"{diag['n_used']} similar benefited patients "
                   f"(MMR-diversified; nearest {diag['best_dist']:.3f}, mean {diag['mean_dist']:.3f}; "
                   f"{diag['attempts']} retrieval attempt(s))")
        cohort_txt = "\n".join(f"  - {d} (worked for {n} of {diag['n_used']} similar patients who benefited)"
                               for d, n in top.items()) or "  (none)"
        print(f"Match quality: {quality}")
        print("Treatment plans that worked for similar patients who benefited:")
        print(cohort_txt)
        grounded = llm.invoke(
            "You are assisting an oncologist (decision-support, not diagnosis). Recommend next actions for the "
            "patient, USING ONLY the evidence from similar patients below. Name only treatments that appear in "
            "the evidence. State the match quality so the clinician can judge confidence.\n\n"
            f"PATIENT:\n{card}\n\nMATCH QUALITY: {quality}\n"
            f"EVIDENCE — treatment plans that worked for similar patients who benefited:\n{cohort_txt}"
        ).content
        print("\nAgent recommendation (grounded):\n" + grounded)
    print("\n================= OBSERVABLE DIFFERENCE =================")
    print("(A) is generic and ungrounded; (B) names specific regimens from real similar patients,\n"
          "MMR-diversified, distance-thresholded, with abstention, surfaced match quality, and a\n"
          "Reflexion loop that relaxes and retries when first-pass evidence is weak.")

    # ---------- Reflexion in action: find a patient whose STRICT pass is weak ----------
    print("\n================= REFLEXION IN ACTION (weak first-pass case) =================")
    chosen = None
    for _, r in test.head(40).iterrows():
        c = make_card(r)
        _, _, d0 = retrieve_cohort(db, emb, c, stage=str(r.get("stage")))   # strict pass
        if evaluate_evidence(d0) != "strong":
            chosen = (r, c, d0); break
    if chosen is None:
        print("(No weak first-pass patient found in the sample; strict retrieval was strong for all.)")
        return
    r, c, _ = chosen
    print("Patient:\n" + c)
    print("\nReflexion trace (Actor -> Evaluator -> Self-Reflection -> retry):")
    docs, top, diag, verdict = retrieve_cohort_reflexive(db, emb, c, str(r.get("stage")), llm)
    print(f"\nFinal verdict after {diag['attempts']} attempt(s): {verdict.upper()}")
    if diag["abstain"]:
        print("Outcome: principled ABSTENTION — insufficient similar evidence even after relaxation.")
    else:
        best = top.index[0] if len(top) else "n/a"
        print(f"Outcome: recovered {diag['n_used']} matches after relaxation; top plan that worked: {best}")

if __name__ == "__main__":
    print("[1/4] Building patient table (features + benefited label)...")
    df = build_patient_table()
    print(f"      patients: {len(df)} | benefited=1: {int(df['benefited'].sum())} ({df['benefited'].mean()*100:.1f}%)")
    print("[2/4] Train/test split (patient-level, stratified, TRAIN-only index)...")
    train, test = split(df)
    print(f"      train: {len(train)} | test: {len(test)}")
    print("[3/4] Building FAISS index from TRAIN patient cards (OpenAI embeddings)...")
    db, emb = build_index(train)
    print("[4/4] Running observable retrieval demo...")
    demo(db, emb, test)
