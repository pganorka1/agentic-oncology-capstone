"""
journey_builder.py  —  Agent 1: Journey-Builder

Assembles a single TIME-ORDERED clinical timeline for a patient from the 9
GENIE-BPC files, joined on record_id and ordered by days-from-diagnosis.

KEY PROPERTY (the leakage firewall): the timeline is TRUNCATED at the decision
point T (the first genomic test). Only events strictly BEFORE T are included;
the test at T is marked as "now / decision point". No future event can reach
downstream agents.

Output:
  - header:    static facts known at diagnosis (age, sex, stage, histology)
  - timeline:  chronological list of events (day, category, description) before T
  - narrative: a short LLM summary of the journey up to the decision point

Run:  python journey_builder.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd, numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

HERE = os.path.dirname(os.path.abspath(__file__))   # repo folder, resolved at runtime (portable; no hard-coded path)
# load .env from the repo folder OR its parent (supports the original layout where .env sits one level up)
for _d in (HERE, os.path.dirname(HERE)):
    if os.path.exists(os.path.join(_d, ".env")):
        load_dotenv(os.path.join(_d, ".env")); break
API_KEY = os.getenv("OPENAI_API_KEY")
DATA = os.path.join(HERE, "data", "PJ_Data")

def L(f): return pd.read_csv(os.path.join(DATA, f), low_memory=False)

def load_all():
    return {
        "cpt":  L("cancer_panel_test_level_dataset.csv"),
        "idx":  L("cancer_level_dataset_index.csv"),
        "path": L("pathology_report_level_dataset.csv"),
        "img":  L("imaging_level_dataset.csv"),
        "mon":  L("med_onc_note_level_dataset.csv"),
        "reg":  L("regimen_cancer_level_dataset.csv"),
        "pt":   L("patient_level_dataset.csv"),
    }

def num(s): return pd.to_numeric(s, errors="coerce")

# ----------------------------------------------------------------------------
# Core: build the leakage-safe timeline for one patient
# ----------------------------------------------------------------------------
def build_timeline(record_id, dfs, T=None):
    cpt = dfs["cpt"]; rid = record_id
    pat_cpt = cpt[cpt.record_id == rid]
    if T is None:
        T = float(num(pat_cpt["dx_cpt_rep_days"]).min())   # decision point = first genomic test

    # ---- static header (known at diagnosis) ----
    irow = dfs["idx"][dfs["idx"].record_id == rid]
    prow = dfs["pt"][dfs["pt"].record_id == rid]
    def g(df, col):
        return df.iloc[0][col] if (len(df) and col in df and pd.notna(df.iloc[0][col])) else None
    header = {
        "record_id": rid,
        "age_dx": g(irow, "age_dx"),
        "sex": g(prow, "naaccr_sex_code"),
        "stage_dx": g(irow, "stage_dx"),
        "ca_type": g(irow, "ca_type"),
        "grade": g(irow, "ca_grade"),
        "T": T,
    }

    events = []
    def add(day, cat, desc):
        d = num(pd.Series([day]))[0]
        if pd.notna(d):
            events.append({"day": int(d), "category": cat, "description": desc})

    # diagnosis at day 0
    add(0, "Diagnosis",
        f"{header['ca_type'] or 'NSCLC'}, {header['stage_dx'] or 'stage unknown'}, "
        f"grade {header['grade'] or 'NA'}, age {header['age_dx'] or 'NA'}")

    # pathology reports
    p = dfs["path"]; p = p[p.record_id == rid]
    for _, r in p.iterrows():
        add(r.get("dx_path_proc_days"), "Pathology",
            f"{r.get('path_proc_type','report') if pd.notna(r.get('path_proc_type')) else 'report'}")

    # imaging scans
    im = dfs["img"]; im = im[im.record_id == rid]
    for _, r in im.iterrows():
        st = r.get("image_overall"); st = st if pd.notna(st) else "status n/a"
        add(r.get("dx_scan_days"), "Imaging",
            f"{r.get('image_scan_type','scan') if pd.notna(r.get('image_scan_type')) else 'scan'}: {st}")

    # oncology visits
    mo = dfs["mon"]; mo = mo[mo.record_id == rid]
    for _, r in mo.iterrows():
        st = r.get("md_ca_status"); st = st if pd.notna(st) else "status n/a"
        add(r.get("dx_md_visit_days"), "Onc visit", f"cancer status: {st}")

    # treatment regimens
    rg = dfs["reg"]; rg = rg[rg.record_id == rid]
    for _, r in rg.iterrows():
        drugs = r.get("regimen_drugs"); drugs = drugs if pd.notna(drugs) else "drugs n/a"
        ln = r.get("regimen_number")
        add(r.get("dx_drug_start_int_1"), "Treatment",
            f"regimen{(' line '+str(int(ln))) if pd.notna(ln) else ''} started: {drugs}")

    # ---- FIREWALL: keep only events strictly before T ----
    before = [e for e in events if e["day"] < T]
    after_count = len([e for e in events if e["day"] >= T])
    before.sort(key=lambda e: e["day"])

    # mark the decision point itself
    samp = g(pat_cpt.sort_values("dx_cpt_rep_days"), "sample_type")
    onc = g(pat_cpt.sort_values("dx_cpt_rep_days"), "cpt_oncotree_code")
    decision = {"day": int(T), "category": "DECISION POINT",
                "description": f"Genomic panel test ordered (sample: {samp or 'NA'}, {onc or 'NSCLC'})"}
    return header, before, decision, after_count

def render_timeline(header, before, decision):
    lines = [f"PATIENT {header['record_id']} | {header['ca_type'] or 'NSCLC'}, "
             f"{header['stage_dx'] or 'stage NA'}, age {header['age_dx'] or 'NA'}, sex {header['sex'] or 'NA'}",
             f"Decision point T = day {int(header['T'])} (first genomic test)",
             "-" * 70]
    for e in before:
        lines.append(f"  day {e['day']:>5}  [{e['category']:<10}] {e['description']}")
    lines.append(f"  day {decision['day']:>5}  [{decision['category']}] {decision['description']}  <== NOW")
    return "\n".join(lines)

def summarize(llm, header, before, decision):
    timeline_txt = render_timeline(header, before, decision)
    return llm.invoke(
        "You are the Journey-Builder agent in a clinical decision-support system. Summarize this patient's "
        "journey UP TO the decision point in 4-6 sentences for an oncologist: diagnosis, how the disease and "
        "treatment evolved over time, and the clinical situation at the moment the genomic test is ordered. "
        "Use ONLY the timeline below; do not invent facts.\n\n" + timeline_txt
    ).content

# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("[1/3] Loading the 9 datasets...")
    dfs = load_all()
    cpt = dfs["cpt"]
    T_all = cpt.groupby("record_id")["dx_cpt_rep_days"].min()

    print("[2/3] Choosing a patient with a rich pre-T history for the demo...")
    # count events before T per patient (notes+imaging+regimens+pathology), prefer T<=1000
    def pre_counts(df, daycol):
        d = df[["record_id", daycol]].copy(); d[daycol] = num(d[daycol])
        d = d.join(T_all.rename("T"), on="record_id")
        return d[d[daycol] < d["T"]].groupby("record_id").size()
    counts = (pre_counts(dfs["mon"], "dx_md_visit_days")
              .add(pre_counts(dfs["img"], "dx_scan_days"), fill_value=0)
              .add(pre_counts(dfs["reg"], "dx_drug_start_int_1"), fill_value=0)
              .add(pre_counts(dfs["path"], "dx_path_proc_days"), fill_value=0))
    elig = counts[T_all.reindex(counts.index).between(100, 1000)]
    rid = elig.sort_values(ascending=False).index[0]

    print("[3/3] Building the leakage-safe timeline...\n")
    header, before, decision, after_count = build_timeline(rid, dfs)
    print("================= JOURNEY TIMELINE (leakage-safe, truncated at T) =================")
    print(render_timeline(header, before, decision))
    print("-" * 70)
    print(f"Firewall check: {len(before)} events kept (day < T); "
          f"{after_count} later events EXCLUDED (day >= T) to prevent look-ahead leakage.")

    print("\n================= NARRATIVE SUMMARY (Journey-Builder agent) =================")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=API_KEY)
    print(summarize(llm, header, before, decision))
