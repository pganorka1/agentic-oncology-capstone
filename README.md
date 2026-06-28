# Agentic Clinical Decision-Support for Proactive Genomic-Test Timing in NSCLC (Non-Small-Cell Lung Cancer)

An agentic, clinician-supervised decision-support system that synthesizes a lung-cancer (NSCLC)
patient's fragmented longitudinal record into a single time-ordered journey, and proactively flags
the **earliest point** at which the patient's profile resembles prior patients whose genomic testing
was followed by a beneficial outcome. At that point it surfaces a **calibrated benefit likelihood**
and **evidence-backed "what worked" treatment plans** drawn from similar, benefited patients — for the
oncologist to weigh alongside their own judgment.

> **Decision-support, not autonomous diagnosis.** Every recommendation is grounded in real cohort
> evidence, carries a mandatory *association-not-causation* caveat, and requires clinician sign-off.

---

## Problem

Oncologists must decide, across a long and fragmented patient journey, *when* a genomics-based
diagnostic test is most likely to change care for the better. The needed information is scattered
across many clinical systems (demographics, pathology, imaging, oncology notes, treatment regimens,
genomic panel results) and accumulates over months or years — impractical to review manually for
every patient at every visit to spot the right moment.

**Intended user:** the treating oncologist.

---

## Architecture

The system runs in two phases.

**Offline (deterministic, one-time):** build leakage-safe labels and features → train a calibrated
benefit-risk classifier → build a FAISS index of *training* patient profiles.

**Runtime (LangGraph state machine, per patient, clinician-supervised):** a four-agent pipeline.

| Agent | Role | Key tools |
|-------|------|-----------|
| **1 — Journey-Builder** | Joins the 9 files into one time-ordered timeline, **truncated at decision point T** (first genomic test) — the leakage firewall (only pre-T data used). | pandas + LLM (narrative) |
| **2 — Cohort-Retrieval** | FAISS similar-patient search → filter to `benefited=1` → mine the `regimen_drugs` that worked. Bounded **Reflexion** loop relaxes filters when matches are sparse; abstains if evidence is too thin. | FAISS + pandas |
| **3 — Timing & Recommendation** | Combines the **calibrated classifier** score with Agent 2's evidence to produce the verdict — **ACT NOW / NOT YET–MONITOR / INSUFFICIENT EVIDENCE** — plus next actions. Bounded **Tree-of-Thoughts** beam search over candidate strategies. | classifier + ToT + LLM |
| **4 — Safety / Critic** | Deterministic **groundedness check** (every named therapy must trace to retrieved evidence and the index/lung cancer), abstention enforcement, mandatory caveat. Capped Reflexion loop → one revision. | rule-based (no LLM) |

**Flow:** `1 → 2 → 3 → 4 → clinician` (Critic can loop back once to Agent 3).
**Predictor:** a **hybrid** — calibrated probability score *plus* real-cohort evidence.

```
            OFFLINE                                   RUNTIME (per patient)
  ┌───────────────────────────┐        ┌─────────────────────────────────────────────┐
  │ labels + leakage-safe      │        │  Agent 1  Journey-Builder  (timeline @ T)     │
  │ features                   │        │      │                                        │
  │ train calibrated classifier├──────► │      ▼                                        │
  │ build FAISS cohort index   ├──────► │  Agent 2  Cohort-Retrieval (what worked)      │
  └───────────────────────────┘        │      │                                        │
                                        │      ▼                                        │
                                        │  Agent 3  Timing & Recommendation (verdict)   │
                                        │      │            ▲ (revise, capped at 1)     │
                                        │      ▼            │                            │
                                        │  Agent 4  Safety / Critic ────────────────────┤
                                        │      │                                        │
                                        │      ▼                                        │
                                        │   Clinician decides                           │
                                        └─────────────────────────────────────────────┘
```

---

## Repository layout

| File | Purpose |
|------|---------|
| `build_classifier.py` | **Offline.** Trains LogisticRegression + HistGradientBoosting (Platt-calibrated), picks the winner on a held-out test, reports ROC-AUC / PR-AUC / Brier / calibration curve / subgroup AUC; saves the model artifact. |
| `journey_builder.py` | **Agent 1.** Joins the 9 files; truncates the timeline at T (leakage firewall); emits header + timeline + narrative. |
| `retrieval_agent.py` | **Agent 2.** Builds leakage-safe patient "cards" + the `benefited` label, the patient-level split, the FAISS index (train-only), and the reflexive cohort retrieval. |
| `agent3_timing.py` | **Agent 3.** Converges the classifier score + cohort evidence into the timing verdict + recommendation. |
| `agent3_tot.py` | **Agent 3 (ToT).** Bounded beam search over candidate action-strategies, with a deterministic rubric. |
| `agent4_critic.py` | **Agent 4.** Independent deterministic groundedness/safety review + caveat. |
| `orchestrator.py` | Wires Agents 1–4 into a LangGraph `StateGraph` with a capped revise edge. |
| `evaluate.py` | The 4-tier evaluation harness. |
| `demo.py` | Narrated end-to-end walkthrough of contrasting patients. |
| `plot_calibration.py` | Renders the calibration curve from the saved model. |

---

## Data

**AACR GENIE-BPC NSCLC v2.0-public** (lung cancer) — 9 linked CSVs keyed on `record_id`, all
de-identified with day/month/year **offset** dates (not calendar dates). The GENIE **cancer panel
test (NGS sequencing)** is used as a **proxy** for the genomics-based diagnostic test.

> ⚠️ **The dataset is NOT included in this repository.** It is obtained separately from the public
> AACR Project GENIE BPC release (via Synapse, with registration/terms). Place the 9 CSVs under
> `Capstone/data/PJ_Data/`. The `data/` folder, `.env`, and generated artifacts are excluded via
> `.gitignore`.

Files expected: `patient_level_dataset.csv`, `cancer_level_dataset_index.csv`,
`cancer_level_dataset_non_index.csv`, `cancer_panel_test_level_dataset.csv`,
`regimen_cancer_level_dataset.csv`, `pathology_report_level_dataset.csv`,
`med_onc_note_level_dataset.csv`, `imaging_level_dataset.csv`, `manifest.csv`.

---

## Setup

**Requirements:** Python 3.12+.

```bash
pip install pandas numpy scikit-learn faiss-cpu joblib matplotlib \
            langchain langchain-community langchain-openai langgraph python-dotenv
```

1. **OpenAI API key** — create a `.env` file (used for embeddings + narrative/synthesis only):
   ```
   OPENAI_API_KEY=sk-...
   ```
2. **Data** — place the 9 GENIE-BPC CSVs under `Capstone/data/PJ_Data/` (see above).
3. **Paths** — the scripts reference a few path constants (e.g. `DATA`, `INDEX_DIR`, the `.env`
   location) near the top of `retrieval_agent.py` and `build_classifier.py`. Adjust them to your
   local layout if you are not running from the original location.

Key tunable constants (in `retrieval_agent.py`): observation window `WIN=140` days,
`DIST_MAX=0.045` (FAISS distance cutoff), `MIN_MATCHES=5` (abstention threshold), `SEED=42`.

---

## Usage

```bash
# 1) Offline: train + calibrate the benefit-risk classifier (saves the model artifact)
python build_classifier.py

# 2) Build the cohort retrieval index + see prompt-only vs retrieval-grounded output
python retrieval_agent.py

# 3) Run the full multi-agent pipeline on one patient
python orchestrator.py

# 4) Narrated end-to-end demo for a chosen patient
python demo.py <record_id>

# 5) Run the 4-tier evaluation
python evaluate.py

# 6) Render the calibration curve from the saved model
python plot_calibration.py
```

---

## Evaluation & results

A 4-tier evaluation mirrors the system's responsibilities:

- **Tier 1 — Classifier (held-out test):** ROC-AUC **0.782**, PR-AUC 0.644, Brier **0.164**, and
  **well-calibrated**; beats majority-class and stage-only baselines. *(Primary criterion:
  ROC-AUC ≥ 0.70 + good calibration — met.)*
- **Tier 2 — Cohort retrieval:** "what-worked" **hit-rate@20 = 77.8%** (28/36).
- **Tier 3 — Proactivity (proxy):** median **lead-time ≈ 110 days** from flag to next progression.
- **Tier 4 — End-to-end:** **groundedness 100%** (no hallucinated therapy), correct deferral on
  abstention cases, active and explainable safety critic at low latency.

Subgroup AUCs are reported by institution, stage, sex, and age.

---

## Safety, reliability & human oversight

- **Input guardrails:** T-truncation firewall + leakage blocklist (no future information).
- **Output guardrails:** deterministic groundedness — every recommended therapy must trace to
  retrieved evidence and to the index (lung) cancer; hallucinated / wrong-cancer drugs are blocked.
- **Tool limits:** agents use fixed deterministic tools; the LLM does not freely choose actions.
- **Abstention:** triggers on sparse evidence or low confidence — the system prefers "insufficient
  evidence" over a guess.
- **Human intervention:** the system defers to the clinician on uncertain mid-range likelihood,
  sparse/low-quality cohorts, outliers with no close matches, critic-flagged ungrounded output, or
  any therapy that would be named without support. The clinician signs off on **all** recommendations.
- **Design trade-off:** a **deterministic** pipeline with bounded Reflexion + Tree-of-Thoughts was
  chosen over an open-ended ReAct agent — accepting less flexibility for predictability,
  reproducibility, and auditability, which a clinical setting demands.

---

## Limitations & next steps

**Limitations.** No mutation-level genomic data (panel file has sample/assay IDs only → modest lift
over a stage-only baseline); benefit is an **association, not causation** (all patients are tested —
no untested control group); the survival component can dominate the label; templated cards make
text-embedding similarity weakly discriminative; rare patients have no close benefited neighbors;
single cancer type; offset dates preclude a calendar split.

**Next steps.** Integrate real variant-level features (link panel samples to GENIE genomic data);
swap the tissue panel for a **blood-based liquid-biopsy** test for earlier detection; replace
text-embedding cards with a **clinically-weighted structured similarity vector**; implement the full
visit-by-visit Tier-3 simulation; validate prospectively against clinician judgment and outcomes.

---

## Disclaimer

Research / educational capstone project. **Not a medical device and not for clinical use.** Outputs
are associations from retrospective, de-identified data and must not be used to make patient-care
decisions.
