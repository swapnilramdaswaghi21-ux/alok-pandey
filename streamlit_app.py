"""
Streamlit AI B2B Dashboard
- Upload CSV to score accounts (P-Score, CLV, segmentation)
- Upload PDF / Word to ingest text, extract percentages & monetary values
- Optional OpenAI postprocessing if OPENAI_API_KEY is set in Streamlit Cloud secrets
"""
import os
import io
import json
import re
import tempfile
from math import exp
import streamlit as st
import pandas as pd
import plotly.express as px

# Optional libraries
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import docx
except Exception:
    docx = None

try:
    from sklearn.cluster import KMeans
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# OpenAI optional
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
if OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
    except Exception:
        openai = None
else:
    openai = None

# Default demo weights
DEMO_WEIGHTS = {
    "intercept": -2.2,
    "company_size": 0.018,
    "time_on_page": 0.035,
    "emails_opened": 0.12,
    "webinar_attendance": 0.85,
    "competitor_search_score": 0.5,
    "acv": 0.0008
}

WEIGHTS_FILE = "weights.json"

def load_weights():
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEMO_WEIGHTS
    return DEMO_WEIGHTS

def save_weights(w):
    try:
        with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(w, f, indent=2)
        return True
    except Exception:
        return False

def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

def compute_pscore_row(row, weights):
    try:
        cs = float(row.get("company_size", 0) or 0)
        top = float(row.get("time_on_page", 0) or 0)
        emails = float(row.get("emails_opened", 0) or 0)
        webbin = 1 if str(row.get("webinar_attendance", "")).lower() in ["1", "true", "yes"] or (row.get("webinar_attendance") and str(row.get("webinar_attendance")).isdigit() and float(row.get("webinar_attendance")) > 0) else 0
        comp = float(row.get("competitor_search_score", 0) or 0)
        acv = float(row.get("acv", 0) or 0)
        lin = weights.get("intercept", 0)
        lin += weights.get("company_size", 0) * cs
        lin += weights.get("time_on_page", 0) * top
        lin += weights.get("emails_opened", 0) * emails
        lin += weights.get("webinar_attendance", 0) * webbin
        lin += weights.get("competitor_search_score", 0) * comp
        lin += weights.get("acv", 0) * acv
        return sigmoid(lin)
    except Exception:
        return 0.0

def compute_clv_row(row):
    try:
        acv = float(row.get("acv", 0) or 0)
        baseRetention = min(0.95, 0.5 + (float(row.get("time_on_page", 0) or 0) / 200.0) + (1 if str(row.get("webinar_attendance", "")).lower() in ["1", "true", "yes"] else 0) * 0.12 - (float(row.get("competitor_search_score", 0) or 0) * 0.02))
        lifespan = max(1, round(baseRetention * 5))
        clv = acv * lifespan
        return round(clv)
    except Exception:
        return 0

def nba_rule(p, clv):
    if p > 0.75 and clv > 50000:
        return "Assign Enterprise AE + Exec Review"
    if p > 0.7:
        return "AE outreach — book discovery"
    if p > 0.5:
        return "Nurture - personalized content"
    return "Monitor intent; retarget via ABM ads"

st.set_page_config(page_title="AI B2B Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("Interactive AI B2B Dashboard (Streamlit)")

# Sidebar controls
st.sidebar.header("Model & Deployment")
weights = load_weights()
st.sidebar.subheader("Model weights (editable)")
weights_edit = {}
for k, v in weights.items():
    weights_edit[k] = st.sidebar.number_input(k, value=float(v), format="%.6f")
if st.sidebar.button("Save weights"):
    ok = save_weights(weights_edit)
    if ok:
        st.sidebar.success("Weights saved to local weights.json")
    else:
        st.sidebar.error("Failed to save weights (read/write perms?)")

st.sidebar.markdown("---")
st.sidebar.write("To deploy: push this repo to GitHub and connect it in Streamlit Cloud (https://streamlit.io/cloud).")

# Main UI: two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.header("1) Ingest Project Document (PDF or Word)")
    doc_file = st.file_uploader("Upload PDF or Word (.pdf, .docx, .doc)", type=["pdf", "docx", "doc"])
    if st.button("Extract from example local report"):
        example_path = "/mnt/data/B2B WAI Project 2025-2026.pdf"
        if os.path.exists(example_path):
            doc_file = open(example_path, "rb")
            st.success("Using local example report.")
        else:
            st.warning("No local example found.")
    if doc_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(getattr(doc_file, "name", "doc"))[1])
        try:
            # handle both UploadedFile and raw file object
            if hasattr(doc_file, "read"):
                tmp.write(doc_file.read())
            else:
                tmp.write(doc_file.read())
        except Exception:
            pass
        tmp.flush()
        tmp.close()
        path = tmp.name
        pages = []
        if fitz and path.lower().endswith(".pdf"):
            try:
                doc = fitz.open(path)
                for i in range(doc.page_count):
                    page = doc.load_page(i)
                    pages.append({"page": i + 1, "text": page.get_text("text")})
                doc.close()
            except Exception as e:
                pages = [{"page": 1, "text": f"(PyMuPDF error: {e})"}]
        elif docx and (path.lower().endswith(".docx") or path.lower().endswith(".doc")):
            try:
                d = docx.Document(path)
                text = "\n\n".join([p.text for p in d.paragraphs])
                pages = [{"page": 1, "text": text}]
            except Exception as e:
                pages = [{"page": 1, "text": f"(python-docx error: {e})"}]
        else:
            pages = [{"page": 1, "text": "(No extractor available)"}]

        full_text = "\n\n".join([p["text"] for p in pages])
        st.subheader("Extracted snippet")
        st.text_area("Text snippet", value=full_text[:3000], height=300)

        # quick parses
        pct = [float(m.group(1)) for m in re.finditer(r"([+-]?\d{1,3}(?:\.\d+)?)\s*%", full_text)]
        monies = re.findall(r"\$[\d,]+(?:\.\d+)?", full_text)
        st.write("Percentages found:", pct[:20])
        st.write("Monetary values found:", monies[:20])

        if openai:
            if st.button("Run OpenAI postprocess (may cost)"):
                with st.spinner("Calling OpenAI..."):
                    prompt = f"Extract top findings and recommended actions from the following text:\n\n{full_text[:4000]}"
                    try:
                        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "You are a data analyst."}, {"role": "user", "content": prompt}], max_tokens=600)
                        out = resp["choices"][0]["message"]["content"]
                        st.subheader("LLM output")
                        st.write(out)
                    except Exception as e:
                        st.error(f"OpenAI error: {e}")
        else:
            st.info("Set OPENAI_API_KEY in Streamlit Cloud secrets to enable LLM postprocessing.")

    st.markdown("---")
    st.header("2) Score Data (CSV)")
    csv_file = st.file_uploader("Upload CSV with required columns", type=["csv"])
    st.info("Required columns: account_id, company_name, company_size, time_on_page, emails_opened, webinar_attendance, competitor_search_score, acv")
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            st.write("Uploaded data (first rows):")
            st.dataframe(df.head())
            df = df.fillna(0)
            w = load_weights()
            df["pscore"] = df.apply(lambda r: compute_pscore_row(r, w), axis=1)
            df["clv"] = df.apply(lambda r: compute_clv_row(r), axis=1)
            # segmentation
            if HAVE_SKLEARN:
                features = [c for c in ["company_size", "time_on_page", "emails_opened"] if c in df.columns]
                if features and len(df) >= 3:
                    X = df[features].astype(float).values
                    k = min(3, len(df))
                    km = KMeans(n_clusters=k, random_state=42).fit(X)
                    df["segment"] = km.labels_
                else:
                    df["segment"] = 0
            else:
                df = df.sort_values(by="pscore", ascending=False).reset_index(drop=True)
                df["segment"] = (df.index % 3)
            df["nba"] = df.apply(lambda r: nba_rule(r["pscore"], r["clv"]), axis=1)

            st.subheader("Scored accounts (top 50 shown)")
            st.dataframe(df.sort_values("pscore", ascending=False).head(50))

            # plot P-score bar chart
            name_col = "company_name" if "company_name" in df.columns else ("account_id" if "account_id" in df.columns else df.index)
            fig = px.bar(df, x=name_col, y=(df["pscore"] * 100).round(1),
                         labels={"y": "P-Score (%)", "x": "Account"},
                         title="P-Score (%) by Account")
            st.plotly_chart(fig, use_container_width=True)

            # download scored CSV
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download scored CSV", data=csv_bytes, file_name="scored_accounts.csv", mime="text/csv")

            # quick insights
            top = df.sort_values("pscore", ascending=False).head(5)
            st.subheader("Top accounts & recommended actions")
            for _, r in top.iterrows():
                st.markdown(f"- **{r.get('company_name', r.get('account_id','—'))}** — P-Score: {round(r.pscore*100)}% | CLV: ${int(r.clv)} → _{r.nba}_")
        except Exception as e:
            st.error(f"CSV processing error: {e}")

with col2:
    st.header("Model Weights & Quick Controls")
    st.write("Current weights (weights.json) — edit & save in sidebar.")
    st.code(json.dumps(load_weights(), indent=2))
    st.markdown("---")
    st.header("Deploy on Streamlit Cloud")
    st.markdown("""
    1. Create a GitHub repo and push these files:
       - streamlit_app.py
       - requirements.txt
       - README.md

    2. Go to https://streamlit.io/cloud → New app → Connect your GitHub repo → Select branch and file `streamlit_app.py`.

    3. (Optional) In Streamlit Cloud → Settings → Secrets, add OPENAI_API_KEY to enable LLM postprocessing.

    After deployment you can use the public Streamlit URL that the Cloud provides.
    """)
    st.markdown("When ready, the GitHub URL you will paste into Streamlit Cloud looks like:")
    st.code("https://github.com/<your-username>/b2b-ai-dashboard-streamlit")

st.markdown("---")
st.caption("Tip: If you want me to generate the exact git commands to create a repo and push these files, tell me your GitHub username and I will prepare them.")