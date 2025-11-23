```markdown
# B2B AI Dashboard — Streamlit

This repository contains a Streamlit dashboard to score B2B accounts and ingest project documents (PDF/Word).

Files:
- streamlit_app.py — Streamlit application (UI + scoring + ingest)
- requirements.txt — dependencies

Quickstart (local):
1. Create virtualenv and install:
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt

2. Run locally:
   streamlit run streamlit_app.py

Deploy on Streamlit Cloud:
1. Create a public GitHub repo and push these files.
2. Go to https://streamlit.io/cloud and create a new app.
3. Connect your GitHub account, choose the repo and `streamlit_app.py`.
4. (Optional) In Streamlit Cloud -> Settings -> Secrets, add `OPENAI_API_KEY` to enable LLM postprocessing.
5. Deploy. Streamlit Cloud will provide a public URL.

If you'd like, I can also:
- Provide the exact git commands to create a repo and push these files (tell me your GitHub username).
- Convert this into a Flask->Streamlit migration commit if you're storing your current Flask code in a repo.
```