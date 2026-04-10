import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from rag_pipeline import RAGSystem

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER  = BASE_DIR          # All PDFs are in the same folder as app.py
XLSX_PATH   = os.path.join(BASE_DIR, r"C:\Users\chbel bh\Desktop\RAG_FREE\FallahTech_BusinessPlan_Complet.xlsx")

# ── Init RAG ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_rag():
    return RAGSystem(
        pdf_folder=PDF_FOLDER,
        xlsx_path=XLSX_PATH,
    )

rag = load_rag()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("FallahTech — Scoring Automatique du Dossier d'Investissement")
st.caption("Tâche 3 — RAG multicritère avec citation de source")

tab1, tab2 = st.tabs(["Scoring Automatique", "Question Libre"])

with tab1:
    if st.button("Lancer le scoring"):
        with st.spinner("Analyse en cours..."):
            result = rag.score_investment()
        st.metric("Score total", f"{result['total_score']} / 100")
        st.subheader(result["decision"])
        for c in result["criteria"]:
            with st.expander(f"{c['criterion']} — {c['score_10']}/10 (pondéré: {c['weighted_score']})"):
                st.write(f"**Source :** {c['source']}")
                st.write(f"**Justification :** {c['justification']}")

with tab2:
    st.subheader("Question libre sur le dossier")
    question = st.text_input("Pose ta question :")
    if st.button("Analyser"):
        if question:
            with st.spinner("Recherche en cours..."):
                answer = rag.query(question)
            st.subheader("Réponse")
            st.write(answer)
        else:
            st.warning("Veuillez entrer une question.")
