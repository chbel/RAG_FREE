import streamlit as st
from rag_pipeline import RAGSystem

st.set_page_config(page_title="FallahTech — Analyse Financière")
st.title("FallahTech — Analyse Financière et Décision d'Investissement")

PDF_PATH = "C:\\Users\\chbel bh\\Desktop\\RAG_FREE\\Etats_Financiers_Historiques_NCT.pdf"
XLSX_PATH = "C:\\Users\\chbel bh\\Desktop\\RAG_FREE\\FallahTech_BusinessPlan_Complet.xlsx"
ZIP_PATH = "C:\\Users\\chbel bh\\Desktop\\RAG_FREE\\DataRoom_FallahTech_Professionnelle.zip"

@st.cache_resource(show_spinner="Chargement des documents...")
def load_rag():
    return RAGSystem(
        pdf_path=PDF_PATH,
        xlsx_path=XLSX_PATH,
        zip_path=ZIP_PATH
    )

rag = load_rag()

question = st.text_input("Pose ta question :", placeholder="Ex: Est-ce que FallahTech est rentable ?")

if st.button("Analyser", type="primary"):
    if not question.strip():
        st.warning("Entre une question d'abord.")
    else:
        with st.spinner("Analyse en cours..."):
            try:
                result = rag.query(question)
                st.markdown("### 📋 Résultat de l'analyse")
                st.markdown(result)
            except Exception as e:
                st.error(f"Erreur : {e}")
