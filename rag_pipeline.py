import requests
import zipfile
import io
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openpyxl import load_workbook

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"


def load_xlsx_as_documents(xlsx_path: str) -> list[Document]:
    """Load an Excel workbook and convert each sheet to a Document."""
    wb = load_workbook(xlsx_path, read_only=True)
    docs = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        lines = []
        for row in ws.iter_rows(values_only=True):
            if any(v is not None for v in row):
                lines.append(" | ".join(str(v) if v is not None else "" for v in row))
        content = f"[FEUILLE: {sheet_name}]\n" + "\n".join(lines)
        docs.append(Document(
            page_content=content,
            metadata={"source": xlsx_path, "sheet": sheet_name}
        ))
    return docs


def load_zip_pdfs_as_documents(zip_path: str) -> list[Document]:
    """Extract all PDFs from a ZIP archive and load them as Documents."""
    docs = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        pdf_entries = [name for name in zf.namelist() if name.lower().endswith(".pdf")]
        for entry in pdf_entries:
            with zf.open(entry) as f:
                pdf_bytes = f.read()

            # Write to a temp file so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            try:
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                for page in pages:
                    page.metadata["source"] = entry  # Use zip-internal path as source
                docs.extend(pages)
            finally:
                os.unlink(tmp_path)

    return docs


class RAGSystem:

    def __init__(self, pdf_path: str, xlsx_path: str = None, zip_path: str = None):
        all_documents = []

        # 1. Load the main financial PDF
        loader = PyPDFLoader(pdf_path)
        all_documents.extend(loader.load())
        print(f"✅ PDF chargé : {pdf_path} ({len(all_documents)} pages)")

        # 2. Load the Excel business plan (all sheets)
        if xlsx_path and os.path.exists(xlsx_path):
            xlsx_docs = load_xlsx_as_documents(xlsx_path)
            all_documents.extend(xlsx_docs)
            print(f"✅ Excel chargé : {xlsx_path} ({len(xlsx_docs)} feuilles)")

        # 3. Load all PDFs from the DataRoom ZIP
        if zip_path and os.path.exists(zip_path):
            zip_docs = load_zip_pdfs_as_documents(zip_path)
            all_documents.extend(zip_docs)
            print(f"✅ DataRoom ZIP chargé : {zip_path} ({len(zip_docs)} pages PDF)")

        print(f"\n📚 Total documents chargés : {len(all_documents)}")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(all_documents)
        print(f"🔪 Chunks générés : {len(chunks)}")

        # Build vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        print("✅ Index vectoriel construit.\n")

    def query(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'inconnu')}]\n{doc.page_content}"
            for doc in docs
        )

        prompt = f"""Tu es un analyste financier expert sur FallahTech SARL.
Tu as accès aux états financiers historiques (2023-2025), au business plan prévisionnel (2026-2029), \
aux statuts juridiques, aux contrats de partenariat avec les coopératives, \
au registre du personnel et à l'étude de marché AgriTech Maghreb.

Si la question ne concerne pas FallahTech ou ses activités, réponds uniquement : \
"Cette question ne concerne pas le document financier de FallahTech."
Utilise uniquement les données du contexte. Ne génère aucun chiffre inventé. Réponds en français de façon concise.

CONTEXTE:
{context}

QUESTION: {question}
RÉPONSE:"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 600,
                    "num_ctx": 4096,
                }
            },
            timeout=180
        )
        response.raise_for_status()
        raw = response.json()
        return raw.get("response") or raw.get("message", {}).get("content", "Pas de réponse reçue.")


# ── Usage example ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rag = RAGSystem(
        pdf_path="C:\\Users\\chbel bh\\Desktop\\RAG_FREE\\Etats_Financiers_Historiques_NCT.pdf",
        xlsx_path="C:\\Users\\chbel bh\\Desktop\\RAG_FREE\\FallahTech_BusinessPlan_Complet.xlsx",
        zip_path="C:\\Users\\chbel bh\\Desktop\\RAG_FREE\\DataRoom_FallahTech_Professionnelle.zip",
    )

    questions = [
        "FallahTech a-t-elle des dettes long terme ?",
        "Quel est le chiffre d'affaires prévisionnel en 2027 ?",
        "Combien d'employés a FallahTech en 2025 ?",
        "Quelle est la structure de l'actionnariat de FallahTech ?",
        "Est-ce que FallahTech est un bon investissement ?",
    ]

    for q in questions:
        print(f"\n❓ {q}")
        print(f"💬 {rag.query(q)}")
