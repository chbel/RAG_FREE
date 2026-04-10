import requests
import os
import fitz  # PyMuPDF — better font handling than PyPDFLoader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openpyxl import load_workbook

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

CRITERIA = {
    "Financier": 40,
    "Commercial": 30,
    "Équipe": 15,
    "Marché": 15,
}

# ── PDF loader using PyMuPDF (handles custom fonts better) ────────────────────
def load_pdf_pymupdf(pdf_path: str) -> list:
    docs = []
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": f"{os.path.basename(pdf_path)}:page{i+1}"}
                ))
        doc.close()
    except Exception as e:
        print(f"[WARN] Could not read {pdf_path}: {e}")
    return docs


# ── Load ALL PDFs from a folder ───────────────────────────────────────────────
def load_all_pdfs_from_folder(folder_path: str) -> list:
    docs = []
    if not os.path.exists(folder_path):
        print(f"[WARN] Folder not found: {folder_path}")
        return docs
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            loaded = load_pdf_pymupdf(full_path)
            print(f"  ✓ {filename} — {len(loaded)} pages loaded")
            docs.extend(loaded)
    return docs


# ── Excel loader ──────────────────────────────────────────────────────────────
def load_xlsx_as_documents(xlsx_path: str) -> list:
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
            metadata={"source": f"Excel:{sheet_name}"}
        ))
    return docs


def read_xlsx_raw(xlsx_path: str) -> str:
    wb = load_workbook(xlsx_path, read_only=True)
    output = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        output.append(f"\n=== FEUILLE: {sheet_name} ===")
        for row in ws.iter_rows(values_only=True):
            if any(v is not None for v in row):
                output.append(" | ".join(str(v) if v is not None else "" for v in row))
    return "\n".join(output)


def _call_ollama(prompt: str, max_tokens: int = 500) -> str:
    if len(prompt) > 4000:
        prompt = prompt[:4000] + "\n...[contexte tronqué]"
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": max_tokens,
                    "num_ctx": 2048,
                    "num_thread": 4,
                }
            },
            timeout=180
        )
        response.raise_for_status()
        return response.json().get("response", "Pas de réponse reçue.")
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        return f"[Erreur Ollama {status}] Redémarrez Ollama et réessayez."
    except requests.exceptions.ConnectionError:
        return "[Erreur] Ollama n'est pas démarré. Lancez `ollama serve`."
    except requests.exceptions.Timeout:
        return "[Erreur] Timeout. Réessayez."
    except Exception as e:
        return f"[Erreur inattendue] {str(e)}"


class RAGSystem:

    def __init__(self, pdf_folder: str, xlsx_path: str = None):
        self.xlsx_path = xlsx_path
        all_documents  = []

        # Load all PDFs from folder
        print(f"\n📂 Loading PDFs from: {pdf_folder}")
        all_documents.extend(load_all_pdfs_from_folder(pdf_folder))

        # Load Excel
        if xlsx_path and os.path.exists(xlsx_path):
            xl_docs = load_xlsx_as_documents(xlsx_path)
            print(f"  ✓ Excel — {len(xl_docs)} sheets loaded")
            all_documents.extend(xl_docs)

        print(f"\n✅ Total documents loaded: {len(all_documents)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        chunks = splitter.split_documents(all_documents)
        print(f"✅ Total chunks: {len(chunks)}")

        embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def _get_excel_context(self) -> str:
        if self.xlsx_path and os.path.exists(self.xlsx_path):
            return read_xlsx_raw(self.xlsx_path)[:2500]
        return ""

    def _score_criterion(self, criterion: str, weight: int) -> dict:
        docs = self.retriever.invoke(f"FallahTech {criterion} performance données chiffres")
        rag_context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'inconnu')}]\n{doc.page_content}"
            for doc in docs
        )
        excel_context = self._get_excel_context() if criterion == "Financier" else ""
        context = ""
        if excel_context:
            context += f"[DONNÉES EXCEL]\n{excel_context}\n\n"
        context += rag_context

        prompt = f"""Tu es un analyste financier expert évaluant FallahTech SARL.

CRITÈRE : {criterion} (pondération : {weight}%)

DONNÉES :
{context}

RÈGLES : Utilise UNIQUEMENT les chiffres présents. Ne calcule pas. Ne déduis pas.

Réponds EXACTEMENT :
SCORE: X/10
SOURCE: [feuille ou fichier exact]
JUSTIFICATION: [1 phrase avec chiffres lus directement]"""

        text = _call_ollama(prompt, max_tokens=200)

        score = 5.0
        source = "inconnu"
        justification = text

        for line in text.splitlines():
            if line.startswith("SCORE:"):
                try:
                    score = float(line.replace("SCORE:", "").strip().split("/")[0])
                except Exception:
                    score = 5.0
            elif line.startswith("SOURCE:"):
                source = line.replace("SOURCE:", "").strip()
            elif line.startswith("JUSTIFICATION:"):
                justification = line.replace("JUSTIFICATION:", "").strip()

        return {
            "criterion": criterion,
            "weight": weight,
            "score_10": round(score, 1),
            "weighted_score": round((score / 10) * weight, 2),
            "source": source,
            "justification": justification,
        }

    def query(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        rag_context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'inconnu')}]\n{doc.page_content}"
            for doc in docs
        )
        excel_context = self._get_excel_context()
        full_context  = f"[DONNÉES EXCEL]\n{excel_context}\n\n{rag_context}"

        prompt = f"""Tu es un analyste financier expert sur FallahTech SARL.

RÈGLES STRICTES :
1. Si la valeur est dans le contexte → cite-la avec sa SOURCE exacte (fichier:page ou feuille Excel)
2. Si absente → réponds : "Donnée non trouvée dans les documents fournis."
3. NE JAMAIS calculer ou inventer un chiffre

DONNÉES :
{full_context}

QUESTION: {question}
RÉPONSE (avec source obligatoire) :"""

        return _call_ollama(prompt, max_tokens=400)

    def score_investment(self) -> dict:
        results = []
        total   = 0
        for criterion, weight in CRITERIA.items():
            result = self._score_criterion(criterion, weight)
            results.append(result)
            total += result["weighted_score"]

        if total >= 75:
            decision = "✅ Investir"
        elif total >= 50:
            decision = "⚠️ Investir sous conditions"
        else:
            decision = "❌ No-Go"

        return {
            "criteria": results,
            "total_score": round(total, 1),
            "decision": decision,
        }
