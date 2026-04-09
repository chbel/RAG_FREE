import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"


class RAGSystem:

    def __init__(self, PDF_PATH: str):
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def query(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""Tu es un analyste financier expert sur FallahTech SARL.
Si la question ne concerne pas FallahTech ou ses finances, réponds uniquement : "Cette question ne concerne pas le document financier de FallahTech."
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
