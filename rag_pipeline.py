import pypdf
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"


def load_pdf(pdf_path: str) -> str:
    reader = pypdf.PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


FINANCIAL_FACTS = """
=== CHIFFRES EXACTS À UTILISER ===

COMPTE DE RÉSULTAT:
- CA 2023: 250 000 TND | CA 2024: 780 000 TND | CA 2025: 1 650 000 TND
- Croissance CA: +212% en 2024, +111,5% en 2025
- Résultat net 2023: -50 000 TND | 2024: -40 000 TND | 2025: +45 000 TND
- Marge brute: 60% (2023) → 64% (2024) → 70% (2025)
- Marge nette: -20% (2023) → -5,1% (2024) → +2,7% (2025)
- Charges personnel: 180 000 (2023) → 320 000 (2024) → 480 000 TND (2025)
- R&D: 30 000 (2023) → 50 000 (2024) → 200 000 TND (2025)

BILAN:
- Total Actif: 465 000 (2023) → 500 000 (2024) → 930 000 TND (2025)
- Trésorerie: 320 000 (2023) → 220 000 (2024) → 510 000 TND (2025)
- Capitaux propres: 495 000 (2023) → 465 000 (2024) → 500 000 TND (2025)
- Dettes fournisseurs: 0 (2023) → 20 000 (2024) → 300 000 TND (2025)
- Capital social: 500 000 TND (inchangé)
- Aucune dette long terme

RATIOS:
- ROE 2025: 9,0% | ROA 2025: 4,8%
- Ratio courant 2025: 1,56 | Ratio trésorerie: 1,19
- Ratio fonds propres/actif: 0,54
- Ratio endettement: 0,86

ÉQUIPE:
- Effectif: 5 (2023) → 10 (2024) → 18 (2025)

RISQUES:
- Concentration client: 5 coopératives = 40% du CA
- Dépendance marché AgriTech
- Concurrence internationale

FLUX DE TRÉSORERIE:
- Flux exploitation 2025: +410 000 TND
- Flux investissement 2025: -120 000 TND
- Variation nette trésorerie 2025: +290 000 TND
"""


class RAGSystem:

    def __init__(self, pdf_path: str):
        self.text = load_pdf(pdf_path)

    def query(self, question: str) -> str:

        prompt = f"""Tu es un analyste financier expert. Réponds en français.

RÈGLES STRICTES :
- Utilise UNIQUEMENT les chiffres du bloc ci-dessous
- Ne répète jamais la même phrase
- Réponse concise, termine après la conclusion

{FINANCIAL_FACTS}

QUESTION: {question}

FORMAT DE RÉPONSE (puis STOP) :

## 1. Finance (X/10)
2 phrases avec chiffres exacts, sans crochets

## 2. Marché (X/10)
2 phrases avec chiffres exacts, sans crochets

## 3. Équipe (X/10)
2 phrases avec chiffres exacts, sans crochets

## 4. Croissance (X/10)
2 phrases avec chiffres exacts, sans crochets

## Score final : XX/100 (calcule la moyenne des 4 notes multipliée par 2.5, arrondis à l'entier)
## Décision : Investir / Sous conditions / No-Go
## Justification : 3 phrases maximum avec chiffres exacts, sans crochets"""
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 600,
                }
            },
            timeout=180
        )
        response.raise_for_status()
        return response.json()["response"]