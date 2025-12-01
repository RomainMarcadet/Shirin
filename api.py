from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from mistralai import Mistral

from notebooks.data_preprocessing import clean_review, get_lemmatizer, get_stopwords


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_BASENAME = "amazon_reviews_multi_fr"

LABEL_MAP = {0: "negatif", 1: "neutre", 2: "positif"}


def load_dotenv_if_present() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        os.environ.setdefault(key, value)


def get_mistral_client() -> Mistral:
    load_dotenv_if_present()
    api_key = os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_SECRET_KEY")
    if not api_key:
        raise RuntimeError(
            "Mistral API key not found. "
            "Set MISTRAL_API_KEY or MISTRAL_SECRET_KEY in your environment or .env file."
        )
    return Mistral(api_key=api_key)


def load_classifier_and_vectorizer():
    model_path = MODELS_DIR / f"{MODEL_BASENAME}_logreg.joblib"
    vect_path = MODELS_DIR / f"{MODEL_BASENAME}_tfidf.joblib"
    if not model_path.exists() or not vect_path.exists():
        raise FileNotFoundError(
            f"Model or vectorizer not found in {MODELS_DIR}. "
            "Run notebooks/train_classifier.py first."
        )
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)
    return clf, vectorizer


def extract_mistral_content(message) -> str:
    """
    Extract text content from a Mistral AssistantMessage, handling
    both string and list-of-parts formats.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = None
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "".join(parts)
    if isinstance(message, dict):
        maybe = message.get("content")
        if isinstance(maybe, str):
            return maybe
    return str(content) if content is not None else ""


def build_mistral_messages(review_text: str, tone: Literal["polite", "sarcasm"]) -> list[dict]:
    if tone == "sarcasm":
        system_content = (
            "Tu es un conseiller du service client d'un site e-commerce francophone, "
            "au ton sarcastique et pince-sans-rire qui varie à chaque réponse pour éviter la monotonie. "
            "Tu rédiges des réponses courtes (5-8 phrases max), professionnelles mais ironiquement empathiques aux avis clients. "
            "Ta réponse doit : "
            "1) reconnaître le problème avec une ironie dosée et inédite, "
            "2) présenter des excuses feintes avec humour varié, "
            "3) proposer une solution concrète (remboursement, remplacement, contact support), "
            "4) finir sur une note sarcastique légère et unique. "
            "Varie systématiquement les blagues : exagération, autodérision, sous-entendus ou absurde. "
            "Reste toujours poli : le sarcasme doit amuser le client agacé, pas l'énerver."
        )
    else:
        system_content = (
            "Tu es un conseiller du service client d'un site e-commerce francophone. "
            "Tu rédiges des réponses courtes, professionnelles et empathiques aux avis clients. "
            "Ta réponse doit : 1) reconnaître le problème décrit, 2) présenter des excuses, "
            "3) proposer une solution concrète (remboursement, remplacement, contact support), "
            "4) rester en français et en 5 à 8 phrases maximum."
        )

    user_content = (
        "Voici un avis client négatif :\n"
        f"\"{review_text}\"\n\n"
        "Rédige une réponse adaptée en français, adressée directement au client, "
        "en utilisant un ton poli et empathique, sans inventer de détails factuellement faux."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def predict_sentiment(text: str, clf, vectorizer, stop_words, lemmatizer) -> int:
    cleaned = clean_review(text, stop_words=stop_words, lemmatizer=lemmatizer)
    X_vec = vectorizer.transform([cleaned])
    pred_id = int(clf.predict(X_vec)[0])
    return pred_id


app = FastAPI(title="Shirin Sentiment & Response API")

_clf, _vectorizer = load_classifier_and_vectorizer()
_stop_words = get_stopwords()
_lemmatizer = get_lemmatizer()
_mistral_client = get_mistral_client()


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sentiment_id: int
    sentiment_label: str
    proba: Optional[float] = None


class RespondRequest(BaseModel):
    text: str
    tone: Literal["polite", "sarcasm"] = "polite"
    force: bool = False
    model: Optional[str] = None  # allow overriding Mistral model if needed


class RespondResponse(BaseModel):
    sentiment_id: int
    sentiment_label: str
    generated: bool
    response_text: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    pred_id = predict_sentiment(req.text, _clf, _vectorizer, _stop_words, _lemmatizer)
    label = LABEL_MAP.get(pred_id, "inconnu")

    proba_val: Optional[float] = None
    if hasattr(_clf, "predict_proba"):
        proba = _clf.predict_proba(_vectorizer.transform([clean_review(req.text, _stop_words, _lemmatizer)]))[0]
        proba_val = float(proba[pred_id])

    return PredictResponse(sentiment_id=pred_id, sentiment_label=label, proba=proba_val)


@app.post("/respond", response_model=RespondResponse)
def respond(req: RespondRequest) -> RespondResponse:
    pred_id = predict_sentiment(req.text, _clf, _vectorizer, _stop_words, _lemmatizer)
    label = LABEL_MAP.get(pred_id, "inconnu")

    # If not negative and not forced, do not generate a reply
    if not req.force and pred_id != 0:
        return RespondResponse(
            sentiment_id=pred_id,
            sentiment_label=label,
            generated=False,
            response_text=None,
        )

    messages = build_mistral_messages(req.text, req.tone)
    model_name = req.model or os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")

    chat = _mistral_client.chat.complete(
        model=model_name,
        messages=messages,
    )
    content = extract_mistral_content(chat.choices[0].message)

    return RespondResponse(
        sentiment_id=pred_id,
        sentiment_label=label,
        generated=True,
        response_text=content,
    )

