from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
from mistralai import Mistral


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
MODELS_DIR = PROJECT_ROOT / "models"

MISTRAL_MODEL_NAME = "mistral-small-latest"
DEFAULT_BASENAME = "amazon_reviews_multi_fr"


def ensure_directories() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)


def load_dotenv_if_present() -> None:
    """
    Simple .env loader (only for current process).
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        # Do not override already-set environment variables
        os.environ.setdefault(key, value)


def get_mistral_client() -> Mistral:
    """
    Init Mistral client using MISTRAL_API_KEY or MISTRAL_SECRET_KEY.
    """
    load_dotenv_if_present()

    api_key = os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_SECRET_KEY")
    if not api_key:
        raise RuntimeError(
            "Mistral API key not found. "
            "Set MISTRAL_API_KEY or MISTRAL_SECRET_KEY in your environment or .env file."
        )

    return Mistral(api_key=api_key)


def load_processed_data(basename: str = DEFAULT_BASENAME) -> pd.DataFrame:
    csv_path = PROCESSED_DIR / f"{basename}_processed.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Preprocessed file not found: {csv_path}. "
            "Run data_preprocessing.py first."
        )
    df = pd.read_csv(csv_path)
    if "clean_text" in df.columns:
        df["clean_text"] = df["clean_text"].fillna("")
    return df


def load_classifier_and_vectorizer(
    basename: str = DEFAULT_BASENAME,
):
    model_path = MODELS_DIR / f"{basename}_logreg.joblib"
    vect_path = MODELS_DIR / f"{basename}_tfidf.joblib"

    if not model_path.exists() or not vect_path.exists():
        raise FileNotFoundError(
            f"Model or vectorizer not found in {MODELS_DIR}. "
            "Run train_classifier.py first."
        )

    clf = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)
    return clf, vectorizer


def add_predictions(
    df: pd.DataFrame,
    clf,
    vectorizer,
    text_col: str = "clean_text",
) -> pd.DataFrame:
    df = df.copy()
    texts = df[text_col].fillna("")
    X_vec = vectorizer.transform(texts)
    df["pred_sentiment_id"] = clf.predict(X_vec)
    id_to_label = {0: "negatif", 1: "neutre", 2: "positif"}
    df["pred_sentiment"] = df["pred_sentiment_id"].map(id_to_label)
    return df


def build_mistral_messages(review_text: str) -> List[Dict[str, str]]:
    """
    Build the prompt to generate a reply to a negative review.
    """
    system_content = (
        "Tu es un conseiller du service client d'un site e-commerce francophone. "
        "Tu rediges des reponses courtes, professionnelles et empathiques aux avis clients. "
        "Ta reponse doit : 1) reconnaitre le probleme decrit, 2) presenter des excuses, "
        "3) proposer une solution concrete (remboursement, remplacement, contact support), "
        "4) rester en francais et en 5 a 8 phrases maximum."
    )

    user_content = (
        "Voici un avis client negatif :\n"
        f"\"{review_text}\"\n\n"
        "Redige une reponse adaptee en francais, adressee directement au client, "
        "en utilisant un ton poli et empathique, sans inventer de details factuellement faux."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def generate_mistral_response(
    client: Mistral,
    review_text: str,
) -> Optional[str]:
    try:
        messages = build_mistral_messages(review_text)
        response = client.chat.complete(
            model=MISTRAL_MODEL_NAME,
            messages=messages,
        )
        message = response.choices[0].message
        # Newer mistralai SDK: message has .content, possibly a string or list.
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
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
        # Fallback for older dict-like shapes
        if isinstance(message, dict):
            maybe = message.get("content")
            if isinstance(maybe, str):
                return maybe
        return str(content) if content is not None else None
    except Exception as exc:
        print(f"Error while generating response: {exc}")
        return None


def build_final_dataframe(
    df_with_pred: pd.DataFrame,
    client: Mistral,
    max_responses: int = 200,
) -> pd.DataFrame:
    """
    Build final dataframe with:
    - id, text, true and predicted sentiment
    - email (placeholder)
    - Mistral-generated response
    """
    negative_df = df_with_pred[df_with_pred["pred_sentiment_id"] == 0].copy()
    if max_responses is not None:
        negative_df = negative_df.head(max_responses)

    rows = []
    for row in negative_df.itertuples(index=False):
        review_id = getattr(row, "id")
        text = getattr(row, "text")
        sentiment = getattr(row, "sentiment", None)
        pred_sentiment = getattr(row, "pred_sentiment", None)

        response_text = generate_mistral_response(client, text)

        # Placeholder email: to be replaced by a real column if available
        email = f"client_{review_id}@example.com"

        rows.append(
            {
                "id": review_id,
                "text": text,
                "sentiment": sentiment,
                "pred_sentiment": pred_sentiment,
                "email": email,
                "generated_response": response_text,
            }
        )

    final_df = pd.DataFrame(rows)
    return final_df


def save_final_dataframe(
    final_df: pd.DataFrame,
    basename: str = DEFAULT_BASENAME,
) -> None:
    ensure_directories()
    csv_path = FINAL_DIR / f"{basename}_with_responses.csv"
    jsonl_path = FINAL_DIR / f"{basename}_with_responses.jsonl"

    final_df.to_csv(csv_path, index=False)
    final_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    print(f"Final dataframe saved to: {csv_path}")
    print(f"Final dataframe (JSONL) saved to: {jsonl_path}")


def main() -> None:
    df = load_processed_data()
    clf, vectorizer = load_classifier_and_vectorizer()

    df_with_pred = add_predictions(df, clf, vectorizer)
    client = get_mistral_client()

    final_df = build_final_dataframe(df_with_pred, client, max_responses=200)
    save_final_dataframe(final_df)


if __name__ == "__main__":
    main()

