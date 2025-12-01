from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Iterable

import nltk
import pandas as pd
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from unidecode import unidecode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_nltk_resources() -> None:
    for resource in ("punkt", "punkt_tab", "stopwords", "wordnet"):
        nltk.download(resource, quiet=True)


def get_stopwords(language: str = "french") -> set[str]:
    sw = set(stopwords.words(language))
    keep = {"pas", "jamais", "rien", "aucun", "aucune", "plus", "sans", "ne"}
    return sw - keep


def get_lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


_PUNCT_DIGITS_RE = re.compile(rf"[{re.escape(string.punctuation)}\d]+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_review(
    text: str,
    stop_words: Iterable[str] | None = None,
    lemmatizer: WordNetLemmatizer | None = None,
) -> str:
    """
    Nettoie une review texte (lowercase, accents, ponctuation, stopwords, lemmatisation simple).
    """
    if not isinstance(text, str):
        return ""

    if stop_words is None:
        stop_words = get_stopwords()
    if lemmatizer is None:
        lemmatizer = get_lemmatizer()

    # --- règles de nettoyage proposées ---
    texte = text
    texte = re.sub(r"<.*?>", "", texte)  # enlever tags HTML
    texte = re.sub(r"[\u200b\u200c\u200d]", "", texte)  # caractères invisibles
    texte = texte.lower().strip()
    texte = texte.translate(str.maketrans("", "", string.punctuation))
    texte = unidecode(texte)  # enlever accents
    #texte = re.sub(r"\d+", "", texte)  # chiffres
    texte = re.sub(r"€|euro|\$", "euro", texte)  # normaliser euro
    texte = re.sub(r"\s+", " ", texte)  # espaces multiples -> un seul

    # --- tokenisation + stopwords + lemmatisation ---
    try:
        tokens = word_tokenize(texte, language="french")
    except LookupError:
        tokens = texte.split()

    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]

    cleaned = " ".join(lemmas).strip()
    return cleaned



def load_raw_dataset(split: str = "train[:200000]") -> pd.DataFrame:
    dataset = load_dataset("SetFit/amazon_reviews_multi_fr", split=split)
    df = pd.DataFrame(dataset)
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "text" in df.columns:
        df = df.dropna(subset=["text"])
        df = df.drop_duplicates(subset="text")

        stop_words = get_stopwords()
        lemmatizer = get_lemmatizer()
        df["clean_text"] = df["text"].apply(clean_review, stop_words=stop_words, lemmatizer=lemmatizer)

    return df


def save_processed(df: pd.DataFrame, basename: str = "amazon_reviews_multi_fr") -> None:
    ensure_directories()

    csv_path = PROCESSED_DIR / f"{basename}_processed.csv"
    jsonl_path = PROCESSED_DIR / f"{basename}_processed.jsonl"

    df.to_csv(csv_path, index=False)
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)


def main() -> None:
    download_nltk_resources()
    ensure_directories()

    df_raw = load_raw_dataset()
    df_processed = preprocess_dataframe(df_raw)
    save_processed(df_processed)


if __name__ == "__main__":
    main()
