from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_directories() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data(basename: str = "amazon_reviews_multi_fr") -> pd.DataFrame:
    """
    Load preprocessed data produced by data_preprocessing.py.
    """
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


def train_val_test_split(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "sentiment_id",
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split DataFrame into train / validation / test sets.
    """
    X = df[text_col]
    y = df[label_col]

    # first train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # then train vs val from train_val
    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_vectorizer() -> TfidfVectorizer:
    """
    Create a simple TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
    )
    return vectorizer


def train_model(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
):
    """
    Train a logistic regression classifier on TF-IDF features.
    """
    X_train = X_train.fillna("")
    X_val = X_val.fillna("")

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    clf = LogisticRegression(
        max_iter=1_000,
        n_jobs=-1,
    )
    clf.fit(X_train_vec, y_train)

    y_val_pred = clf.predict(X_val_vec)
    print("=== Validation classification report ===")
    print(classification_report(y_val, y_val_pred))

    print("=== Validation confusion matrix ===")
    labels = sorted(pd.unique(y_val))
    cm_val = confusion_matrix(y_val, y_val_pred, labels=labels)
    cm_val_df = pd.DataFrame(cm_val, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    print(cm_val_df)

    return vectorizer, clf


def evaluate_model(
    vectorizer: TfidfVectorizer,
    clf: LogisticRegression,
    X_test: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Evaluate the model on the test set.
    """
    X_test = X_test.fillna("")
    X_test_vec = vectorizer.transform(X_test)
    y_test_pred = clf.predict(X_test_vec)
    print("=== Test classification report ===")
    print(classification_report(y_test, y_test_pred))

    print("=== Test confusion matrix ===")
    labels = sorted(pd.unique(y_test))
    cm_test = confusion_matrix(y_test, y_test_pred, labels=labels)
    cm_test_df = pd.DataFrame(cm_test, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    print(cm_test_df)


def save_artifacts(
    vectorizer: TfidfVectorizer,
    clf: LogisticRegression,
    basename: str = "amazon_reviews_multi_fr",
) -> None:
    """
    Save model and vectorizer into the models/ directory.
    """
    ensure_directories()
    model_path = MODELS_DIR / f"{basename}_logreg.joblib"
    vect_path = MODELS_DIR / f"{basename}_tfidf.joblib"

    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vect_path)

    print(f"Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vect_path}")


def main() -> None:
    df = load_processed_data()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)

    vectorizer, clf = train_model(X_train, y_train, X_val, y_val)
    evaluate_model(vectorizer, clf, X_test, y_test)
    save_artifacts(vectorizer, clf)


if __name__ == "__main__":
    main()
