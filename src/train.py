"""
Feature engineering and model training for the Fake News Detection project.

Trains and compares two classifiers (Logistic Regression and Multinomial
Naive Bayes) on TF-IDF features, prints a metric comparison table, and
persists the best model + vectorizer to ``models/`` with joblib.

Run:
    python -m src.train
    # or
    python src/train.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Allow running both as a module (`python -m src.train`) and as a script
# (`python src/train.py`).
try:
    from src.preprocess import load_and_preprocess
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.preprocess import load_and_preprocess


# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.joblib")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Label semantics for the WELFake dataset.
LABEL_NAMES = {0: "FAKE", 1: "REAL"}


@dataclass
class ModelResult:
    """Container holding a trained estimator and its evaluation metrics."""

    name: str
    model: object
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    report: str


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def build_vectorizer() -> TfidfVectorizer:
    """Create the TF-IDF vectorizer with the project's configured hyperparams."""
    return TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def _evaluate(name: str, model, X_test, y_test) -> ModelResult:
    """Compute the standard classification metrics for a fitted model."""
    preds = model.predict(X_test)
    return ModelResult(
        name=name,
        model=model,
        accuracy=accuracy_score(y_test, preds),
        precision=precision_score(y_test, preds, zero_division=0),
        recall=recall_score(y_test, preds, zero_division=0),
        f1=f1_score(y_test, preds, zero_division=0),
        confusion=confusion_matrix(y_test, preds),
        report=classification_report(
            y_test, preds, target_names=[LABEL_NAMES[0], LABEL_NAMES[1]], zero_division=0
        ),
    )


def _print_comparison(results: list[ModelResult]) -> None:
    """Pretty-print a side-by-side metric comparison table."""
    header = f"{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}"
    print("\n" + "=" * len(header))
    print("MODEL COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<22}{r.accuracy:>10.4f}{r.precision:>11.4f}"
            f"{r.recall:>9.4f}{r.f1:>9.4f}"
        )
    print("=" * len(header))


def train(csv_path: str | None = None, verbose: bool = True) -> ModelResult:
    """Run the full training pipeline and persist the best model.

    Args:
        csv_path: Optional override for the dataset location.
        verbose: Print progress and evaluation details.

    Returns:
        The :class:`ModelResult` for the best-performing model (by F1 score).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load + preprocess --------------------------------------------------
    if csv_path is None:
        df = load_and_preprocess(verbose=verbose)
    else:
        df = load_and_preprocess(csv_path=csv_path, verbose=verbose)

    X_raw = df["clean_content"].tolist()
    y = df["label"].values

    # 2. Train/test split (stratified to preserve class balance) ------------
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 3. TF-IDF vectorization ----------------------------------------------
    if verbose:
        print("\nFitting TF-IDF vectorizer ...")
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    if verbose:
        print(f"TF-IDF feature matrix: {X_train.shape}")

    # 4. Train both models --------------------------------------------------
    results: list[ModelResult] = []

    if verbose:
        print("\nTraining Logistic Regression ...")
    logreg = LogisticRegression(C=1.0, max_iter=1000)
    logreg.fit(X_train, y_train)
    results.append(_evaluate("Logistic Regression", logreg, X_test, y_test))

    if verbose:
        print("Training Multinomial Naive Bayes ...")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    results.append(_evaluate("Multinomial Naive Bayes", nb, X_test, y_test))

    # 5. Report -------------------------------------------------------------
    if verbose:
        _print_comparison(results)
        for r in results:
            print(f"\n--- {r.name} ---")
            print("Confusion matrix [rows=actual, cols=pred] (0=FAKE, 1=REAL):")
            print(r.confusion)
            print(r.report)

    # 6. Select & persist the best model (highest F1) -----------------------
    best = max(results, key=lambda r: r.f1)
    if verbose:
        print(f"\nBest model: {best.name} (F1 = {best.f1:.4f})")
        print(f"Saving artifacts to {MODELS_DIR} ...")

    joblib.dump(best.model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(
        {
            "best_model_name": best.name,
            "accuracy": best.accuracy,
            "precision": best.precision,
            "recall": best.recall,
            "f1": best.f1,
            "n_features": len(vectorizer.get_feature_names_out()),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "label_names": LABEL_NAMES,
        },
        METADATA_PATH,
    )

    if verbose:
        print("Saved:")
        print(f"  - {MODEL_PATH}")
        print(f"  - {VECTORIZER_PATH}")
        print(f"  - {METADATA_PATH}")

    return best


if __name__ == "__main__":
    train()
