"""
Inference module for the Fake News Detection project.

Loads the persisted model + TF-IDF vectorizer and exposes a small API used by
both the CLI and the Streamlit app:

    - load_artifacts()        -> (model, vectorizer, metadata)
    - predict(text)           -> PredictionResult
    - top_influential_words() -> list[(word, signed_score)]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

try:
    from src.preprocess import clean_text, PREPROCESSING_STEPS
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.preprocess import clean_text, PREPROCESSING_STEPS

import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.joblib")

LABEL_NAMES = {0: "FAKE", 1: "REAL"}


@dataclass
class PredictionResult:
    """Structured output returned by :func:`predict`."""

    label: int                       # 0 = fake, 1 = real
    label_name: str                  # "FAKE" / "REAL"
    confidence: float                # probability of the predicted class (0..1)
    proba_fake: float
    proba_real: float
    cleaned_text: str
    top_words: list[tuple[str, float]] = field(default_factory=list)
    steps: list[str] = field(default_factory=lambda: list(PREPROCESSING_STEPS))


def load_artifacts(models_dir: str = MODELS_DIR):
    """Load and return ``(model, vectorizer, metadata)`` from disk.

    Raises:
        FileNotFoundError: If the model artifacts have not been trained yet.
    """
    for path in (MODEL_PATH, VECTORIZER_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model artifact missing: {path}. "
                "Train the model first with: python -m src.train"
            )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    metadata = joblib.load(METADATA_PATH) if os.path.exists(METADATA_PATH) else {}
    return model, vectorizer, metadata


def _signed_word_contributions(
    model, vectorizer, features, top_n: int = 10
) -> list[tuple[str, float]]:
    """Return the words in ``features`` that most influenced the prediction.

    Works for both Logistic Regression (``coef_``) and Multinomial Naive
    Bayes (``feature_log_prob_``). Score sign indicates direction:
    positive -> pushes toward REAL, negative -> pushes toward FAKE.
    """
    feature_names = vectorizer.get_feature_names_out()
    row = features.tocoo()
    present = {idx: val for idx, val in zip(row.col, row.data)}
    if not present:
        return []

    if hasattr(model, "coef_"):
        # Binary LogisticRegression: coef_ row is weight toward class "1" (REAL).
        weights = model.coef_[0]
        contributions = {
            idx: weights[idx] * tfidf for idx, tfidf in present.items()
        }
    elif hasattr(model, "feature_log_prob_"):
        # NB: difference of log-probabilities between REAL and FAKE classes.
        diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        contributions = {idx: diff[idx] * tfidf for idx, tfidf in present.items()}
    else:  # pragma: no cover - defensive
        return []

    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [(feature_names[idx], float(score)) for idx, score in ranked[:top_n]]


def predict(text: str, model=None, vectorizer=None, top_n: int = 10) -> PredictionResult:
    """Classify a news article as FAKE or REAL.

    Args:
        text: Raw article text (title + body is fine).
        model: Optional pre-loaded estimator (avoids re-loading from disk).
        vectorizer: Optional pre-loaded TF-IDF vectorizer.
        top_n: Number of influential words to surface.

    Returns:
        A :class:`PredictionResult`.
    """
    if model is None or vectorizer is None:
        model, vectorizer, _ = load_artifacts()

    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])

    pred = int(model.predict(features)[0])

    # Probabilities (both supported models implement predict_proba).
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        proba_fake = float(proba[0])
        proba_real = float(proba[1])
    else:  # pragma: no cover - defensive fallback
        proba_fake, proba_real = (1.0, 0.0) if pred == 0 else (0.0, 1.0)

    confidence = proba_real if pred == 1 else proba_fake
    top_words = _signed_word_contributions(model, vectorizer, features, top_n=top_n)

    return PredictionResult(
        label=pred,
        label_name=LABEL_NAMES[pred],
        confidence=confidence,
        proba_fake=proba_fake,
        proba_real=proba_real,
        cleaned_text=cleaned,
        top_words=top_words,
    )


if __name__ == "__main__":
    import sys

    article = " ".join(sys.argv[1:]) or (
        "Government officials confirmed the new infrastructure budget was "
        "approved by parliament after months of committee review."
    )
    result = predict(article)
    print(f"Prediction : {result.label_name}")
    print(f"Confidence : {result.confidence:.2%}")
    print(f"P(fake)={result.proba_fake:.2%}  P(real)={result.proba_real:.2%}")
    print("Top words  :")
    for word, score in result.top_words:
        direction = "REAL" if score > 0 else "FAKE"
        print(f"  {word:<20} {score:+.4f}  -> {direction}")
