"""
Text preprocessing module for the Fake News Detection project.

Loads the WELFake dataset, performs cleaning / normalization, and exposes
reusable functions so the same preprocessing is applied at train time and at
prediction time (this consistency is critical for a working ML pipeline).
"""

from __future__ import annotations

import os
import re
import string
from functools import lru_cache
from typing import Iterable

import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# ---------------------------------------------------------------------------
# NLTK resource bootstrap
# ---------------------------------------------------------------------------

# We use a RegexpTokenizer (pure-regex, no external data) instead of the
# `punkt` tokenizer to avoid the brittle punkt/punkt_tab download issues that
# vary across NLTK versions. Only `stopwords` and `wordnet` need downloading.
_REQUIRED_CORPORA = {
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
}


def ensure_nltk_resources() -> None:
    """Download the NLTK corpora we depend on if they are not already present."""
    for path, package in _REQUIRED_CORPORA.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_resources()


# ---------------------------------------------------------------------------
# Lazy singletons (built once, reused for every row — big speedup)
# ---------------------------------------------------------------------------

_TOKENIZER = RegexpTokenizer(r"[a-z]+")


@lru_cache(maxsize=1)
def _get_stopwords() -> frozenset[str]:
    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=1)
def _get_lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


# Pre-compiled regexes for the cleaning passes.
_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NUM_RE = re.compile(r"\d+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# Core cleaning
# ---------------------------------------------------------------------------

# Human-readable description of each step, surfaced in the Streamlit UI.
PREPROCESSING_STEPS = [
    "Lowercased all text",
    "Removed URLs and web links",
    "Stripped HTML tags",
    "Removed punctuation",
    "Removed numbers",
    "Tokenized into words",
    "Removed English stopwords",
    "Lemmatized tokens to base form",
]


def clean_text(text: str) -> str:
    """Clean a single document and return a normalized, space-joined string.

    Steps: lowercase -> remove URLs -> strip HTML -> remove punctuation ->
    remove numbers -> tokenize -> drop stopwords -> lemmatize.

    Args:
        text: Raw input text. Non-string / NaN values are treated as empty.

    Returns:
        The cleaned text as a single space-separated string of lemmas.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    text = _NUM_RE.sub(" ", text)

    tokens = _TOKENIZER.tokenize(text)

    stop_words = _get_stopwords()
    lemmatizer = _get_lemmatizer()

    cleaned = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) > 2
    ]
    return " ".join(cleaned)


def clean_series(texts: Iterable[str]) -> list[str]:
    """Vectorized convenience wrapper to clean an iterable of documents."""
    return [clean_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "WELFake_Dataset.csv",
)


def load_and_preprocess(
    csv_path: str = DEFAULT_DATASET_PATH,
    text_column: str = "content",
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the WELFake dataset and return a cleaned DataFrame.

    The returned DataFrame has at least:
        - ``content``: combined title + text (raw)
        - ``clean_content``: fully preprocessed text ready for TF-IDF
        - ``label``: 0 = fake, 1 = real

    Args:
        csv_path: Path to ``WELFake_Dataset.csv``.
        text_column: Name to give the combined title+text column.
        verbose: If True, print progress / shape information.

    Raises:
        FileNotFoundError: If the dataset cannot be located.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Download WELFake_Dataset.csv and place it in the data/ folder."
        )

    if verbose:
        print(f"Loading dataset from {csv_path} ...")

    df = pd.read_csv(csv_path)

    # WELFake sometimes ships an unnamed index column.
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    required = {"title", "text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    if verbose:
        print(f"Raw shape: {df.shape}")

    # Fill missing title/text with empty strings before combining, then drop
    # rows that have no label at all.
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df = df.dropna(subset=["label"])

    # Combine title + text into a single content field.
    df[text_column] = (df["title"].astype(str) + " " + df["text"].astype(str)).str.strip()

    # Drop rows where the combined content is empty, then drop duplicates.
    df = df[df[text_column].str.len() > 0]
    df = df.drop_duplicates(subset=[text_column])

    df["label"] = df["label"].astype(int)

    if verbose:
        print("Cleaning text (this can take a minute on the full dataset) ...")

    df["clean_content"] = clean_series(df[text_column].tolist())

    # Remove rows that became empty after cleaning (e.g. all stopwords).
    df = df[df["clean_content"].str.len() > 0].reset_index(drop=True)

    if verbose:
        print(f"Final cleaned shape: {df.shape}")
        print("Class distribution (0=fake, 1=real):")
        print(df["label"].value_counts().sort_index())

    return df


if __name__ == "__main__":
    # Quick smoke test of the cleaning function.
    sample = (
        "<p>BREAKING!!! Visit http://example.com NOW — "
        "Scientists discovered 12345 amazing facts about the WORLD.</p>"
    )
    print("RAW   :", sample)
    print("CLEAN :", clean_text(sample))
