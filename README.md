# 📰 Fake News Detection using NLP & Machine Learning

An end-to-end Natural Language Processing project that classifies news articles
as **FAKE** or **REAL**. It includes a reusable preprocessing pipeline, two
trained models (Logistic Regression & Multinomial Naive Bayes), an exploratory
Jupyter notebook, and an interactive **Streamlit** web app with explainable
predictions.

---

## 🎯 Project Overview

| Stage | Description |
|-------|-------------|
| **Preprocessing** | Clean & normalize text with NLTK (lowercasing, URL/HTML/punctuation/number removal, stopword removal, lemmatization). |
| **Features** | TF-IDF vectorization (`max_features=50000`, `ngram_range=(1,2)`, `sublinear_tf=True`). |
| **Models** | Logistic Regression (`C=1.0`) and Multinomial Naive Bayes — the best (by F1) is saved. |
| **App** | A Streamlit UI that predicts FAKE/REAL, shows a confidence bar, and explains the result with the most influential words (Plotly). |

The model is trained on the **WELFake** dataset (`title`, `text`, `label` where
`0 = fake`, `1 = real`).

---

## 📁 Project Structure

```
fake-news-detector/
├── data/                       # Place WELFake_Dataset.csv here
├── notebooks/
│   └── fake_news_analysis.ipynb
├── models/                     # Saved model + vectorizer (.joblib)
├── app/
│   └── streamlit_app.py
├── src/
│   ├── preprocess.py           # Text cleaning + dataset loading
│   ├── train.py                # TF-IDF + model training/comparison
│   └── predict.py              # Inference API used by the app
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/rishikesh-2k6/AI-Powered-Fake-News-Detection-Using-NLP-Machine-Learning.git
cd AI-Powered-Fake-News-Detection-Using-NLP-Machine-Learning   # (or the fake-news-detector folder)

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. NLTK data

The required NLTK corpora (`stopwords`, `wordnet`, `omw-1.4`) are downloaded
**automatically** on first run by `src/preprocess.py`. To download them
manually:

```bash
python -c "import nltk; [nltk.download(p) for p in ['stopwords','wordnet','omw-1.4']]"
```

### 3. Get the dataset

Download `WELFake_Dataset.csv` (e.g. from Kaggle) and place it in the `data/`
folder:

```
data/WELFake_Dataset.csv
```

---

## 🚀 Usage

### Train the models

```bash
python -m src.train
```

This cleans the data, fits TF-IDF, trains both models, prints a comparison
table + confusion matrices, and saves the best model and vectorizer to
`models/`.

### Run the notebook

```bash
jupyter notebook notebooks/fake_news_analysis.ipynb
```

The notebook walks through EDA (class distribution, article-length histograms,
word clouds, top words per class), preprocessing, training, evaluation
(confusion-matrix heatmaps), and feature-importance analysis.

### Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Then open the URL Streamlit prints (default `http://localhost:8501`).

Quick CLI prediction:

```bash
python -m src.predict "Paste an article here to classify it"
```

---

## 📊 Expected Performance

On the full WELFake dataset, both models perform strongly; **Logistic
Regression** is typically the best, reaching roughly:

| Metric | Logistic Regression | Naive Bayes |
|--------|--------------------:|------------:|
| Accuracy | ~0.96 | ~0.92 |
| Precision | ~0.96 | ~0.92 |
| Recall | ~0.96 | ~0.92 |
| F1 | ~0.96 | ~0.92 |

> Exact numbers depend on your dataset version and train/test split.

---

## 🧰 Tech Stack

`pandas` · `numpy` · `scikit-learn` · `nltk` · `streamlit` · `plotly` ·
`wordcloud` · `matplotlib` · `seaborn` · `joblib`

---

## ⚠️ Disclaimer

This tool provides **probabilistic** predictions based on text patterns learned
from historical data. It is an educational/demonstration project and should not
be the sole basis for judging the truthfulness of any article. Always verify
information with trusted, authoritative sources.
