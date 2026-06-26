"""
Streamlit web app for the Fake News Detection project.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys

import plotly.graph_objects as go
import streamlit as st

# Make the project root importable so `from src...` works regardless of the
# directory Streamlit is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import load_artifacts, predict  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

MIN_WORDS = 20  # Warn the user below this many words.


# ---------------------------------------------------------------------------
# Example articles (for the "try it" section)
# ---------------------------------------------------------------------------

EXAMPLE_ARTICLES = {
    "🔴 Sample Fake #1": (
        "SHOCKING: Scientists CONFIRM that drinking lemon water every morning "
        "completely CURES cancer in just 3 days! Big Pharma doesn't want you to "
        "know this ONE simple trick that doctors are HIDING from the public. "
        "Share before they DELETE this!!!"
    ),
    "🔴 Sample Fake #2": (
        "BREAKING: Secret government documents leaked online prove that the moon "
        "landing was filmed in a basement studio and aliens have been controlling "
        "world leaders for decades. Anonymous insiders reveal the truth THEY don't "
        "want you to see."
    ),
    "🟢 Sample Real #1": (
        "The Federal Reserve announced on Wednesday that it would hold interest "
        "rates steady, citing moderating inflation and a resilient labor market. "
        "Chair Jerome Powell said the central bank would continue to monitor "
        "economic data before making further policy adjustments."
    ),
    "🟢 Sample Real #2": (
        "NASA's Perseverance rover collected its latest rock sample from the Jezero "
        "Crater on Mars, part of an ongoing mission to study the planet's geology "
        "and search for signs of ancient microbial life. The samples are intended "
        "for return to Earth in a future mission."
    ),
}


# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_artifacts():
    """Load model + vectorizer once and cache across reruns/sessions."""
    return load_artifacts()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "article_text" not in st.session_state:
    st.session_state.article_text = ""


def _set_example(text: str) -> None:
    st.session_state.article_text = text


def _clear_text() -> None:
    st.session_state.article_text = ""


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def influential_words_chart(top_words):
    """Build a horizontal bar chart of the most influential words."""
    # Show in ascending magnitude so the strongest bar sits at the top.
    words = [w for w, _ in top_words][::-1]
    scores = [s for _, s in top_words][::-1]
    colors = ["#16a34a" if s > 0 else "#dc2626" for s in scores]  # green=real, red=fake

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=words,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top influential words (green → REAL, red → FAKE)",
        xaxis_title="Signed influence score",
        yaxis_title="",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(metadata: dict) -> None:
    with st.sidebar:
        st.header("📰 Fake News Detector")
        st.markdown(
            "An NLP + Machine Learning app that classifies news articles as "
            "**FAKE** or **REAL** using TF-IDF features and a trained classifier."
        )

        st.divider()
        st.subheader("🧠 Model")
        if metadata:
            st.markdown(f"**Algorithm:** {metadata.get('best_model_name', 'N/A')}")
            st.markdown(f"**Accuracy:** {metadata.get('accuracy', 0):.2%}")
            st.markdown(f"**F1 score:** {metadata.get('f1', 0):.2%}")
            st.markdown(f"**Features:** {metadata.get('n_features', 'N/A'):,}")
        else:
            st.info("Model metadata unavailable.")

        st.divider()
        st.subheader("📚 Dataset")
        st.markdown(
            "Trained on the **WELFake** dataset — a combined corpus of real and "
            "fake news articles (title + body), labelled `0 = fake`, `1 = real`."
        )

        st.divider()
        st.subheader("⚙️ How it works")
        st.markdown(
            "1. Clean & normalize text (NLTK)\n"
            "2. TF-IDF vectorization (uni+bigrams)\n"
            "3. Classify with the trained model\n"
            "4. Explain via influential words"
        )
        st.caption("⚠️ Predictions are probabilistic — always verify with trusted sources.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load artifacts (graceful error if the model has not been trained).
    try:
        model, vectorizer, metadata = get_artifacts()
    except FileNotFoundError:
        st.error(
            "⚠️ **Model not found.** Train it first by running:\n\n"
            "```\npython -m src.train\n```\n\n"
            "Make sure `data/WELFake_Dataset.csv` exists."
        )
        st.stop()

    render_sidebar(metadata)

    st.title("📰 Fake News Detector")
    st.markdown(
        "Paste a news article below and let the model assess whether it reads "
        "as **fake** or **real** news."
    )

    # --- Example articles ---------------------------------------------------
    with st.expander("✨ Try an example article", expanded=False):
        st.caption("Click any button to load a sample into the text box.")
        cols = st.columns(2)
        for i, (name, text) in enumerate(EXAMPLE_ARTICLES.items()):
            cols[i % 2].button(
                name,
                key=f"ex_{i}",
                use_container_width=True,
                on_click=_set_example,
                args=(text,),
            )

    # --- Input area ---------------------------------------------------------
    st.text_area(
        "News article text",
        key="article_text",
        height=240,
        placeholder="Paste the news article (headline + body) here...",
    )

    col_analyze, col_clear, _ = st.columns([1, 1, 4])
    analyze = col_analyze.button("🔍 Analyze Article", type="primary", use_container_width=True)
    col_clear.button("🗑️ Clear", use_container_width=True, on_click=_clear_text)

    if not analyze:
        return

    text = st.session_state.article_text.strip()

    # --- Edge cases ---------------------------------------------------------
    if not text:
        st.warning("⚠️ Please enter some article text before analyzing.")
        return

    word_count = len(text.split())
    if word_count < MIN_WORDS:
        st.warning(
            f"⚠️ The text is quite short ({word_count} words). Predictions on very "
            f"short text are unreliable — paste at least {MIN_WORDS} words for a "
            "more confident result."
        )

    # --- Predict ------------------------------------------------------------
    with st.spinner("Analyzing..."):
        result = predict(text, model=model, vectorizer=vectorizer, top_n=10)

    st.divider()

    # --- Result badge -------------------------------------------------------
    is_real = result.label == 1
    badge_color = "#16a34a" if is_real else "#dc2626"
    badge_emoji = "🟢" if is_real else "🔴"
    badge_text = "REAL NEWS" if is_real else "FAKE NEWS"

    st.markdown(
        f"""
        <div style="
            background-color:{badge_color};
            padding:22px;
            border-radius:12px;
            text-align:center;
            color:white;
            font-size:30px;
            font-weight:800;
            letter-spacing:1px;">
            {badge_emoji} {badge_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # --- Confidence ---------------------------------------------------------
    left, right = st.columns([2, 1])
    with left:
        st.markdown("**Confidence**")
        st.progress(min(max(result.confidence, 0.0), 1.0))
        st.caption(
            f"P(real) = {result.proba_real:.1%} | P(fake) = {result.proba_fake:.1%}"
        )
    with right:
        st.metric("Confidence", f"{result.confidence:.1%}")

    # --- Influential words --------------------------------------------------
    st.subheader("🔬 Most influential words")
    if result.top_words:
        st.plotly_chart(influential_words_chart(result.top_words), use_container_width=True)
    else:
        st.info("No recognizable vocabulary found to explain this prediction.")

    # --- Preprocessing steps ------------------------------------------------
    with st.expander("🧹 Preprocessing steps applied"):
        for step in result.steps:
            st.markdown(f"- {step}")
        st.markdown("**Cleaned text fed to the model:**")
        st.code(result.cleaned_text or "(empty after cleaning)", language="text")


if __name__ == "__main__":
    main()
