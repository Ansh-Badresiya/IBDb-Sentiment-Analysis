from pathlib import Path

import joblib
import streamlit as st
from text_utils import tokenize_review


BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "artifacts" / "sentiment_bundle.joblib"
@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found at {BUNDLE_PATH}. Run train_model.py first."
        )
    return joblib.load(BUNDLE_PATH)


def predict_sentiment(model, vectorizer, review: str) -> tuple[str, float]:
    transformed = vectorizer.transform([review])
    prediction = model.predict(transformed)[0]
    probability = float(model.predict_proba(transformed)[0][prediction])
    label = "Positive" if prediction == 1 else "Negative"
    return label, probability


def main():
    st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="centered")

    st.markdown("""
    <style>

    /* Main container */
    .main .block-container {
        max-width: 85%;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Title */
    h1 {
        font-size: 3rem !important;
        font-weight: 700 !important;
    }

    /* Normal text */
    p, label, div {
        font-size: 1.15rem !important;
    }

    /* Text area */
    textarea {
        font-size: 1.1rem !important;
    }

    /* Buttons */
    .stButton button {
        font-size: 1.1rem !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 10px !important;
    }

    /* Expander title */
    details summary p {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }

    /* Code blocks */
    pre {
        font-size: 1rem !important;
    }

    /* Success/Error messages */
    [data-testid="stAlert"] {
        font-size: 1.1rem !important;
    }

    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    # IMDb Movie Review Sentiment Analysis

    Analyze IMDb movie reviews and predict their sentiment using the Naive Bayes machine learning algorithm.
    """)

    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    accuracy = bundle["accuracy"]
    sample_count = bundle["sample_count"]

    col1, col2 = st.columns(2)

    card_style = """
    background: #111827;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #374151;
    text-align: center;
    """

    with col1:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:20px;">Training Samples</div>
            <div style="font-size:42px;font-weight:bold;">{sample_count:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:20px;">Test Accuracy</div>
            <div style="font-size:42px;font-weight:bold;">{accuracy:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    review = st.text_area(
        "Paste a review here",
        height=220,
        placeholder="This movie was surprisingly good. The acting was great and the story kept me engaged.",
    )

    if st.button("Predict sentiment", type="primary"):
        if not review.strip():
            st.error("Please enter a review before predicting.")
            return

        label, confidence = predict_sentiment(model, vectorizer, review)
        if label=='Positive':
            st.success(f"Predicted sentiment: {label}")
        elif label=='Negative':
            st.error(f"Predicted sentiment: {label}")
        st.progress(confidence)
        st.write(f"Confidence: {confidence:.2%}")

    positive_review = "This movie was absolutely fantastic. The storyline was engaging, the acting was outstanding, and the direction was excellent. Every scene felt meaningful and kept me interested throughout the film. The characters were well-developed, and the emotional moments were portrayed beautifully. The cinematography and background music enhanced the overall experience. I highly recommend this movie to anyone looking for quality entertainment. It is definitely one of the best films I have watched recently"

    negative_review = "I was very disappointed with this movie. The plot was confusing and lacked originality, while the pacing was extremely slow. Most of the characters were poorly developed, making it difficult to connect with them emotionally. The acting felt forced in many scenes, and several parts of the story seemed unnecessary. Despite having a promising concept, the movie failed to deliver an engaging experience. I would not recommend it and believe it does not live up to the expectations created by its rating."

    with st.expander("Sample Reviews"):
        st.markdown("Positive Review")
        st.code(positive_review)

        st.markdown("Negative Review")
        st.code(negative_review)


if __name__ == "__main__":
    main()
