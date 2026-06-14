from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from text_utils import tokenize_review


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "IMDB Dataset.csv"
ARTIFACT_PATH = BASE_DIR / "artifacts" / "sentiment_bundle.joblib"
SAMPLE_SIZE = 40000
RANDOM_STATE = 42

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required_columns = {"review", "sentiment"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["sentiment"] = df["sentiment"].str.lower().map({"positive": 1, "negative": 0})
    df = df.dropna(subset=["sentiment"]).copy()
    df["sentiment"] = df["sentiment"].astype(int)
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def train_and_save():
    df = load_data()
    vectorizer = CountVectorizer(
        tokenizer=tokenize_review,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        max_features=2500,
    )
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer,
            "accuracy": accuracy,
            "sample_count": len(df),
        },
        ARTIFACT_PATH,
    )

    print(f"Saved model bundle to {ARTIFACT_PATH}")
    print(f"Training samples: {len(df)}")
    print(f"Test accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    train_and_save()
