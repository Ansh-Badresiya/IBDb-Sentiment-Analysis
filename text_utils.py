import re
from html import unescape

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


STEMMER = PorterStemmer()
STOP_WORDS = ENGLISH_STOP_WORDS


def tokenize_review(text: str) -> list[str]:
    text = unescape(str(text))
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return [STEMMER.stem(word) for word in text.split() if word not in STOP_WORDS]
