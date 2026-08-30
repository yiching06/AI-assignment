import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from ai_assignment.core.constants import NLTK_DATA_DIR


nltk.data.path.insert(0, str(NLTK_DATA_DIR))


NEGATION_REPLACEMENTS = {
    "can't": "can not",
    "cannot": "can not",
    "cant": "can not",
    "couldn't": "could not",
    "couldnt": "could not",
    "didn't": "did not",
    "didnt": "did not",
    "doesn't": "does not",
    "doesnt": "does not",
    "don't": "do not",
    "dont": "do not",
    "hadn't": "had not",
    "hadnt": "had not",
    "hasn't": "has not",
    "hasnt": "has not",
    "haven't": "have not",
    "havent": "have not",
    "isn't": "is not",
    "isnt": "is not",
    "mustn't": "must not",
    "mustnt": "must not",
    "shouldn't": "should not",
    "shouldnt": "should not",
    "wasn't": "was not",
    "wasnt": "was not",
    "weren't": "were not",
    "werent": "were not",
    "won't": "will not",
    "wont": "will not",
    "wouldn't": "would not",
    "wouldnt": "would not",
}


def ensure_nltk_data():
    NLTK_DATA_DIR.mkdir(exist_ok=True)
    packages = {
        "stopwords": ["corpora/stopwords", "corpora/stopwords.zip"],
        "punkt": ["tokenizers/punkt", "tokenizers/punkt.zip"],
        "punkt_tab": ["tokenizers/punkt_tab", "tokenizers/punkt_tab.zip"],
        "wordnet": ["corpora/wordnet", "corpora/wordnet.zip"],
        "omw-1.4": ["corpora/omw-1.4", "corpora/omw-1.4.zip"],
        "opinion_lexicon": [
            "corpora/opinion_lexicon",
            "corpora/opinion_lexicon.zip",
        ],
        "vader_lexicon": [
            "sentiment/vader_lexicon.zip",
            "sentiment/vader_lexicon",
        ],
    }

    for package, resources in packages.items():
        resource_exists = False
        for resource in resources:
            try:
                nltk.data.find(resource)
                resource_exists = True
                break
            except LookupError:
                continue

        if not resource_exists:
            print(f"Downloading NLTK package: {package}")
            nltk.download(package, download_dir=str(NLTK_DATA_DIR), quiet=True)


def build_stop_words():
    stop_words = set(stopwords.words("english"))
    words_to_keep = [
        "not",
        "no",
        "nor",
        "but",
        "however",
        "though",
        "although",
        "yet",
        "doesn",
        "too",
        "isn",
        "wasn",
        "shouldn",
        "wouldn",
        "couldn",
        "won",
        "can",
        "didn",
        "don",
        "aren",
        "haven",
        "hasn",
        "hadn",
    ]

    for word in words_to_keep:
        stop_words.discard(word)

    return stop_words


def expand_negations(text):
    text = str(text).lower()
    for original, replacement in NEGATION_REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(original)}\b", replacement, text)

    return text


def clean_review(text, lemmatizer, stop_words):
    text = expand_negations(text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = word_tokenize(text)
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    return " ".join(cleaned_words)


def normalize_short_response(text):
    text = expand_negations(text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()
