from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"
SENTIMENT_PHRASES_PATH = PACKAGE_ROOT / "resources" / "sentiment_phrases.json"

KAGGLE_DATASETS = {
    "SemEval 2014 Restaurant Reviews": "charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis",
    "10000 Restaurant Reviews": "joebeachcapital/restaurant-reviews",
}
EXTERNAL_DATASETS = {
    "Yelp Restaurant Reviews Sentiment": "https://zenodo.org/api/records/18723813/files/yelp_sentiment_master_dataset.zip/content",
}

SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
SENTIMENT_SCORES = {
    "Negative": 0,
    "Neutral": 0.5,
    "Positive": 1,
}

MIXED_SENTIMENT_CONNECTORS = {
    "but",
    "however",
    "though",
    "although",
    "yet",
}

NEUTRAL_SHORT_RESPONSES = {"ok", "okay", "okok"}
NEUTRAL_KEYWORDS = {
    "average",
    "acceptable",
    "decent",
    "fine",
    "normal",
    "moderate",
    "ordinary",
    "okay",
    "unremarkable",
}
NEUTRAL_TRAINING_EXAMPLES = [
    "ok",
    "okay",
    "okok",
    "ok ok",
    "food ok",
    "food okay",
    "service ok",
    "service okay",
    "restaurant ok",
    "restaurant okay",
    "it was ok",
    "it was okay",
    "average",
    "nothing special",
    "nothing remarkable",
    "ordinary dining experience",
    "not bad not great",
]
NEUTRAL_EXAMPLE_WEIGHT = 20
