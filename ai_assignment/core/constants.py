from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"
SENTIMENT_PHRASES_PATH = PACKAGE_ROOT / "resources" / "sentiment_phrases.json"
APP_TITLE = "The Grand Restaurant Review"

ROLE_CUSTOMER = "Customer"
ROLE_RESTAURANT_OWNER = "Restaurant owner"
ROLE_DEVELOPER = "Developer"

RESTAURANT_OWNER_USERNAME = "restaurantowner"
RESTAURANT_OWNER_DISPLAY_USERNAME = "restaurantOwner"
DEVELOPER_USERNAME = "admin"

KAGGLE_DATASETS = {
    "10000 Restaurant Reviews": "joebeachcapital/restaurant-reviews",
}

SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
SENTIMENT_SCORES = {
    "Negative": 0,
    "Neutral": 0.5,
    "Positive": 1,
}
POSITIVE_REVIEW_SAMPLE_SIZE = 4500
DATASET_SAMPLE_RANDOM_STATE = 42

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
POSITIVE_SHORT_RESPONSES = {
    "amazing",
    "beautiful",
    "delicious",
    "excellent",
    "good",
    "great",
    "nice",
    "perfect",
    "pretty",
    "tasty",
    "wonderful",
}
STRONG_NEGATIVE_KEYWORDS = {
    "awful",
    "bad",
    "bland",
    "burnt",
    "disappointing",
    "disgusting",
    "filthy",
    "gross",
    "horrible",
    "inedible",
    "nasty",
    "overpriced",
    "rude",
    "shit",
    "stale",
    "tasteless",
    "terrible",
    "undercooked",
    "worst",
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
