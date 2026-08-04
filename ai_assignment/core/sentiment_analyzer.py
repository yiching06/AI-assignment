from pathlib import Path
import re
import warnings

try:
    import kagglehub
except ImportError:
    kagglehub = None

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"
KAGGLE_DATASET = "joebeachcapital/restaurant-reviews"
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
MODEL_NAMES = ["SVM", "Naive Bayes", "Logistic Regression"]

nltk.data.path.insert(0, str(NLTK_DATA_DIR))


def ensure_nltk_data():
    NLTK_DATA_DIR.mkdir(exist_ok=True)
    packages = {
        "stopwords": ["corpora/stopwords", "corpora/stopwords.zip"],
        "punkt": ["tokenizers/punkt", "tokenizers/punkt.zip"],
        "punkt_tab": ["tokenizers/punkt_tab", "tokenizers/punkt_tab.zip"],
        "wordnet": ["corpora/wordnet", "corpora/wordnet.zip"],
        "omw-1.4": ["corpora/omw-1.4", "corpora/omw-1.4.zip"],
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
        "doesn",
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


def clean_review(text, lemmatizer, stop_words):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = word_tokenize(text)
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    return " ".join(cleaned_words)


def download_kaggle_dataset():
    if kagglehub is None:
        raise ImportError(
            "kagglehub is not installed. Run: pip install kagglehub"
        )

    try:
        dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        return dataset_dir
    except Exception as error:
        cached_dir = find_cached_kaggle_dataset()
        if cached_dir is not None:
            return cached_dir
        raise error


def find_cached_kaggle_dataset():
    owner, dataset = KAGGLE_DATASET.split("/", maxsplit=1)
    cache_dir = (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "datasets"
        / owner
        / dataset
        / "versions"
    )

    if not cache_dir.exists():
        return None

    versions = [path for path in cache_dir.iterdir() if path.is_dir()]
    if not versions:
        return None

    return sorted(versions)[-1]


def find_dataset_file(dataset_dir):
    dataset_dir = Path(dataset_dir)
    candidates = list(dataset_dir.glob("*.tsv")) + list(dataset_dir.glob("*.csv"))

    if not candidates:
        raise FileNotFoundError(f"No CSV or TSV dataset file found in {dataset_dir}")

    return candidates[0]


def read_reviews_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    separator = "\t" if dataset_path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(dataset_path, sep=separator)

    if "Review" not in df.columns:
        raise ValueError("Dataset must contain a 'Review' column.")

    if "Rating" in df.columns:
        ratings = pd.to_numeric(df["Rating"], errors="coerce")
        df = df.assign(Sentiment=ratings.apply(rating_to_sentiment))
    elif "Liked" in df.columns:
        df = df.assign(
            Sentiment=df["Liked"].map({0: "Negative", 1: "Positive"})
        )
    else:
        raise ValueError("Dataset must contain either 'Rating' or 'Liked'.")

    return df.dropna(subset=["Review", "Sentiment"])


def rating_to_sentiment(rating):
    if pd.isna(rating):
        return None
    if rating < 2.5:
        return "Negative"
    if rating < 4:
        return "Neutral"
    return "Positive"


def load_and_clean_dataset():
    dataset_dir = download_kaggle_dataset()
    dataset_path = find_dataset_file(dataset_dir)
    df = read_reviews_dataset(dataset_path)

    lemmatizer = WordNetLemmatizer()
    stop_words = build_stop_words()
    df["cleaned_review"] = df["Review"].apply(
        lambda review: clean_review(review, lemmatizer, stop_words)
    )

    return df, lemmatizer, stop_words


def build_sentiment_models():
    return {
        "SVM": SVC(kernel="linear", random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }


def train_sentiment_models(df):
    vectorizer = TfidfVectorizer(max_features=2500)
    features = vectorizer.fit_transform(df["cleaned_review"])
    labels = df["Sentiment"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    trained_models = {}
    metrics_rows = []

    for model_name, model in build_sentiment_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        trained_models[model_name] = model
        metrics_rows.append(
            {
                "Model": model_name,
                **calculate_classification_metrics(y_test, predictions),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)

    return trained_models, vectorizer, metrics_df


def train_sentiment_model(df):
    trained_models, vectorizer, metrics_df = train_sentiment_models(df)
    svm_metrics = (
        metrics_df.loc[metrics_df["Model"] == "SVM"]
        .drop(columns="Model")
        .iloc[0]
        .to_dict()
    )

    return trained_models["SVM"], vectorizer, svm_metrics


def calculate_classification_metrics(y_test, predictions):
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


def predict_sentiment(custom_review, model, vectorizer, lemmatizer, stop_words):
    if not custom_review or custom_review.strip() == "":
        raise ValueError("Input cannot be empty.")

    cleaned_text = clean_review(custom_review, lemmatizer, stop_words)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    label = str(prediction)

    return label, cleaned_text


def main():
    ensure_nltk_data()
    df, lemmatizer, stop_words = load_and_clean_dataset()
    trained_models, vectorizer, metrics_df = train_sentiment_models(df)
    best_model_name = metrics_df.loc[metrics_df["Accuracy"].idxmax(), "Model"]
    label, cleaned_text = predict_sentiment(
        "The food was absolutely delicious and the service was amazing!",
        trained_models[best_model_name],
        vectorizer,
        lemmatizer,
        stop_words,
    )
    print(f"Dataset rows: {len(df)}")
    print("Model comparison:")
    print(metrics_df.to_string(index=False))
    print(f"Best model by accuracy: {best_model_name}")
    print(f"Prediction: {label}")
    print(f"Cleaned text: {cleaned_text}")


if __name__ == "__main__":
    main()
