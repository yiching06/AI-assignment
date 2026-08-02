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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "restaurant_reviews.tsv"
NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"
KAGGLE_DATASET = "joebeachcapital/restaurant-reviews"

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


def run_basic_demo():
    print("\nBASIC NAIVE BAYES DEMO")
    print("-" * 30)

    demo_data = {
        "review": [
            "The food was absolutely amazing and the service was great.",
            "Terrible experience. The soup was cold and the waiter was rude.",
            "I loved the pasta, it tasted very authentic.",
            "Worst meal of my life. Will never go back.",
        ],
        "sentiment": ["positive", "negative", "positive", "negative"],
    }

    demo_df = pd.DataFrame(demo_data)
    stop_words_list = stopwords.words("english")
    model = make_pipeline(
        TfidfVectorizer(stop_words=stop_words_list),
        MultinomialNB(),
    )

    model.fit(demo_df["review"], demo_df["sentiment"])

    test_review = ["The waiter was awful but the food was okay."]
    prediction = model.predict(test_review)

    print(demo_df)
    print(f"\nReview: {test_review[0]}")
    print(f"Predicted Sentiment: {prediction[0].upper()}")


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

    if "Liked" not in df.columns:
        if "Rating" not in df.columns:
            raise ValueError("Dataset must contain either 'Liked' or 'Rating'.")

        ratings = pd.to_numeric(df["Rating"], errors="coerce")
        df = df.assign(Liked=ratings.where(ratings.isna(), (ratings >= 4).astype(int)))
        df = df[ratings.le(2) | ratings.ge(4)]

    return df.dropna(subset=["Review", "Liked"])


def load_and_clean_dataset(dataset_path=None, use_kaggle=False):
    if use_kaggle:
        dataset_dir = download_kaggle_dataset()
        dataset_path = find_dataset_file(dataset_dir)
    else:
        dataset_path = Path(dataset_path) if dataset_path else DATASET_PATH

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = read_reviews_dataset(dataset_path)

    lemmatizer = WordNetLemmatizer()
    stop_words = build_stop_words()
    df["cleaned_review"] = df["Review"].apply(
        lambda review: clean_review(review, lemmatizer, stop_words)
    )

    return df, lemmatizer, stop_words


def train_sentiment_model(df, verbose=True):
    vectorizer = TfidfVectorizer(max_features=2500)
    features = vectorizer.fit_transform(df["cleaned_review"]).toarray()
    labels = df["Liked"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
    )

    if verbose:
        print("\nDATASET MODEL")
        print("-" * 30)
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Training data shape: {x_train.shape}")
        print(f"Testing data shape: {x_test.shape}")

    svm_model = SVC(kernel="linear", random_state=42)

    if verbose:
        print("\nTraining the SVM model...")
    svm_model.fit(x_train, y_train)

    if verbose:
        print("Testing the model...")
    predictions = svm_model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    if verbose:
        print("-" * 30)
        print(f"Final Accuracy: {accuracy * 100:.2f}%\n")
        print("Detailed Classification Report:")
        print(classification_report(y_test, predictions))

        results_df = pd.DataFrame(
            {
                "Review": df.loc[y_test.index, "Review"],
                "Actual_Sentiment": y_test.values,
                "Predicted_Sentiment": predictions,
            }
        )

        label_map = {1: "Positive", 0: "Negative"}
        results_df["Actual_Sentiment"] = results_df["Actual_Sentiment"].map(label_map)
        results_df["Predicted_Sentiment"] = results_df["Predicted_Sentiment"].map(label_map)

        print("Sample Predictions:")
        print(results_df.head(10).to_string(index=False))

    return svm_model, vectorizer


def predict_sentiment(custom_review, svm_model, vectorizer, lemmatizer, stop_words):
    if not custom_review or custom_review.strip() == "":
        raise ValueError("Input cannot be empty.")

    cleaned_text = clean_review(custom_review, lemmatizer, stop_words)
    vectorized_text = vectorizer.transform([cleaned_text]).toarray()
    prediction = svm_model.predict(vectorized_text)[0]
    label = "Positive" if prediction == 1 else "Negative"

    return label, cleaned_text


def test_my_ai(custom_review, svm_model, vectorizer, lemmatizer, stop_words):
    print("\n" + "=" * 55)
    print("LIVE SENTIMENT ANALYSIS SYSTEM")
    print("=" * 55)

    if not custom_review or custom_review.strip() == "":
        print("VALIDATION ERROR: Input cannot be empty.")
        print("Please provide a valid text review to analyze.")
        print("=" * 55)
        return

    try:
        label, _ = predict_sentiment(
            custom_review,
            svm_model,
            vectorizer,
            lemmatizer,
            stop_words,
        )

        print(f'Target Text  : "{custom_review}"')
        if label == "Positive":
            print("Prediction   : POSITIVE")
        else:
            print("Prediction   : NEGATIVE")
    except Exception as error:
        print(f"SYSTEM ERROR: An unexpected fault occurred: {error}")

    print("=" * 55)


def main():
    ensure_nltk_data()
    print("Libraries and NLTK data loaded successfully.")

    run_basic_demo()

    df, lemmatizer, stop_words = load_and_clean_dataset()
    print("\nCleaned Review Examples:")
    print(df[["Review", "cleaned_review"]].head().to_string(index=False))

    svm_model, vectorizer = train_sentiment_model(df)

    test_my_ai(
        "The food was absolutely delicious and the service was amazing!",
        svm_model,
        vectorizer,
        lemmatizer,
        stop_words,
    )
    test_my_ai(
        "I waited an hour for my meal and it was completely cold. Never coming back.",
        svm_model,
        vectorizer,
        lemmatizer,
        stop_words,
    )
    test_my_ai("    ", svm_model, vectorizer, lemmatizer, stop_words)


if __name__ == "__main__":
    main()
