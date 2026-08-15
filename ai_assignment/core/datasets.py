from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

try:
    import kagglehub
except ImportError:
    kagglehub = None

import pandas as pd
from nltk.stem import WordNetLemmatizer

from ai_assignment.core.constants import (
    EXTERNAL_DATA_DIR,
    EXTERNAL_DATASETS,
    KAGGLE_DATASETS,
    SENTIMENT_SCORES,
)
from ai_assignment.core.preprocessing import build_stop_words, clean_review


def download_kaggle_dataset(dataset_id):
    if kagglehub is None:
        raise ImportError(
            "kagglehub is not installed. Run: pip install kagglehub"
        )

    try:
        dataset_dir = Path(kagglehub.dataset_download(dataset_id))
        return dataset_dir
    except Exception as error:
        cached_dir = find_cached_kaggle_dataset(dataset_id)
        if cached_dir is not None:
            return cached_dir
        raise error


def find_cached_kaggle_dataset(dataset_id):
    owner, dataset = dataset_id.split("/", maxsplit=1)
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


def download_external_zip_dataset(dataset_name, dataset_url):
    dataset_dir = EXTERNAL_DATA_DIR / slugify(dataset_name)
    zip_path = dataset_dir / "dataset.zip"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if list(dataset_dir.rglob("*.csv")):
        return dataset_dir

    urllib.request.urlretrieve(dataset_url, zip_path)

    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(dataset_dir)

    return dataset_dir


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def find_dataset_file(dataset_dir):
    dataset_dir = Path(dataset_dir)
    candidates = (
        list(dataset_dir.rglob("*.xml"))
        + list(dataset_dir.rglob("*.tsv"))
        + list(dataset_dir.rglob("*.csv"))
    )
    restaurant_candidates = [
        path
        for path in candidates
        if "restaurant" in path.name.lower()
        and "laptop" not in path.name.lower()
    ]
    if restaurant_candidates:
        train_candidates = [
            path for path in restaurant_candidates if "train" in path.name.lower()
        ]
        return sorted(train_candidates or restaurant_candidates)[0]

    if not candidates:
        raise FileNotFoundError(
            f"No XML, CSV, or TSV dataset file found in {dataset_dir}"
        )

    return candidates[0]


def read_reviews_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    if dataset_path.suffix.lower() == ".xml":
        return read_semeval_restaurant_dataset(dataset_path)

    separator = "\t" if dataset_path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(dataset_path, sep=separator)

    review_column = find_first_column(
        df,
        ["Review", "review", "review_text", "text", "comment", "clean_text"],
    )
    if review_column is None:
        raise ValueError(
            "Dataset must contain a review text column, such as 'Review', "
            "'review_text', or 'text'."
        )

    df = df.rename(columns={review_column: "Review"})

    rating_column = find_first_column(
        df,
        ["Rating", "rating", "stars", "star_rating", "stars_review", "review_stars"],
    )
    liked_column = find_first_column(df, ["Liked", "liked"])
    sentiment_column = find_first_column(
        df,
        [
            "Sentiment",
            "sentiment",
            "sentiment_label",
            "polarity",
            "rating_review",
            "label",
        ],
    )

    if rating_column is not None:
        if rating_column != "Rating":
            df = df.rename(columns={rating_column: "Rating"})
        ratings = pd.to_numeric(df["Rating"], errors="coerce")
        df = df.assign(Sentiment=ratings.apply(rating_to_sentiment))
    elif liked_column is not None:
        df = df.assign(
            Sentiment=df[liked_column].map(
                {
                    0: "Negative",
                    0.5: "Neutral",
                    1: "Positive",
                }
            )
        )
    elif sentiment_column is not None:
        df = df.assign(Sentiment=map_sentiment_column(df[sentiment_column]))
    else:
        raise ValueError(
            "Dataset must contain a sentiment label, rating, or liked column."
        )

    df = df.dropna(subset=["Review", "Sentiment"])
    df = df.assign(SentimentScore=df["Sentiment"].map(SENTIMENT_SCORES))

    return df.dropna(subset=["SentimentScore"])


def find_first_column(df, possible_names):
    lookup = {column.lower().strip(): column for column in df.columns}
    for possible_name in possible_names:
        column = lookup.get(possible_name.lower())
        if column is not None:
            return column

    return None


def map_sentiment_column(series):
    values = series.dropna()
    numeric_values = pd.to_numeric(values, errors="coerce")

    if numeric_values.notna().all() and not numeric_values.empty:
        unique_values = set(numeric_values.unique())
        if unique_values.issubset({0, 0.5, 1}):
            return pd.to_numeric(series, errors="coerce").map(
                {
                    0: "Negative",
                    0.5: "Neutral",
                    1: "Positive",
                }
            )
        if unique_values.issubset({1, 2}):
            return pd.to_numeric(series, errors="coerce").map(
                {
                    1: "Negative",
                    2: "Positive",
                }
            )

    return series.apply(normalize_sentiment_label)


def normalize_sentiment_label(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()
    sentiment_map = {
        "negative": "Negative",
        "neg": "Negative",
        "neutral": "Neutral",
        "neu": "Neutral",
        "mixed": "Neutral",
        "conflict": "Neutral",
        "positive": "Positive",
        "pos": "Positive",
    }

    return sentiment_map.get(value)


def read_semeval_restaurant_dataset(dataset_path):
    tree = ET.parse(dataset_path)
    rows = []

    for sentence in tree.findall(".//sentence"):
        text_element = sentence.find("text")
        review = (
            text_element.text.strip()
            if text_element is not None and text_element.text
            else ""
        )
        polarities = extract_semeval_polarities(sentence)
        sentiment = semeval_polarities_to_sentiment(polarities)

        if review and sentiment is not None:
            rows.append(
                {
                    "Review": review,
                    "Sentiment": sentiment,
                    "SentimentScore": SENTIMENT_SCORES[sentiment],
                    "AspectPolarities": ", ".join(polarities),
                    "SourceFile": dataset_path.name,
                }
            )

    if not rows:
        raise ValueError(
            f"No labelled restaurant review sentences found in {dataset_path}"
        )

    return pd.DataFrame(rows)


def extract_semeval_polarities(sentence):
    polarities = []
    for element in sentence.findall(".//*[@polarity]"):
        polarity = element.attrib.get("polarity", "").strip().lower()
        if polarity:
            polarities.append(polarity)

    return sorted(set(polarities))


def semeval_polarities_to_sentiment(polarities):
    polarity_set = set(polarities)
    if not polarity_set:
        return None
    if "conflict" in polarity_set:
        return "Neutral"
    if {"positive", "negative"}.issubset(polarity_set):
        return "Neutral"
    if "neutral" in polarity_set and not polarity_set.intersection(
        {"positive", "negative"}
    ):
        return "Neutral"
    if polarity_set == {"positive"}:
        return "Positive"
    if polarity_set == {"negative"}:
        return "Negative"

    return "Neutral"


def rating_to_sentiment(rating):
    if pd.isna(rating):
        return None
    if rating < 2.5:
        return "Negative"
    if rating < 4:
        return "Neutral"
    return "Positive"


def load_dataset(dataset_name, dataset_id):
    dataset_dir = download_kaggle_dataset(dataset_id)
    dataset_path = find_dataset_file(dataset_dir)
    df = read_reviews_dataset(dataset_path)
    return df.assign(DatasetSource=dataset_name)


def load_external_dataset(dataset_name, dataset_url):
    dataset_dir = download_external_zip_dataset(dataset_name, dataset_url)
    dataset_path = find_dataset_file(dataset_dir)
    df = read_reviews_dataset(dataset_path)
    return df.assign(DatasetSource=dataset_name)


def load_combined_dataset(): #combine 3 datasets into 1 central dataset
    dataframes = []
    for dataset_name, dataset_id in KAGGLE_DATASETS.items():
        dataframes.append(load_dataset(dataset_name, dataset_id))

    for dataset_name, dataset_url in EXTERNAL_DATASETS.items():
        dataframes.append(load_external_dataset(dataset_name, dataset_url))

    df = pd.concat(dataframes, ignore_index=True)

    return df.drop_duplicates(subset=["Review", "Sentiment"])


def load_and_clean_dataset():
    df = load_combined_dataset()

    lemmatizer = WordNetLemmatizer()
    stop_words = build_stop_words()
    df["cleaned_review"] = df["Review"].apply(
        lambda review: clean_review(review, lemmatizer, stop_words)
    )

    return df, lemmatizer, stop_words
