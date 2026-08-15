import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ai_assignment.core.constants import (
    NEUTRAL_EXAMPLE_WEIGHT,
    NEUTRAL_TRAINING_EXAMPLES,
)
from ai_assignment.core.preprocessing import build_stop_words, clean_review


def build_sentiment_models():
    return {
        "SVM": SVC(kernel="linear", random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }


def add_custom_neutral_training_examples(x_train, y_train):
    lemmatizer = WordNetLemmatizer()
    stop_words = build_stop_words()
    cleaned_examples = [
        clean_review(review, lemmatizer, stop_words)
        for review in NEUTRAL_TRAINING_EXAMPLES
    ]
    custom_reviews = pd.Series(
        cleaned_examples * NEUTRAL_EXAMPLE_WEIGHT,
        name="cleaned_review",
    )
    custom_labels = pd.Series(
        ["Neutral"] * len(custom_reviews),
        name="Sentiment",
    )

    augmented_reviews = pd.concat(
        [x_train.reset_index(drop=True), custom_reviews],
        ignore_index=True,
    )
    augmented_labels = pd.concat(
        [y_train.reset_index(drop=True), custom_labels],
        ignore_index=True,
    )

    return augmented_reviews, augmented_labels


def train_sentiment_models(df): #trains the sentiment models using the provided dataset and returns the trained models
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    reviews = df["cleaned_review"]
    labels = df["Sentiment"]

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        reviews,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    x_train_text, y_train = add_custom_neutral_training_examples(
        x_train_text,
        y_train,
    )
    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)

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
