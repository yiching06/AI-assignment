import warnings

from ai_assignment.core.constants import (
    KAGGLE_DATASETS,
    POSITIVE_REVIEW_SAMPLE_SIZE,
    SENTIMENT_LABELS,
    SENTIMENT_SCORES,
)
from ai_assignment.core.datasets import load_and_clean_dataset
from ai_assignment.core.models import (
    build_sentiment_models,
    calculate_classification_metrics,
    train_sentiment_models,
)
from ai_assignment.core.prediction import predict_sentiment
from ai_assignment.core.preprocessing import (
    build_stop_words,
    clean_review,
    ensure_nltk_data,
)
from ai_assignment.core.rules import (
    has_mixed_sentiment,
    is_neutral_review,
    is_neutral_short_response,
    is_positive_review,
)


warnings.filterwarnings("ignore")


def main():
    ensure_nltk_data()
    df, lemmatizer, stop_words = load_and_clean_dataset()
    trained_models, vectorizer, metrics_df, _ = train_sentiment_models(df)
    best_model_name = metrics_df.loc[metrics_df["Accuracy"].idxmax(), "Model"]
    label, score, cleaned_text = predict_sentiment(
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
    print(f"Score: {score}")
    print(f"Cleaned text: {cleaned_text}")


if __name__ == "__main__":
    main()
