from ai_assignment.core.constants import SENTIMENT_SCORES
from ai_assignment.core.preprocessing import clean_review
from ai_assignment.core.rules import (
    is_negative_review,
    is_neutral_review,
    is_positive_review,
)


def predict_sentiment(custom_review, model, vectorizer, lemmatizer, stop_words):
    if not custom_review or custom_review.strip() == "":
        raise ValueError("Input cannot be empty.")

    cleaned_text = clean_review(custom_review, lemmatizer, stop_words)
    if is_neutral_review(custom_review, cleaned_text):
        return "Neutral", SENTIMENT_SCORES["Neutral"], cleaned_text

    if is_positive_review(custom_review, cleaned_text):
        return "Positive", SENTIMENT_SCORES["Positive"], cleaned_text

    if is_negative_review(custom_review, cleaned_text):
        return "Negative", SENTIMENT_SCORES["Negative"], cleaned_text

    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    label = str(prediction)
    score = SENTIMENT_SCORES[label]

    return label, score, cleaned_text
