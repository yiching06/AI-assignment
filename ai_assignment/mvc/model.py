from ai_assignment.core.sentiment_analyzer import (
    ensure_nltk_data,
    load_and_clean_dataset,
    predict_sentiment,
    train_sentiment_model,
)


class SentimentModel:
    def __init__(self):
        self.svm_model = None
        self.vectorizer = None
        self.lemmatizer = None
        self.stop_words = None
        self.is_ready = False

    def load(self, use_kaggle=False):
        ensure_nltk_data()
        df, self.lemmatizer, self.stop_words = load_and_clean_dataset(
            use_kaggle=use_kaggle
        )
        self.svm_model, self.vectorizer = train_sentiment_model(df, verbose=False)
        self.is_ready = True

    def predict(self, review):
        if not self.is_ready:
            raise RuntimeError("Model is not ready yet.")

        return predict_sentiment(
            review,
            self.svm_model,
            self.vectorizer,
            self.lemmatizer,
            self.stop_words,
        )
