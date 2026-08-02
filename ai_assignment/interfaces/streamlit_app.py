import streamlit as st

from ai_assignment.core.sentiment_analyzer import (
    KAGGLE_DATASET,
    DATASET_PATH,
    ensure_nltk_data,
    load_and_clean_dataset,
    predict_sentiment,
    train_sentiment_model,
)


st.set_page_config(
    page_title="Restaurant Review Sentiment",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_model(use_kaggle):
    ensure_nltk_data()
    df, lemmatizer, stop_words = load_and_clean_dataset(use_kaggle=use_kaggle)
    svm_model, vectorizer = train_sentiment_model(df, verbose=False)
    return df, lemmatizer, stop_words, svm_model, vectorizer


def main():
    st.title("Restaurant Review Sentiment Tester")
    st.write("Train the SVM model and test a restaurant review as Positive or Negative.")

    source = st.radio(
        "Dataset source",
        ["Local TSV file", "KaggleHub latest dataset"],
        horizontal=True,
    )
    use_kaggle = source == "KaggleHub latest dataset"

    if use_kaggle:
        st.caption(f"KaggleHub dataset: `{KAGGLE_DATASET}`")
    else:
        st.caption(f"Local dataset: `{DATASET_PATH.name}`")

    try:
        with st.spinner("Loading dataset and training model..."):
            df, lemmatizer, stop_words, svm_model, vectorizer = load_model(use_kaggle)
    except Exception as error:
        st.error(f"Could not load model: {error}")
        st.stop()

    st.success("Model ready")

    col1, col2 = st.columns(2)
    col1.metric("Reviews", len(df))
    col2.metric("Features", len(vectorizer.get_feature_names_out()))

    with st.expander("Preview cleaned dataset"):
        st.dataframe(df[["Review", "Liked", "cleaned_review"]].head(10), use_container_width=True)

    review = st.text_area(
        "Enter a restaurant review",
        value="The food was absolutely delicious and the service was amazing!",
        height=140,
    )

    if st.button("Predict Sentiment", type="primary"):
        try:
            label, cleaned_text = predict_sentiment(
                review,
                svm_model,
                vectorizer,
                lemmatizer,
                stop_words,
            )
        except ValueError:
            st.warning("Please enter a review first.")
            return
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            return

        if label == "Positive":
            st.success("Prediction: Positive")
        else:
            st.error("Prediction: Negative")

        st.caption(f"Cleaned text: `{cleaned_text}`")


if __name__ == "__main__":
    main()
