import streamlit as st

from ai_assignment.core.sentiment_analyzer import (
    KAGGLE_DATASET,
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
def load_model():
    ensure_nltk_data()
    df, lemmatizer, stop_words = load_and_clean_dataset()
    svm_model, vectorizer = train_sentiment_model(df)
    return df, lemmatizer, stop_words, svm_model, vectorizer


st.title("Restaurant Review Sentiment Tester")
st.write("Train an SVM sentiment model from the KaggleHub restaurant reviews dataset.")
st.caption(f"KaggleHub dataset: `{KAGGLE_DATASET}`")

try:
    with st.spinner("Downloading dataset and training model..."):
        df, lemmatizer, stop_words, svm_model, vectorizer = load_model()
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

st.success("Model ready")

summary = st.container(border=True)
with summary:
    col1, col2 = st.columns(2)
    col1.metric("Training reviews", len(df))
    col2.metric("TF-IDF features", len(vectorizer.get_feature_names_out()))

with st.expander("Preview cleaned dataset"):
    st.dataframe(df[["Review", "Liked", "cleaned_review"]].head(10))

with st.form("sentiment_form"):
    review = st.text_area(
        "Enter a restaurant review",
        value="The food was absolutely delicious and the service was amazing!",
        height=140,
    )
    submitted = st.form_submit_button("Predict sentiment", type="primary")

if submitted:
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
    except Exception as error:
        st.error(f"Prediction failed: {error}")
    else:
        if label == "Positive":
            st.success("Prediction: Positive")
        else:
            st.error("Prediction: Negative")

        st.caption(f"Cleaned text: `{cleaned_text}`")
