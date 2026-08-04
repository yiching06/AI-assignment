import pandas as pd
import streamlit as st

from ai_assignment.core.sentiment_analyzer import (
    KAGGLE_DATASET,
    SENTIMENT_LABELS,
    ensure_nltk_data,
    load_and_clean_dataset,
    predict_sentiment,
    train_sentiment_models,
)


st.set_page_config(
    page_title="Restaurant Review Sentiment",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_models():
    ensure_nltk_data()
    df, lemmatizer, stop_words = load_and_clean_dataset()
    trained_models, vectorizer, metrics_df = train_sentiment_models(df)
    return df, lemmatizer, stop_words, trained_models, vectorizer, metrics_df


st.title("Restaurant Review Sentiment Tester")
st.write(
    "Train TF-IDF based NLP models from the KaggleHub restaurant reviews dataset."
)
st.caption(f"KaggleHub dataset: `{KAGGLE_DATASET}`")

try:
    with st.spinner("Downloading dataset and training NLP models..."):
        df, lemmatizer, stop_words, trained_models, vectorizer, metrics_df = (
            load_models()
        )
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

st.success("Models ready")

customer_tab, owner_tab = st.tabs(["Customer review", "Restaurant owner"])

with customer_tab:
    st.subheader("Test a customer review")

    with st.form("sentiment_form"):
        selected_model_name = st.selectbox(
            "Choose NLP model",
            options=list(trained_models.keys()),
        )
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
                trained_models[selected_model_name],
                vectorizer,
                lemmatizer,
                stop_words,
            )
        except ValueError:
            st.warning("Please enter a review first.")
        except Exception as error:
            st.error(f"Prediction failed: {error}")
        else:
            if label == "Negative":
                st.error("Prediction: Negative")
            elif label == "Neutral":
                st.info("Prediction: Neutral")
            else:
                st.success("Prediction: Positive")

            st.caption(f"Model used: `{selected_model_name}`")
            st.caption(f"Cleaned text: `{cleaned_text}`")

with owner_tab:
    st.subheader("Restaurant owner dashboard")
    best_model_row = metrics_df.loc[metrics_df["Accuracy"].idxmax()]

    with st.container(horizontal=True):
        st.metric("Training reviews", len(df), border=True)
        st.metric(
            "TF-IDF features",
            len(vectorizer.get_feature_names_out()),
            border=True,
        )
        st.metric("NLP models", len(trained_models), border=True)
        st.metric(
            "Best accuracy",
            best_model_row["Model"],
            f"{best_model_row['Accuracy']:.2%}",
            border=True,
        )

    sentiment_counts = (
        df["Sentiment"]
        .value_counts()
        .reindex(SENTIMENT_LABELS, fill_value=0)
        .rename_axis("Sentiment")
        .reset_index(name="Reviews")
    )

    with st.container(border=True):
        st.subheader("Review sentiment distribution")
        st.bar_chart(
            sentiment_counts,
            x="Sentiment",
            y="Reviews",
            horizontal=True,
        )

    metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metrics_display = metrics_df.copy()
    metrics_display[metric_columns] = (
        metrics_display[metric_columns]
        .mul(100)
        .round(2)
    )

    with st.container(border=True):
        st.subheader("Model evaluation comparison")
        st.write(
            "Compare SVM, Naive Bayes, and Logistic Regression using weighted "
            "Accuracy, Precision, Recall, and F1 Score."
        )
        st.bar_chart(
            metrics_display,
            x="Model",
            y=metric_columns,
            y_label="Score (%)",
        )
        st.dataframe(metrics_display)

    with st.expander("Preview cleaned dataset"):
        st.dataframe(df[["Review", "Rating", "Sentiment", "cleaned_review"]].head(10))
