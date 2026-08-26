import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_assignment.core.sentiment_analyzer import (
    POSITIVE_REVIEW_SAMPLE_SIZE,
    SENTIMENT_LABELS,
    ensure_nltk_data,
    load_and_clean_dataset,
    predict_sentiment,
    train_sentiment_models,
)


st.set_page_config(
    page_title="Restaurant Review Sentiment",
    layout="wide",
)


APP_DATA_DIR = Path(__file__).resolve().parent / "data"
CUSTOMER_ACCOUNTS_PATH = APP_DATA_DIR / "customer_accounts.csv"
CUSTOMER_REVIEWS_PATH = APP_DATA_DIR / "customer_reviews.csv"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_HASH = hashlib.sha256("admin123".encode("utf-8")).hexdigest()
TRAINING_DATASET_CACHE_VERSION = f"positive-cap-{POSITIVE_REVIEW_SAMPLE_SIZE}"


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def initialize_session_state():
    st.session_state.setdefault("authenticated_user", None)
    st.session_state.setdefault("authenticated_role", None)


@st.cache_resource(show_spinner=False)
def load_models(positive_review_sample_size, cache_version):
    del cache_version
    ensure_nltk_data()
    df, lemmatizer, stop_words = load_and_clean_dataset(
        positive_review_sample_size
    )
    trained_models, vectorizer, metrics_df = train_sentiment_models(df)
    return df, lemmatizer, stop_words, trained_models, vectorizer, metrics_df


def ensure_storage_files():
    APP_DATA_DIR.mkdir(exist_ok=True)
    if not CUSTOMER_ACCOUNTS_PATH.exists():
        pd.DataFrame(columns=["username", "password_hash"]).to_csv(
            CUSTOMER_ACCOUNTS_PATH,
            index=False,
        )
    if not CUSTOMER_REVIEWS_PATH.exists():
        pd.DataFrame(
            columns=[
                "submitted_at",
                "username",
                "review",
            ]
        ).to_csv(CUSTOMER_REVIEWS_PATH, index=False)


def load_customer_accounts():
    ensure_storage_files()
    return pd.read_csv(CUSTOMER_ACCOUNTS_PATH, dtype=str).fillna("")


def load_customer_reviews():
    ensure_storage_files()
    reviews_df = pd.read_csv(CUSTOMER_REVIEWS_PATH, dtype=str).fillna("")
    for column in ["submitted_at", "username", "review"]:
        if column not in reviews_df.columns:
            reviews_df[column] = ""

    return reviews_df[["submitted_at", "username", "review"]]


def save_customer_account(username, password):
    accounts_df = load_customer_accounts()
    new_account = pd.DataFrame(
        [
            {
                "username": username,
                "password_hash": hash_password(password),
            }
        ]
    )
    pd.concat([accounts_df, new_account], ignore_index=True).to_csv(
        CUSTOMER_ACCOUNTS_PATH,
        index=False,
    )


def save_customer_review(username, review):
    reviews_df = load_customer_reviews()
    new_review = pd.DataFrame(
        [
            {
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "username": username,
                "review": review.strip(),
            }
        ]
    )
    pd.concat([reviews_df, new_review], ignore_index=True).to_csv(
        CUSTOMER_REVIEWS_PATH,
        index=False,
    )


def authenticate_user(username, password):
    normalized_username = username.strip().lower()
    if normalized_username == ADMIN_USERNAME:
        if hash_password(password) == ADMIN_PASSWORD_HASH:
            return True, "Restaurant owner"
        return False, None

    accounts_df = load_customer_accounts()
    account_rows = accounts_df[accounts_df["username"].eq(normalized_username)]
    if account_rows.empty:
        return False, None

    account = account_rows.iloc[0]
    if account["password_hash"] == hash_password(password):
        return True, "Customer"

    return False, None


def create_account(username, password, confirm_password):
    normalized_username = username.strip().lower()
    if not normalized_username:
        return False, "Please enter a username."
    if normalized_username == ADMIN_USERNAME:
        return False, "The admin account is fixed and cannot be created."

    accounts_df = load_customer_accounts()
    if normalized_username in accounts_df["username"].values:
        return False, "This username already exists."
    if not password:
        return False, "Please enter a password."
    if password != confirm_password:
        return False, "Passwords do not match."

    save_customer_account(normalized_username, password)
    st.session_state.authenticated_user = normalized_username
    st.session_state.authenticated_role = "Customer"
    return True, "Customer account created successfully."


def render_authentication():
    with st.sidebar:
        st.title("Restaurant review")
        auth_mode = st.segmented_control(
            "Account",
            options=["Login", "Create account"],
            default="Login",
        )

    st.title("Restaurant sentiment app")
    st.write("Login or create an account to continue.")

    if auth_mode == "Login":
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Login",
                type="primary",
                icon=":material/login:",
            )

        if submitted:
            normalized_username = username.strip().lower()
            authenticated, role = authenticate_user(normalized_username, password)
            if authenticated:
                st.session_state.authenticated_user = normalized_username
                st.session_state.authenticated_role = role
                st.rerun()

            st.error("Invalid username or password.")

        st.caption("Customers can create their own account.")
    else:
        with st.form("create_account_form"):
            username = st.text_input("New username")
            password = st.text_input("New password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button(
                "Create customer account",
                type="primary",
                icon=":material/person_add:",
            )

        if submitted:
            created, message = create_account(
                username,
                password,
                confirm_password,
            )
            if created:
                st.success(message)
                st.rerun()

            st.error(message)


def render_sidebar_navigation():
    role = st.session_state.authenticated_role
    username = st.session_state.authenticated_user

    with st.sidebar:
        st.title("Restaurant review")
        st.caption(f"Signed in as `{username}`")
        st.caption(f"Role: `{role}`")

        if role == "Restaurant owner":
            page = st.radio(
                "Navigation",
                options=["Restaurant owner"],
                key="owner_navigation",
            )
        else:
            page = st.radio(
                "Navigation",
                options=["Customer review"],
                key="customer_navigation",
            )

        if st.button("Logout", icon=":material/logout:"):
            st.session_state.authenticated_user = None
            st.session_state.authenticated_role = None
            st.rerun()

    return page


def render_customer_review():
    st.title("Customer review")
    st.write("Enter your restaurant review below.")

    with st.form("sentiment_form"):
        review = st.text_area(
            "Restaurant review",
            placeholder="Example: The food was delicious and the staff were friendly.",
            height=160,
        )
        submitted = st.form_submit_button(
            "Submit review",
            type="primary",
            icon=":material/rate_review:",
        )

    if submitted:
        if not review.strip(): #using strip() to check
            st.error("Please enter a review before continuing.") #if the customer submit an empty review, display an error message
            st.stop()

        try:
            save_customer_review(st.session_state.authenticated_user, review)
        except Exception as error:
            st.error(f"Review submission failed: {error}")
        else:
            st.success("Thank you. Your review has been recorded.")


def get_submitted_review_predictions(
    reviews_df,
    selected_model,
    vectorizer,
    lemmatizer,
    stop_words,
):
    if reviews_df.empty:
        return reviews_df

    prediction_rows = [
        predict_sentiment(
            review,
            selected_model,
            vectorizer,
            lemmatizer,
            stop_words,
        )
        for review in reviews_df["review"]
    ]
    display_df = reviews_df.copy()
    display_df["selected_model_prediction"] = [
        prediction[0] for prediction in prediction_rows
    ]
    display_df["sentiment_score"] = [
        prediction[1] for prediction in prediction_rows
    ]
    display_df["cleaned_review"] = [
        prediction[2] for prediction in prediction_rows
    ]
    return display_df


def render_submitted_reviews_tab(
    trained_models,
    vectorizer,
    metrics_df,
    lemmatizer,
    stop_words,
):
    best_model_row = metrics_df.loc[metrics_df["Accuracy"].idxmax()]
    model_names = list(trained_models.keys())

    selected_model_name = st.selectbox(
        "Choose NLP model",
        options=model_names,
        index=model_names.index(best_model_row["Model"]),
        key="submitted_reviews_model",
    )
    selected_model = trained_models[selected_model_name]
    model_metric_row = metrics_df.loc[
        metrics_df["Model"].eq(selected_model_name)
    ].iloc[0]
    submitted_reviews_df = get_submitted_review_predictions(
        load_customer_reviews(),
        selected_model,
        vectorizer,
        lemmatizer,
        stop_words,
    )

    with st.container(horizontal=True):
        st.metric(
            "Submitted reviews",
            f"{len(submitted_reviews_df):,}",
            border=True,
        )
        st.metric(
            "Review model",
            selected_model_name,
            f"{model_metric_row['Accuracy']:.2%} accuracy",
            border=True,
        )

    if submitted_reviews_df.empty:
        st.info("No customer reviews have been submitted yet.")
        return

    submitted_prediction_counts = (
        submitted_reviews_df["selected_model_prediction"]
        .value_counts()
        .reindex(SENTIMENT_LABELS, fill_value=0)
        .rename_axis("Sentiment")
        .reset_index(name="Reviews")
    )

    with st.container(border=True):
        st.subheader(f"{selected_model_name} submitted review predictions")
        st.bar_chart(
            submitted_prediction_counts,
            x="Sentiment",
            y="Reviews",
            horizontal=True,
        )

    with st.container(border=True):
        st.subheader("All submitted customer reviews")
        reviews_display = submitted_reviews_df[
            [
                "submitted_at",
                "username",
                "review",
                "selected_model_prediction",
                "sentiment_score",
            ]
        ].rename(
            columns={
                "submitted_at": "Submitted at",
                "username": "Customer",
                "review": "Review",
                "selected_model_prediction": "Sentiment analysis result",
                "sentiment_score": "Sentiment score",
            }
        )
        st.dataframe(
            reviews_display,
            hide_index=True,
        )


def render_labelled_sentiment_tab(
    df,
    metrics_df,
):
    best_model_row = metrics_df.loc[metrics_df["Accuracy"].idxmax()]

    selected_sentiment = st.segmented_control(
        "Filter reviews by labelled sentiment",
        options=["All", *SENTIMENT_LABELS],
        default="All",
    )
    filtered_df = (
        df
        if selected_sentiment == "All"
        else df[df["Sentiment"] == selected_sentiment]
    ).copy()

    with st.container(horizontal=True):
        st.metric("Reviews", f"{len(filtered_df):,}", border=True)
        st.metric("NLP models", len(metrics_df), border=True)
        st.metric(
            "Best accuracy",
            best_model_row["Model"],
            f"{best_model_row['Accuracy']:.2%}",
            border=True,
        )

    sentiment_counts = (
        filtered_df["Sentiment"]
        .value_counts()
        .reindex(SENTIMENT_LABELS, fill_value=0)
        .rename_axis("Sentiment")
        .reset_index(name="Reviews")
    )

    with st.container(border=True):
        st.subheader("Labelled sentiment distribution")
        st.bar_chart(
            sentiment_counts,
            x="Sentiment",
            y="Reviews",
            horizontal=True,
        )

    source_counts = (
        filtered_df["DatasetSource"]
        .value_counts()
        .rename_axis("Dataset")
        .reset_index(name="Reviews")
    )

    with st.container(border=True):
        st.subheader("Dataset source distribution")
        st.bar_chart(source_counts, x="Dataset", y="Reviews", horizontal=True)

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
            "Compare SVM, Decision Tree, and Logistic Regression using weighted "
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
        preview_columns = [
            column
            for column in [
                "Review",
                "Sentiment",
                "SentimentScore",
                "DatasetSource",
                "AspectPolarities",
                "SourceFile",
                "Rating",
                "cleaned_review",
            ]
            if column in df.columns
            or column in filtered_df.columns
        ]
        st.dataframe(filtered_df[preview_columns], hide_index=True)


def render_owner_dashboard(
    df,
    trained_models,
    vectorizer,
    metrics_df,
    lemmatizer,
    stop_words,
):
    st.title("Restaurant owner dashboard")

    submitted_tab, labelled_tab = st.tabs(
        [
            "Submitted customer reviews",
            "Filter by labelled sentiment",
        ]
    )

    with submitted_tab:
        render_submitted_reviews_tab(
            trained_models,
            vectorizer,
            metrics_df,
            lemmatizer,
            stop_words,
        )

    with labelled_tab:
        render_labelled_sentiment_tab(
            df,
            metrics_df,
        )


initialize_session_state()
ensure_storage_files()

if st.session_state.authenticated_user is None:
    render_authentication()
    st.stop()

page = render_sidebar_navigation()

if page == "Customer review":
    render_customer_review()
    st.stop()

try:
    with st.spinner("Preparing app..."):
        df, lemmatizer, stop_words, trained_models, vectorizer, metrics_df = (
            load_models(
                POSITIVE_REVIEW_SAMPLE_SIZE,
                TRAINING_DATASET_CACHE_VERSION,
            )
        )
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

render_owner_dashboard(
    df,
    trained_models,
    vectorizer,
    metrics_df,
    lemmatizer,
    stop_words,
)
