import hashlib
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_assignment.core.constants import (
    APP_TITLE,
    DEVELOPER_USERNAME,
    RESTAURANT_DESCRIPTION,
    RESTAURANT_IMAGE_PATH,
    RESTAURANT_NAME,
    RESTAURANT_OWNER_DISPLAY_USERNAME,
    RESTAURANT_OWNER_USERNAME,
    ROLE_CUSTOMER,
    ROLE_DEVELOPER,
    ROLE_RESTAURANT_OWNER,
)
from ai_assignment.core.sentiment_analyzer import (
    POSITIVE_REVIEW_SAMPLE_SIZE,
    SENTIMENT_LABELS,
    ensure_nltk_data,
    load_and_clean_dataset,
    predict_sentiment,
    train_sentiment_models,
)


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
)


APP_DATA_DIR = Path(__file__).resolve().parent / "data"
CUSTOMER_ACCOUNTS_PATH = APP_DATA_DIR / "customer_accounts.csv"
CUSTOMER_REVIEWS_PATH = APP_DATA_DIR / "customer_reviews.csv"
DEVELOPER_PASSWORD_HASH = hashlib.sha256("admin123".encode("utf-8")).hexdigest()
RESTAURANT_OWNER_PASSWORD_HASH = hashlib.sha256(
    "restaurantOwner123".encode("utf-8")
).hexdigest()
TRAINING_DATASET_CACHE_VERSION = (
    f"positive-cap-{POSITIVE_REVIEW_SAMPLE_SIZE}-macro-metrics-confusion-v1"
)
SENTIMENT_FILTER_OPTIONS = ["All", *SENTIMENT_LABELS]


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
    trained_models, vectorizer, metrics_df, confusion_matrices = (
        train_sentiment_models(df)
    )
    return (
        df,
        lemmatizer,
        stop_words,
        trained_models,
        vectorizer,
        metrics_df,
        confusion_matrices,
    )


def render_restaurant_review_styles():
    st.html(
        """
        <style>
            [data-testid="stAppViewContainer"] {
                background: #f7f3ef;
            }

            .block-container {
                max-width: 1120px;
                padding-top: 1.4rem;
            }

            .rr-page-header {
                align-items: center;
                display: flex;
                justify-content: space-between;
                margin-bottom: 18px;
                margin-top: 46px;
            }

            .rr-eyebrow {
                color: #b65f18;
                font-size: 13px;
                font-weight: 800;
                letter-spacing: 0;
                margin-bottom: 4px;
            }

            .rr-page-header h2 {
                color: #171717;
                font-size: 24px;
                line-height: 1.15;
                margin: 0;
            }

            .rr-role-pill {
                background: white;
                border: 1px solid #eaded2;
                border-radius: 999px;
                color: #5b4634;
                font-size: 13px;
                font-weight: 750;
                padding: 8px 14px;
            }

            .st-key-restaurant_cover_image {
                background: white;
                border: 1px solid #eaded2;
                border-radius: 8px;
                margin: 0 auto 20px;
                max-width: 720px;
                overflow: hidden;
                padding: 8px;
            }

            .st-key-restaurant_cover_image img {
                aspect-ratio: 5 / 3;
                border-radius: 6px;
                object-fit: cover;
                width: 100%;
            }

            .rr-summary-card,
            .rr-info,
            .rr-review-card {
                background: white;
                border: 1px solid #eaded2;
                border-radius: 8px;
                box-sizing: border-box;
                box-shadow: 0 12px 30px rgba(70, 45, 20, 0.07);
                color: #191919;
            }

            .rr-summary-card {
                min-height: 100%;
                padding: 20px;
            }

            .rr-title {
                font-size: 34px;
                font-weight: 850;
                line-height: 1.1;
                margin: 0 0 12px;
            }

            .rr-status-row {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 18px;
            }

            .rr-status-pill {
                border-radius: 999px;
                font-size: 12px;
                font-weight: 800;
                padding: 5px 10px;
            }

            .rr-open {
                background: #e8f7e8;
                color: #228a34;
            }

            .rr-tag {
                background: #fff1e2;
                color: #a85615;
            }

            .rr-description p {
                color: #49413b;
                font-size: 14px;
                line-height: 1.55;
                margin: 0 0 12px;
            }

            .rr-main-grid {
                margin-top: 18px;
            }

            .rr-rule {
                background: #eaded2;
                height: 2px;
                margin: 4px 0 16px;
                width: 100%;
            }

            .rr-section-heading {
                align-items: center;
                border-bottom: 2px solid #eaded2;
                display: flex;
                justify-content: space-between;
                margin-bottom: 16px;
                padding: 0 0 10px;
            }

            .rr-section-heading h3,
            .rr-info h3 {
                color: #171717;
                font-size: 21px;
                line-height: 1.2;
                margin: 0;
            }

            .rr-section-action {
                color: #b65f18;
                font-size: 13px;
                font-weight: 800;
            }

            .rr-owner-heading {
                border-bottom: 2px solid #eaded2;
                margin: 0 0 16px;
                padding-bottom: 8px;
            }

            .rr-info {
                color: #242424;
                padding: 20px;
            }

            .rr-info-line {
                background: #eaded2;
                height: 2px;
                margin: 10px 0 16px;
                width: 100%;
            }

            .rr-info-item {
                margin-bottom: 18px;
            }

            .rr-info-label {
                font-size: 13px;
                font-weight: 800;
                margin-bottom: 7px;
            }

            .rr-info-value {
                color: #6f655e;
                font-size: 13px;
                line-height: 1.55;
            }

            .rr-review-card {
                margin-bottom: 12px;
                padding: 14px 16px;
            }

            .rr-review-meta {
                color: #6f655e;
                font-size: 12px;
                margin-bottom: 8px;
            }

            .rr-review-author {
                color: #171717;
                font-size: 14px;
                font-weight: 850;
            }

            .rr-review-text {
                color: #39332e;
                font-size: 13px;
                line-height: 1.5;
                margin: 0;
            }

            .stFormSubmitButton button[kind="primary"],
            .stButton button[kind="primary"] {
                background: #b65f18;
                border-color: #b65f18;
            }
        </style>
        """,
    )


def render_page_header(title, subtitle, icon_name="restaurant"):
    st.markdown(f":material/{icon_name}: **{APP_TITLE}**")
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def render_restaurant_review_header(role_label):
    st.html(
        f"""
        <div class="rr-page-header">
            <div>
                <h2>{escape(RESTAURANT_NAME)}</h2>
            </div>
            <div class="rr-role-pill">{escape(role_label)}</div>
        </div>
        """,
    )


def render_restaurant_overview():
    description_html = "".join(
        f"<p>{escape(paragraph)}</p>"
        for paragraph in RESTAURANT_DESCRIPTION.split("\n\n")
    )

    with st.container(key="restaurant_cover_image"):
        st.image(str(RESTAURANT_IMAGE_PATH), width="stretch")

    content_col, info_col = st.columns([1.35, 0.85], gap="large")

    with content_col:
        st.html(
            f"""
            <section class="rr-summary-card">
                <h1 class="rr-title">{escape(RESTAURANT_NAME)}</h1>
                <div class="rr-status-row">
                    <span class="rr-status-pill rr-open">Open</span>
                    <span class="rr-status-pill rr-tag">Fresh ingredients</span>
                    <span class="rr-status-pill rr-tag">Elegant dining</span>
                </div>
                <div class="rr-description">{description_html}</div>
            </section>
            """,
        )

    st.html('<div class="rr-main-grid"></div>')

    with info_col:
        st.html(
            """
            <aside class="rr-info">
                <h3>General info</h3>
                <div class="rr-info-line"></div>
                <div class="rr-info-item">
                    <div class="rr-info-label">Dining atmosphere</div>
                    <div class="rr-info-value">Elegant, warm, comfortable, and softly lit.</div>
                </div>
                <div class="rr-info-item">
                    <div class="rr-info-label">Suitable for</div>
                    <div class="rr-info-value">Family gatherings, casual dining, celebrations, and special occasions.</div>
                </div>
                <div class="rr-info-item">
                    <div class="rr-info-label">Review areas</div>
                    <div class="rr-info-value">Food quality, service, atmosphere, and overall experience.</div>
                </div>
            </aside>
            """,
        )

    return content_col


def render_review_section_heading(title, action_label=None):
    action_html = (
        f'<span class="rr-section-action">{escape(action_label)}</span>'
        if action_label
        else ""
    )
    st.html(
        f"""
        <div class="rr-section-heading">
            <h3>{escape(title)}</h3>
            {action_html}
        </div>
        """,
    )


def render_recent_review_cards(limit=3):
    reviews_df = load_customer_reviews()
    if reviews_df.empty:
        st.caption("No customer reviews yet.")
        return

    recent_reviews = reviews_df.tail(limit).iloc[::-1]
    for _, review_row in recent_reviews.iterrows():
        submitted_at = review_row["submitted_at"] or "Recently"
        username = review_row["username"] or "Customer"
        review = review_row["review"]
        st.html(
            f"""
            <article class="rr-review-card">
                <div class="rr-review-author">{escape(username)}</div>
                <div class="rr-review-meta">Reviewed date - {escape(submitted_at)}</div>
                <p class="rr-review-text">{escape(review)}</p>
            </article>
            """,
        )


def render_table(
    display_df,
    show_sentiment_score=False,
):
    column_config = {
        "Submitted at": st.column_config.TextColumn(
            "Submitted at",
            width="medium",
        ),
        "Customer": st.column_config.TextColumn("Customer", width="small"),
        "Review": st.column_config.TextColumn("Review", width="large"),
        "Sentiment analysis result": st.column_config.TextColumn(
            "Sentiment analysis result",
            width="medium",
        ),
    }
    if show_sentiment_score:
        column_config["Sentiment score"] = st.column_config.NumberColumn(
            "Sentiment score",
            format="%.1f",
            width="small",
        )

    st.dataframe(
        display_df,
        hide_index=True,
        column_config=column_config,
        height=420,
    )


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
    if normalized_username == RESTAURANT_OWNER_USERNAME:
        if hash_password(password) == RESTAURANT_OWNER_PASSWORD_HASH:
            return True, ROLE_RESTAURANT_OWNER, RESTAURANT_OWNER_DISPLAY_USERNAME
        return False, None, None

    if normalized_username == DEVELOPER_USERNAME:
        if hash_password(password) == DEVELOPER_PASSWORD_HASH:
            return True, ROLE_DEVELOPER, DEVELOPER_USERNAME
        return False, None, None

    accounts_df = load_customer_accounts()
    account_rows = accounts_df[accounts_df["username"].eq(normalized_username)]
    if account_rows.empty:
        return False, None, None

    account = account_rows.iloc[0]
    if account["password_hash"] == hash_password(password):
        return True, ROLE_CUSTOMER, normalized_username

    return False, None, None


def create_account(username, password, confirm_password):
    normalized_username = username.strip().lower()
    if not normalized_username:
        return False, "Please enter a username."
    reserved_usernames = {
        DEVELOPER_USERNAME,
        RESTAURANT_OWNER_USERNAME,
    }
    if normalized_username in reserved_usernames:
        return False, "This username is reserved for a fixed role account."

    accounts_df = load_customer_accounts()
    if normalized_username in accounts_df["username"].values:
        return False, "This username already exists."
    if not password:
        return False, "Please enter a password."
    if password != confirm_password:
        return False, "Passwords do not match."

    save_customer_account(normalized_username, password)
    st.session_state.authenticated_user = normalized_username
    st.session_state.authenticated_role = ROLE_CUSTOMER
    return True, "Customer account created successfully."


def render_authentication():
    with st.sidebar:
        st.markdown(f":material/restaurant: **{APP_TITLE}**")
        st.caption("Restaurant review workspace")
        auth_mode = st.segmented_control(
            "Account",
            options=["Login", "Create account"],
            default="Login",
        )

    render_page_header(
        APP_TITLE,
        "",
        "reviews",
    )

    auth_col, note_col = st.columns([1.05, 0.95], gap="large")

    if auth_mode == "Login":
        with auth_col:
            with st.form("login_form"):
                st.subheader("Sign in")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button(
                    "Login",
                    type="primary",
                    icon=":material/login:",
                )

        if submitted:
            normalized_username = username.strip().lower()
            authenticated, role, display_username = authenticate_user(
                normalized_username,
                password,
            )
            if authenticated:
                st.session_state.authenticated_user = display_username
                st.session_state.authenticated_role = role
                st.rerun()

            st.error("Invalid username or password.")

    else:
        with auth_col:
            with st.form("create_account_form"):
                st.subheader("Create customer account")
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
        st.markdown(f":material/restaurant: **{APP_TITLE}**")
        with st.container(border=True):
            st.markdown(f":material/account_circle: `{username}`")
            st.caption(f"Role: {role}")

        if role == ROLE_RESTAURANT_OWNER:
            page = st.radio(
                "Navigation",
                options=["Submitted reviews"],
                key="owner_navigation",
            )
        elif role == ROLE_DEVELOPER:
            page = st.radio(
                "Navigation",
                options=["Developer tools"],
                key="developer_navigation",
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


def get_best_model_details(trained_models, metrics_df):
    best_model_row = metrics_df.loc[metrics_df["Accuracy"].idxmax()]
    best_model_name = best_model_row["Model"]
    return best_model_name, trained_models[best_model_name], best_model_row


def render_customer_review(
    trained_models,
    vectorizer,
    metrics_df,
    lemmatizer,
    stop_words,
):
    render_restaurant_review_styles()
    render_restaurant_review_header("Customer")
    review_col = render_restaurant_overview()

    _, best_model, _ = get_best_model_details(
        trained_models,
        metrics_df,
    )

    with review_col:
        render_review_section_heading("Reviews", "Write a review")
        with st.form("sentiment_form", border=False):
            review = st.text_area(
                "Restaurant review",
                placeholder="Example: The food was delicious and the staff were friendly.",
                height=180,
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
                sentiment, score, _ = predict_sentiment(
                    review,
                    best_model,
                    vectorizer,
                    lemmatizer,
                    stop_words,
                )
                save_customer_review(st.session_state.authenticated_user, review)
            except Exception as error:
                st.error(f"Review submission failed: {error}")
            else:
                st.success(
                    "Thank you for your review to The Grand Restaurant!",
                    icon=":material/check_circle:",
                )

        render_recent_review_cards()


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


def render_submitted_reviews(
    trained_models,
    vectorizer,
    metrics_df,
    lemmatizer,
    stop_words,
    selected_model_name=None,
    sentiment_filter="All",
    show_review_model=True,
    show_prediction_chart=True,
    show_sentiment_score=True,
):
    best_model_name, _, _ = get_best_model_details(trained_models, metrics_df)
    selected_model_name = selected_model_name or best_model_name
    selected_model = trained_models[selected_model_name]
    submitted_reviews_df = get_submitted_review_predictions(
        load_customer_reviews(),
        selected_model,
        vectorizer,
        lemmatizer,
        stop_words,
    )
    total_submitted_reviews = len(submitted_reviews_df)
    if sentiment_filter != "All" and not submitted_reviews_df.empty:
        submitted_reviews_df = submitted_reviews_df[
            submitted_reviews_df["selected_model_prediction"].eq(sentiment_filter)
        ]

    if show_review_model:
        model_metric_row = metrics_df.loc[
            metrics_df["Model"].eq(selected_model_name)
        ].iloc[0]
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
    else:
        st.metric(
            "Submitted reviews",
            f"{len(submitted_reviews_df):,}",
            border=True,
        )

    if submitted_reviews_df.empty:
        if total_submitted_reviews == 0:
            st.info("No customer reviews have been submitted yet.")
        else:
            st.info("No customer reviews match this sentiment filter.")
        return

    if show_prediction_chart:
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
        st.subheader(":material/table_chart: All submitted customer reviews")
        review_columns = [
            "submitted_at",
            "username",
            "review",
            "selected_model_prediction",
        ]
        if show_sentiment_score:
            review_columns.append("sentiment_score")

        reviews_display = submitted_reviews_df[review_columns].rename(
            columns={
                "submitted_at": "Submitted at",
                "username": "Customer",
                "review": "Review",
                "selected_model_prediction": "Sentiment analysis result",
                "sentiment_score": "Sentiment score",
            }
        )
        render_table(
            reviews_display,
            show_sentiment_score=show_sentiment_score,
        )


def render_owner_dashboard(
    trained_models,
    vectorizer,
    metrics_df,
    lemmatizer,
    stop_words,
):
    render_restaurant_review_styles()
    render_restaurant_review_header("Restaurant owner")
    render_restaurant_overview()

    st.html(
        """
        <div class="rr-owner-heading">
            <h3>Reviews</h3>
        </div>
        """,
    )

    with st.container(border=True):
        sentiment_filter = st.segmented_control(
            "Filter customer reviews by sentiment",
            options=SENTIMENT_FILTER_OPTIONS,
            default="All",
            key="owner_sentiment_filter",
        )
    render_submitted_reviews(
        trained_models,
        vectorizer,
        metrics_df,
        lemmatizer,
        stop_words,
        sentiment_filter=sentiment_filter,
        show_review_model=False,
        show_prediction_chart=False,
        show_sentiment_score=False,
    )


def render_developer_customer_reviews_tab(
    trained_models,
    vectorizer,
    metrics_df,
    lemmatizer,
    stop_words,
):
    best_model_name, _, _ = get_best_model_details(trained_models, metrics_df)
    model_names = list(trained_models.keys())
    selected_model_name = st.selectbox(
        "Choose NLP model",
        options=model_names,
        index=model_names.index(best_model_name),
        key="developer_submitted_reviews_model",
    )
    sentiment_filter = st.segmented_control(
        "Filter customer reviews by predicted sentiment",
        options=SENTIMENT_FILTER_OPTIONS,
        default="All",
    )
    render_submitted_reviews(
        trained_models,
        vectorizer,
        metrics_df,
        lemmatizer,
        stop_words,
        selected_model_name=selected_model_name,
        sentiment_filter=sentiment_filter,
    )


def render_labelled_sentiment_tab(
    df,
    metrics_df,
):
    best_model_row = metrics_df.loc[metrics_df["Accuracy"].idxmax()]

    selected_sentiment = st.segmented_control(
        "Filter reviews by labelled sentiment",
        options=SENTIMENT_FILTER_OPTIONS,
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
            "Compare SVM, Decision Tree, and Logistic Regression using overall "
            "Accuracy plus macro-averaged Precision, Recall, and F1 Score."
        )
        st.bar_chart(
            metrics_display,
            x="Model",
            y=metric_columns,
            y_label="Score (%)",
        )
        st.dataframe(
            metrics_display,
            hide_index=True,
            column_config={
                "Accuracy": st.column_config.NumberColumn(
                    "Accuracy",
                    format="%.2f%%",
                ),
                "Precision": st.column_config.NumberColumn(
                    "Precision",
                    format="%.2f%%",
                ),
                "Recall": st.column_config.NumberColumn(
                    "Recall",
                    format="%.2f%%",
                ),
                "F1 Score": st.column_config.NumberColumn(
                    "F1 Score",
                    format="%.2f%%",
                ),
            },
        )

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


def render_confusion_matrix_tab(confusion_matrices, metrics_df):
    best_model_name = metrics_df.loc[metrics_df["Accuracy"].idxmax(), "Model"]
    model_names = list(confusion_matrices.keys())
    selected_model_name = st.selectbox(
        "Choose NLP model",
        options=model_names,
        index=model_names.index(best_model_name),
        key="developer_confusion_matrix_model",
    )
    matrix_df = confusion_matrices[selected_model_name]

    with st.container(border=True):
        st.subheader(f"{selected_model_name} confusion matrix")
        st.caption(
            "Rows are actual sentiment labels from the test set. Columns are "
            "the sentiment labels predicted by the selected model."
        )
        st.dataframe(
            matrix_df,
            column_config={
                "Predicted Positive": st.column_config.NumberColumn(
                    "Predicted Positive",
                    width="medium",
                ),
                "Predicted Neutral": st.column_config.NumberColumn(
                    "Predicted Neutral",
                    width="medium",
                ),
                "Predicted Negative": st.column_config.NumberColumn(
                    "Predicted Negative",
                    width="medium",
                ),
            },
        )


def render_developer_dashboard(
    df,
    trained_models,
    vectorizer,
    metrics_df,
    lemmatizer,
    stop_words,
    confusion_matrices,
):
    render_page_header(
        "Developer tools",
        "Model comparison and labelled sentiment inspection.",
        "analytics",
    )

    submitted_tab, labelled_tab, confusion_matrix_tab = st.tabs(
        [
            "Customer review predictions",
            "Sentiment Analysis Metrics",
            "Confusion matrix",
        ]
    )

    with submitted_tab:
        render_developer_customer_reviews_tab(
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

    with confusion_matrix_tab:
        render_confusion_matrix_tab(
            confusion_matrices,
            metrics_df,
        )


initialize_session_state()
ensure_storage_files()

if st.session_state.authenticated_user is None:
    render_authentication()
    st.stop()

page = render_sidebar_navigation()

try:
    with st.spinner("Preparing app..."):
        (
            df,
            lemmatizer,
            stop_words,
            trained_models,
            vectorizer,
            metrics_df,
            confusion_matrices,
        ) = load_models(
            POSITIVE_REVIEW_SAMPLE_SIZE,
            TRAINING_DATASET_CACHE_VERSION,
        )
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

if page == "Customer review":
    render_customer_review(
        trained_models,
        vectorizer,
        metrics_df,
        lemmatizer,
        stop_words,
    )
elif page == "Submitted reviews":
    render_owner_dashboard(
        trained_models,
        vectorizer,
        metrics_df,
        lemmatizer,
        stop_words,
    )
else:
    render_developer_dashboard(
        df,
        trained_models,
        vectorizer,
        metrics_df,
        lemmatizer,
        stop_words,
        confusion_matrices,
    )
