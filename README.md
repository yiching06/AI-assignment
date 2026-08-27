# Restaurant Review Sentiment Analysis

This project is a Streamlit app that uses one restaurant review dataset, trains multiple TF-IDF based NLP models, and lets you test custom restaurant reviews as Negative, Neutral, or Positive.

The app has a login screen and role-based navigation in the left sidebar:

- Customer review: customers enter a restaurant review and see the sentiment result from the best-accuracy model.
- Restaurant owner: the owner can only view customer-submitted reviews with sentiment results from the best-accuracy model, without sentiment scores.
- Developer tools: developers choose an NLP model, compare model accuracy, view sentiment scores for submitted customer reviews, view labelled training sentiment, and filter submitted customer reviews by Positive, Negative, or Neutral predictions.

The fixed role accounts are:

- Restaurant owner: `restaurantOwner` / `restaurantOwner123`
- Developer: `admin` / `admin123`

Only customer accounts can be created from the app. Customer accounts and submitted reviews are saved locally. Customer reviews are saved as raw submissions first; sentiment analysis is calculated when a role page displays or submits a review:

- Customer accounts: `data/customer_accounts.csv`
- Customer reviews: `data/customer_reviews.csv`

## Project Structure

```text
AI-assignment/
+-- ai_assignment/
|   +-- core/
|       +-- constants.py
|       +-- datasets.py
|       +-- lexicons.py
|       +-- models.py
|       +-- prediction.py
|       +-- preprocessing.py
|       +-- rules.py
|       +-- sentiment_analyzer.py
|   +-- resources/
|       +-- sentiment_phrases.json
+-- streamlit_app.py
+-- requirements.txt
+-- README.md
+-- data/
    +-- customer_accounts.csv
    +-- customer_reviews.csv
```

## Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run The Streamlit App

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

You can also run it with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## KaggleHub Dataset

The app downloads this dataset with KaggleHub:

```python
import kagglehub

reviews_path = kagglehub.dataset_download("joebeachcapital/restaurant-reviews")

print("Path to restaurant reviews dataset files:", reviews_path)
```

The KaggleHub dataset is stored in KaggleHub's cache, not in this project folder. The app loads the selected dataset into one training dataframe and tracks each row with `DatasetSource`.

The app uses this dataset:

- `joebeachcapital/restaurant-reviews`

The `joebeachcapital/restaurant-reviews` dataset stores full restaurant reviews with ratings. The app converts ratings into sentiment classes:

- Negative: rating below `2.5`
- Neutral: rating from `2.5` to below `4`
- Positive: rating `4` and above

To keep the training dataset more balanced, the app deterministically samples at most 4,500 Positive reviews and keeps all available Neutral and Negative reviews. The developer dashboard displays counts and preview rows from this sampled training dataset.

Prediction labels are also shown as numeric sentiment scores:

- Negative: `0`
- Neutral: `0.5`
- Positive: `1`

The training step also adds custom neutral examples such as `ok`, `okok`, and `okay` so short neutral replies are recognized more reliably.
The prediction step uses NLTK's `opinion_lexicon` and `vader_lexicon` corpora for positive and negative cue words instead of storing those word lists in code. Project-specific phrase rules and small restaurant cue overrides, such as `not too crowded`, `tasty`, `not very memorable`, `nothing remarkable`, and `met my expectations`, are stored in `ai_assignment/resources/sentiment_phrases.json`.
Sentences with contrast words such as `but` that contain both positive and negative cues, such as `polite, but there wasn't much interaction`, are treated as Neutral.

Reviews are cleaned by lowercasing text, removing punctuation, removing stop words, and lemmatizing each word to its base form. Important context words such as `not`, `no`, `too`, and `but` are kept because phrases like `not too crowded` or mixed expressions with contrast words change the sentiment.

## NLP Models

The app trains and compares these classification models with the same TF-IDF feature set:

- Support Vector Machine (SVM)
- Decision Tree
- Logistic Regression

Each model is evaluated with:

- Accuracy
- Macro-averaged Precision
- Macro-averaged Recall
- Macro-averaged F1 Score

## File Roles

- `streamlit_app.py` is the main Streamlit entrypoint.
- `ai_assignment/core/constants.py` stores dataset IDs, labels, scores, and shared paths.
- `ai_assignment/core/datasets.py` downloads, parses, and cleans the selected dataset.
- `ai_assignment/core/lexicons.py` loads NLTK sentiment cue words and project phrase rules.
- `ai_assignment/core/models.py` trains SVM, Decision Tree, and Logistic Regression and calculates metrics.
- `ai_assignment/core/prediction.py` predicts sentiment for a custom review.
- `ai_assignment/core/preprocessing.py` handles NLTK setup, stop words, and text cleaning.
- `ai_assignment/core/rules.py` handles rule-based Positive and Neutral overrides.
- `ai_assignment/core/sentiment_analyzer.py` keeps the public imports and console test entrypoint.
- `ai_assignment/resources/sentiment_phrases.json` stores custom phrase rules outside Python code.
