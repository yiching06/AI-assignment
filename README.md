# Restaurant Review Sentiment Analysis

This project is a Streamlit app that uses multiple restaurant review datasets, trains multiple TF-IDF based NLP models, and lets you test custom restaurant reviews as Negative, Neutral, or Positive.

The app has two tabs:

- Customer review: customers enter a review, choose a model, and get a sentiment prediction.
- Restaurant owner: owners view sentiment distribution and compare model evaluation metrics.

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

semeval_path = kagglehub.dataset_download("charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis")
reviews_path = kagglehub.dataset_download("joebeachcapital/restaurant-reviews")

print("Path to SemEval dataset files:", semeval_path)
print("Path to restaurant reviews dataset files:", reviews_path)
```

The KaggleHub datasets are stored in KaggleHub's cache, not in this project folder. The app also downloads a small Yelp Restaurant Reviews Sentiment ZIP from Zenodo into `data/external/`, combines all datasets into one training dataframe, and tracks each row with `DatasetSource`.

The app uses these datasets:

- `charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis`
- `joebeachcapital/restaurant-reviews`
- Yelp Restaurant Reviews Sentiment Dataset from Zenodo: `https://zenodo.org/records/18723813`

The SemEval dataset stores restaurant review sentences in XML with aspect-level polarity annotations. The app converts those annotations into one sentence-level sentiment class:

- Negative: only negative aspect polarity
- Neutral: neutral, conflict, or mixed positive and negative polarity
- Positive: only positive aspect polarity

The `joebeachcapital/restaurant-reviews` dataset stores full restaurant reviews with ratings. The app converts ratings into sentiment classes:

- Negative: rating below `2.5`
- Neutral: rating from `2.5` to below `4`
- Positive: rating `4` and above

The Yelp dataset contains restaurant review text with labelled sentiment and rating metadata. The app standardizes its labels into the same Negative, Neutral, and Positive classes.

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
- Precision
- Recall
- F1 Score

## File Roles

- `streamlit_app.py` is the main Streamlit entrypoint.
- `ai_assignment/core/constants.py` stores dataset IDs, labels, scores, and shared paths.
- `ai_assignment/core/datasets.py` downloads, parses, combines, and cleans datasets.
- `ai_assignment/core/lexicons.py` loads NLTK sentiment cue words and project phrase rules.
- `ai_assignment/core/models.py` trains SVM, Decision Tree, and Logistic Regression and calculates metrics.
- `ai_assignment/core/prediction.py` predicts sentiment for a custom review.
- `ai_assignment/core/preprocessing.py` handles NLTK setup, stop words, and text cleaning.
- `ai_assignment/core/rules.py` handles rule-based Positive and Neutral overrides.
- `ai_assignment/core/sentiment_analyzer.py` keeps the public imports and console test entrypoint.
- `ai_assignment/resources/sentiment_phrases.json` stores custom phrase rules outside Python code.
