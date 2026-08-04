# Restaurant Review Sentiment Analysis

This project is a Streamlit app that uses KaggleHub to download the restaurant reviews dataset, trains multiple TF-IDF based NLP models, and lets you test custom restaurant reviews as Negative, Neutral, or Positive.

The app has two tabs:

- Customer review: customers enter a review, choose a model, and get a sentiment prediction.
- Restaurant owner: owners view sentiment distribution and compare model evaluation metrics.

## Project Structure

```text
AI-assignment/
+-- ai_assignment/
|   +-- core/
|       +-- sentiment_analyzer.py
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

path = kagglehub.dataset_download("joebeachcapital/restaurant-reviews")
print("Path to dataset files:", path)
```

The downloaded dataset is stored in KaggleHub's cache, not in this project folder.

Ratings are converted into three sentiment classes:

- Negative: rating below `2.5`
- Neutral: rating from `2.5` to below `4`
- Positive: rating `4` and above

## NLP Models

The app trains and compares these classification models with the same TF-IDF feature set:

- Support Vector Machine (SVM)
- Naive Bayes
- Logistic Regression

Each model is evaluated with:

- Accuracy
- Precision
- Recall
- F1 Score

## File Roles

- `streamlit_app.py` is the main Streamlit entrypoint.
- `ai_assignment/core/sentiment_analyzer.py` handles KaggleHub loading, text cleaning, model training, model comparison, and prediction.
