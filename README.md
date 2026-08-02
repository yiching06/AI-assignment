# Restaurant Review Sentiment Analysis

This project is now a Streamlit app that uses KaggleHub to download the restaurant reviews dataset, trains an SVM sentiment model, and lets you test custom restaurant reviews.

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

The app uses this dataset:

```python
import kagglehub

path = kagglehub.dataset_download("joebeachcapital/restaurant-reviews")
print("Path to dataset files:", path)
```

The downloaded dataset is stored in KaggleHub's cache, not in this project folder.

## File Roles

- `streamlit_app.py` is the main Streamlit entrypoint.
- `ai_assignment/core/sentiment_analyzer.py` handles KaggleHub loading, text cleaning, model training, and prediction.
