# Restaurant Review Sentiment Analysis

This project trains a machine learning model to classify restaurant reviews as positive or negative. It includes a console runner, a Tkinter desktop app using MVC, and a Streamlit web app with optional KaggleHub dataset loading.

## Project Structure

```text
AI-assignment/
+-- ai_assignment/
|   +-- core/
|   |   +-- sentiment_analyzer.py
|   +-- interfaces/
|   |   +-- desktop_app.py
|   |   +-- streamlit_app.py
|   +-- mvc/
|       +-- controller.py
|       +-- model.py
|       +-- view.py
+-- data/
|   +-- restaurant_reviews.tsv
+-- logs/
+-- main.py
+-- requirements.txt
+-- README.md
```

## Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run The Desktop App

```powershell
.\.venv\Scripts\python.exe main.py
```

## Run The Streamlit App

```powershell
.\.venv\Scripts\streamlit.exe run ai_assignment\interfaces\streamlit_app.py
```

The Streamlit app can use the local dataset or download the latest KaggleHub dataset:

```python
import kagglehub

path = kagglehub.dataset_download("joebeachcapital/restaurant-reviews")
print("Path to dataset files:", path)
```

## File Roles

- `ai_assignment/core/sentiment_analyzer.py` handles dataset loading, text cleaning, model training, and prediction.
- `ai_assignment/mvc/model.py` wraps the machine learning logic for the desktop app.
- `ai_assignment/mvc/view.py` contains the Tkinter layout.
- `ai_assignment/mvc/controller.py` connects UI actions to model predictions.
- `ai_assignment/interfaces/desktop_app.py` starts the Tkinter app.
- `ai_assignment/interfaces/streamlit_app.py` starts the Streamlit web app.
- `main.py` is a short launcher for the desktop app.
