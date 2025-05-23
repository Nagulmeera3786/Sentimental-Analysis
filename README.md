
# Sentiment Analysis App

This project is a simple sentiment analysis web application that classifies user input as **Positive 😃**, **Neutral 😐**, or **Negative 😞**.

## Features

- Clean and preprocess input text.
- Logistic Regression model using scikit-learn.
- Simple and interactive Gradio web interface.

## How to Run

1. Make sure you have `Tweets.csv` in the same directory.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

## Dataset

Use the [Twitter US Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) from Kaggle.
Save the file as `Tweets.csv` in the same folder.

## Output

Predicts one of the following sentiments based on the input text:

- Positive 😃
- Neutral 😐
- Negative 😞
