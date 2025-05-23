
import pandas as pd
import re
import nltk
import gradio as gr
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_data():
    df = pd.read_csv("Tweets.csv")
    df = df[['text', 'airline_sentiment']]
    df = df[df['airline_sentiment'].isin(['positive', 'neutral', 'negative'])]
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+|[^a-zA-Z\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df = load_data()
df['clean_text'] = df['text'].apply(clean_text)

label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
inverse_label_map = {0: 'Negative 😞', 1: 'Neutral 😐', 2: 'Positive 😃'}
df['label'] = df['airline_sentiment'].map(label_map)

X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)
vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

def predict_sentiment(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    return inverse_label_map[pred]

ui = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter your sentence here..."),
    outputs=gr.Label(num_top_classes=1),
    title="Sentiment Analysis",
    description="Predicts whether the input text is Positive 😃, Neutral 😐, or Negative 😞."
)

if __name__ == "__main__":
    ui.launch()
