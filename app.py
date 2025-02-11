from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import mysql.connector
import joblib
import os

app = Flask(__name__)

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="socialmetrics"
)
cursor = db.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS tweets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT NOT NULL,
    positive INT DEFAULT 0,
    negative INT DEFAULT 0
)
""")
db.commit()

def load_annotated_data():
    cursor.execute("SELECT text, positive, negative FROM tweets")
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['text', 'positive', 'negative'])
    return df

def load_or_train_model():
    if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
    else:
        df = load_annotated_data()
        if df.empty:
            return None, None

        X = df['text']
        y = df['positive']

        # Text vectorisation
        vectorizer = TfidfVectorizer()
        X_vec = vectorizer.fit_transform(X)

        # Model training
        model = LogisticRegression()
        model.fit(X_vec, y)

        # Save the model and vectorise it
        joblib.dump(model, "model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")

    return model, vectorizer

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    tweets = data.get('tweets', [])

    model, vectorizer = load_or_train_model()
    if model is None or vectorizer is None:
        return jsonify({"error": "No annotated data available"}), 400

    X_vec = vectorizer.transform(tweets)
    predictions = model.predict_proba(X_vec)[:, 1]
    sentiment_scores = 2 * predictions - 1

    results = {f"tweet{i+1}": float(score) for i, score in enumerate(sentiment_scores)}
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)