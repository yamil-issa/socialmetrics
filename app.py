from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import mysql.connector
import joblib
import os

app = Flask(__name__)

# MysQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="socialmetrics"
)
cursor = db.cursor()

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

        vectorizer = TfidfVectorizer()
        X_vec = vectorizer.fit_transform(X)

        model = LogisticRegression()
        model.fit(X_vec, y)

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

@app.route('/annotate', methods=['POST'])
def annotate_tweet():
    data = request.json

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON object (single tweet) or a list of tweets"}), 400

    for tweet in data:
        text = tweet.get('text')
        positive = tweet.get('positive', 0)
        negative = tweet.get('negative', 0)

        if not text:
            return jsonify({"error": "Each tweet must have a 'text' field"}), 400

        cursor.execute("INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)", (text, positive, negative))

    db.commit()
    return jsonify({"message": f"{len(data)} tweet(s) annotated successfully"}), 201

@app.route('/evaluate', methods=['GET'])
def evaluate_model():
    df = load_annotated_data()
    if df.empty:
        return jsonify({"error": "No annotated data available"}), 400

    X = df['text']
    y = df['positive']

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return jsonify({
        "confusion_matrix": cm.tolist(),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    })

if __name__ == '__main__':
    app.run(debug=True)