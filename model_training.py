from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from app import load_annotated_data
import joblib

def retrain_model():
    df = load_annotated_data()
    if df.empty:
        print("No annotated data available for retraining.")
        return
    
    X = df['text']
    y = df['positive']

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    print("Model retrained successfully.")

if __name__ == '__main__':
    retrain_model()
