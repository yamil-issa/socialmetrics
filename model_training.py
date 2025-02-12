import joblib
from app import load_annotated_data, load_or_train_model

def retrain_model():
    df = load_annotated_data()
    if df.empty:
        print("No annotated data available for retraining.")
        return
    
    model, vectorizer = load_or_train_model()
    print("Model retrained successfully.")

if __name__ == '__main__':
    retrain_model()