import os
import json
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def main():
    print("Starting TF-IDF + Logistic Regression training...")
    
    # Paths
    cleaned_dir = "data/cleaned"
    models_dir = "models"
    results_dir = "results"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    train_path = os.path.join(cleaned_dir, "train.csv")
    test_path = os.path.join(cleaned_dir, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Cleaned data files not found. Please run data_ingestion.py first.")
        return
        
    print("Loading train and test datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Fill any empty values in clean_text
    train_df["clean_text"] = train_df["clean_text"].fillna("")
    test_df["clean_text"] = test_df["clean_text"].fillna("")
    
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    
    # TF-IDF Vectorization
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    y_train = train_df["label"].values
    
    X_test = vectorizer.transform(test_df["clean_text"])
    y_test = test_df["label"].values
    
    # Train Logistic Regression
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    print("Evaluating baseline model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    conf_mat = confusion_matrix(y_test, y_pred).tolist()
    class_report = classification_report(y_test, y_pred, target_names=["FAKE", "REAL"])
    
    print("\nBaseline Model Performance:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:\n", class_report)
    
    # Save model and vectorizer
    model_path = os.path.join(models_dir, "tfidf_logreg.pkl")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
        
    print(f"Saved model to {model_path}")
    print(f"Saved vectorizer to {vectorizer_path}")
    
    # Save results to JSON
    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": conf_mat,
        "classification_report": class_report
    }
    
    results_path = os.path.join(results_dir, "baseline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved results to {results_path}")
    print("Baseline training complete!")

if __name__ == "__main__":
    main()
