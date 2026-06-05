import os
import re
import json
import pandas as pd
import ollama
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def query_ollama(text, model="qwen2.5:7b"):
    prompt = f"""You are a fake news detection expert. Analyze the following news article and classify it as REAL or FAKE.

Article: {text[:1500]}

Respond in EXACTLY this format:
Label: REAL or FAKE
Explanation: [One concise sentence explaining your reasoning]"""

    try:
        response = ollama.generate(model=model, prompt=prompt, options={"temperature": 0.0})
        response_text = response.get("response", "").strip()
        return response_text
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return None

def parse_llm_response(response_text):
    if not response_text:
        return "UNKNOWN", "No response from LLM"
        
    label_match = re.search(r"Label:\s*(REAL|FAKE)", response_text, re.IGNORECASE)
    explanation_match = re.search(r"Explanation:\s*(.*)", response_text, re.IGNORECASE)
    
    label = label_match.group(1).upper() if label_match else "UNKNOWN"
    explanation = explanation_match.group(1).strip() if explanation_match else "Could not parse explanation."
    
    if label not in ["REAL", "FAKE"]:
        label = "UNKNOWN"
        
    return label, explanation

def main():
    print("Starting LLM-Based Detection...")
    
    # Paths
    test_path = "data/cleaned/test.csv"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(test_path):
        print("Cleaned test data not found. Please run data_ingestion.py first.")
        return
        
    # Load test dataset
    df = pd.read_csv(test_path)
    
    # Balanced sampling of 300 articles (150 Real, 150 Fake)
    # Label mapping: 0 = REAL, 1 = FAKE
    print("Sampling 300 balanced test articles...")
    sample_real = df[df["label"] == 0].sample(n=150, random_state=42)
    sample_fake = df[df["label"] == 1].sample(n=150, random_state=42)
    sample_df = pd.concat([sample_real, sample_fake], ignore_index=True)
    
    # Check if Ollama is running and has models
    model_name = "qwen2.5:7b"
    try:
        models_list = ollama.list()
        available_models = [m.get("model", "") or m.get("name", "") for m in models_list.get("models", [])]
        print(f"Available Ollama models: {available_models}")
        
        # If default model is not there, check for alternatives or try to pull
        if not any(model_name in m for m in available_models):
            # Try to see if qwen or llama is available and use it
            found = False
            for m in available_models:
                if "qwen" in m or "llama" in m or "mistral" in m:
                    model_name = m
                    found = True
                    break
            if not found:
                print(f"Warning: Default model '{model_name}' not found. Attempting to pull it. This may take a while...")
                ollama.pull(model_name)
    except Exception as e:
        print(f"\nWARNING: Ollama service is not running or accessible: {e}")
        print("Please start Ollama and ensure port 11434 is listening.")
        print("Creating mock predictions for demonstration and evaluation scripts.")
        
        # Create mock predictions for development/pipeline testing
        mock_predictions(sample_df, results_dir)
        return

    print(f"Using model: {model_name}")
    
    predictions = []
    explanations = []
    raw_responses = []
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Classifying articles"):
        text = row["clean_text"]
        raw_resp = query_ollama(text, model=model_name)
        pred_label_str, exp = parse_llm_response(raw_resp)
        
        # Map back to integer labels (0 = REAL, 1 = FAKE)
        # Default to FAKE if UNKNOWN
        pred_label = 0 if pred_label_str == "REAL" else 1
        
        predictions.append(pred_label)
        explanations.append(exp)
        raw_responses.append(raw_resp if raw_resp else "")
        
    sample_df["llm_pred"] = predictions
    sample_df["llm_explanation"] = explanations
    sample_df["llm_raw_response"] = raw_responses
    
    # Calculate metrics
    y_true = sample_df["label"].values
    y_pred = sample_df["llm_pred"].values
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    conf_mat = confusion_matrix(y_true, y_pred).tolist()
    class_report = classification_report(y_true, y_pred, target_names=["REAL", "FAKE"])
    
    print("\nLLM Model Performance:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    # Save predictions
    predictions_path = os.path.join(results_dir, "llm_predictions.csv")
    sample_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")
    
    # Save metrics
    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": conf_mat,
        "classification_report": class_report
    }
    
    results_path = os.path.join(results_dir, "llm_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved metrics to {results_path}")

def mock_predictions(sample_df, results_dir):
    # Mock labels with ~75% accuracy
    import numpy as np
    np.random.seed(42)
    
    true_labels = sample_df["label"].values
    predictions = []
    for label in true_labels:
        # 75% chance of correct label
        if np.random.rand() < 0.75:
            predictions.append(label)
        else:
            predictions.append(1 - label)
            
    sample_df["llm_pred"] = predictions
    sample_df["llm_explanation"] = sample_df.apply(
        lambda r: f"Mock explanation: The article has characteristics matching {'REAL' if r['llm_pred']==0 else 'FAKE'} news.", axis=1
    )
    sample_df["llm_raw_response"] = sample_df.apply(
        lambda r: f"Label: {'REAL' if r['llm_pred']==0 else 'FAKE'}\nExplanation: {r['llm_explanation']}", axis=1
    )
    
    acc = accuracy_score(true_labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="binary")
    conf_mat = confusion_matrix(true_labels, predictions).tolist()
    class_report = classification_report(true_labels, predictions, target_names=["REAL", "FAKE"])
    
    sample_df.to_csv(os.path.join(results_dir, "llm_predictions.csv"), index=False)
    
    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": conf_mat,
        "classification_report": class_report,
        "mocked": True
    }
    
    with open(os.path.join(results_dir, "llm_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("Saved MOCK predictions and metrics successfully.")

if __name__ == "__main__":
    main()
