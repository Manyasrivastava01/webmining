import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    print("Starting evaluation and comparison...")
    
    # Paths
    results_dir = "results"
    
    baseline_path = os.path.join(results_dir, "baseline_results.json")
    llm_path = os.path.join(results_dir, "llm_results.json")
    rag_path = os.path.join(results_dir, "rag_results.json")
    
    # Check if results exist
    if not (os.path.exists(baseline_path) and os.path.exists(llm_path) and os.path.exists(rag_path)):
        print("Required evaluation results files not found. Ensure baseline, LLM, and RAG models have run.")
        return
        
    # Load results
    with open(baseline_path, "r") as f:
        baseline_res = json.load(f)
    with open(llm_path, "r") as f:
        llm_res = json.load(f)
    with open(rag_path, "r") as f:
        rag_res = json.load(f)
        
    # Create comparison table
    comparison_data = {
        "Model": ["TF-IDF + Logistic Regression", "Qwen (LLM Only)", "Qwen + RAG"],
        "Accuracy": [baseline_res["accuracy"], llm_res["accuracy"], rag_res["accuracy"]],
        "Precision": [baseline_res["precision"], llm_res["precision"], rag_res["precision"]],
        "Recall": [baseline_res["recall"], llm_res["recall"], rag_res["recall"]],
        "F1-score": [baseline_res["f1_score"], llm_res["f1_score"], rag_res["f1_score"]]
    }
    
    comp_df = pd.DataFrame(comparison_data)
    comp_df.to_csv(os.path.join(results_dir, "comparison_table.csv"), index=False)
    print("\nComparison Table Saved to results/comparison_table.csv:")
    print(comp_df.to_string(index=False))
    
    # Generate F1 & Accuracy comparison plot
    plt.figure(figsize=(10, 6), dpi=150)
    df_melted = pd.melt(comp_df, id_vars="Model", var_name="Metric", value_name="Value")
    
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df_melted, x="Metric", y="Value", hue="Model", palette="muted")
    plt.title("Performance Comparison of Detection Models", fontsize=16, fontweight='bold', pad=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.xlabel("")
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, color='black',
                        xytext=(0, 3),
                        textcoords='offset points')
                        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "f1_comparison.png"), bbox_inches="tight")
    plt.close()
    print("Saved metrics comparison plot to results/f1_comparison.png")
    
    # Generate confusion matrices side by side
    plt.figure(figsize=(15, 4), dpi=150)
    
    matrices = [
        (baseline_res["confusion_matrix"], "TF-IDF + LogReg"),
        (llm_res["confusion_matrix"], "Qwen (LLM Only)"),
        (rag_res["confusion_matrix"], "Qwen + RAG")
    ]
    
    for idx, (conf_mat, name) in enumerate(matrices):
        plt.subplot(1, 3, idx + 1)
        sns.heatmap(
            conf_mat, 
            annot=True, 
            fmt="d", 
            cmap="Blues" if idx == 0 else ("Oranges" if idx == 1 else "Greens"),
            xticklabels=["REAL", "FAKE"],
            yticklabels=["REAL", "FAKE"],
            cbar=False,
            annot_kws={"size": 12, "weight": "bold"}
        )
        plt.title(f"Confusion Matrix\n{name}", fontsize=12, fontweight='bold')
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrices.png"), bbox_inches="tight")
    plt.close()
    print("Saved confusion matrices plot to results/confusion_matrices.png")
    
    # Qualitative Comparison (sample 10 predictions)
    llm_preds_path = os.path.join(results_dir, "llm_predictions.csv")
    rag_preds_path = os.path.join(results_dir, "rag_predictions.csv")
    
    if os.path.exists(llm_preds_path) and os.path.exists(rag_preds_path):
        llm_preds = pd.read_csv(llm_preds_path)
        rag_preds = pd.read_csv(rag_preds_path)
        
        # Sample 10 articles (5 real, 5 fake)
        # Ensure we look at the same index
        real_indices = llm_preds[llm_preds["label"] == 0].head(5).index.tolist()
        fake_indices = llm_preds[llm_preds["label"] == 1].head(5).index.tolist()
        sample_indices = real_indices + fake_indices
        
        qualitative = []
        for idx in sample_indices:
            title = llm_preds.loc[idx, "title"]
            label = "REAL" if llm_preds.loc[idx, "label"] == 0 else "FAKE"
            
            llm_pred = "REAL" if llm_preds.loc[idx, "llm_pred"] == 0 else "FAKE"
            llm_exp = llm_preds.loc[idx, "llm_explanation"]
            
            rag_pred = "REAL" if rag_preds.loc[idx, "rag_pred"] == 0 else "FAKE"
            rag_exp = rag_preds.loc[idx, "rag_explanation"]
            
            qualitative.append({
                "Title": title,
                "Ground Truth": label,
                "LLM Pred": llm_pred,
                "LLM Explanation": llm_exp,
                "RAG Pred": rag_pred,
                "RAG Explanation": rag_exp
            })
            
        qual_df = pd.DataFrame(qualitative)
        qual_df.to_csv(os.path.join(results_dir, "qualitative_comparison.csv"), index=False)
        print("Saved qualitative comparison of explanations to results/qualitative_comparison.csv")
        
    print("Evaluation and comparison complete!")

if __name__ == "__main__":
    main()
