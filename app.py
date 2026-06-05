import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import ollama
import faiss
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

# ==========================================
# 1. Text Cleaning Function (matching data_ingestion.py)
# ==========================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Strip HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove emojis and non-ASCII/unusual characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize case and strip
    text = text.strip().lower()
    
    return text

# ==========================================
# 2. Check Ollama Status
# ==========================================
def check_ollama_status(model_name="qwen2.5:7b"):
    try:
        models_list = ollama.list()
        available_models = [m.get("model", "") or m.get("name", "") for m in models_list.get("models", [])]
        
        # Check if requested model or any alternative is available
        status = {"running": True, "model_available": False, "available_models": available_models}
        
        if any(model_name in m for m in available_models):
            status["model_available"] = True
            status["selected_model"] = model_name
        else:
            # Check for alternative models (qwen, llama, mistral, etc.)
            for m in available_models:
                if any(k in m.lower() for k in ["qwen", "llama", "mistral", "gemma"]):
                    status["model_available"] = True
                    status["selected_model"] = m
                    break
                    
        return status
    except Exception:
        return {"running": False, "model_available": False, "available_models": []}

# ==========================================
# 3. Predict functions
# ==========================================
def predict_baseline(text, model, vectorizer):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    
    # Probability of the predicted label
    confidence = prob[pred]
    label_str = "FAKE" if pred == 1 else "REAL"
    
    return label_str, confidence

def predict_llm(text, model_name, live=True, baseline_pred=None):
    if not live:
        # Fallback to simulated prediction aligned with baseline (75% consistency)
        if baseline_pred:
            pred_label = baseline_pred
            if np.random.rand() > 0.90: # 10% flip for variation
                pred_label = "REAL" if baseline_pred == "FAKE" else "FAKE"
        else:
            pred_label = "FAKE" if np.random.rand() > 0.5 else "REAL"
            
        explanation = f"Simulated Explanation: The article contains stylistic elements (such as sentence patterns and punctuation frequency) commonly associated with {pred_label.lower()} reports."
        return pred_label, explanation
        
    prompt = f"""You are a fake news detection expert. Analyze the following news article and classify it as REAL or FAKE.

Article: {text[:1500]}

Respond in EXACTLY this format:
Label: REAL or FAKE
Explanation: [One concise sentence explaining your reasoning]"""

    try:
        response = ollama.generate(model=model_name, prompt=prompt, options={"temperature": 0.0})
        response_text = response.get("response", "").strip()
        
        label_match = re.search(r"Label:\s*(REAL|FAKE)", response_text, re.IGNORECASE)
        explanation_match = re.search(r"Explanation:\s*(.*)", response_text, re.IGNORECASE)
        
        label = label_match.group(1).upper() if label_match else "UNKNOWN"
        explanation = explanation_match.group(1).strip() if explanation_match else "Could not parse explanation."
        
        return label, explanation
    except Exception as e:
        return "UNKNOWN", f"Error generating explanation: {e}"

def predict_rag(text, model_name, evidence, live=True, baseline_pred=None):
    if not live:
        # Fallback to simulated prediction aligned with baseline
        if baseline_pred:
            pred_label = baseline_pred
            if np.random.rand() > 0.95: # 5% flip
                pred_label = "REAL" if baseline_pred == "FAKE" else "FAKE"
        else:
            pred_label = "FAKE" if np.random.rand() > 0.5 else "REAL"
            
        explanation = f"Simulated RAG Explanation: Grounded on retrieved fact-checks, this claim aligns with articles verified as {pred_label.lower()} news. The style and topics match fact-checked documents."
        return pred_label, explanation
        
    evidence_str = ""
    for i, ev in enumerate(evidence):
        evidence_str += f"Evidence {i+1}: [Title: {ev['title']}] {ev['text']} [Label: {ev['label']}]\n"
        
    prompt = f"""You are a fake news detection expert. Analyze the following article and classify it as REAL or FAKE.

Here is relevant evidence from known fact-checked articles:
{evidence_str}
Article to classify: {text[:1500]}

Respond in EXACTLY this format:
Label: REAL or FAKE
Explanation: [Cite the evidence that supports your judgment]"""

    try:
        response = ollama.generate(model=model_name, prompt=prompt, options={"temperature": 0.0})
        response_text = response.get("response", "").strip()
        
        label_match = re.search(r"Label:\s*(REAL|FAKE)", response_text, re.IGNORECASE)
        explanation_match = re.search(r"Explanation:\s*(.*)", response_text, re.IGNORECASE)
        
        label = label_match.group(1).upper() if label_match else "UNKNOWN"
        explanation = explanation_match.group(1).strip() if explanation_match else "Could not parse explanation."
        
        return label, explanation
    except Exception as e:
        return "UNKNOWN", f"Error generating explanation: {e}"

# ==========================================
# 4. Streamlit App Layout
# ==========================================
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Explainable Fake News Detector",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for dark-premium styling
    st.markdown("""
    <style>
        .reportview-container {
            background: #0e1117;
        }
        .main {
            background-color: #0f111a;
            color: #e2e8f0;
        }
        h1, h2, h3 {
            color: #f8fafc;
            font-family: 'Outfit', 'Inter', sans-serif;
        }
        .stButton>button {
            background-color: #3b82f6;
            color: white;
            border-radius: 8px;
            font-weight: 600;
            padding: 10px 24px;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #2563eb;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }
        .metric-card {
            background-color: #1e293b;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #334155;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            text-align: center;
        }
        .metric-val {
            font-size: 2rem;
            font-weight: 800;
            color: #3b82f6;
        }
        .metric-label {
            font-size: 0.875rem;
            color: #94a3b8;
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .badge-real {
            background-color: #065f46;
            color: #34d399;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 700;
            display: inline-block;
            border: 1px solid #059669;
        }
        .badge-fake {
            background-color: #7f1d1d;
            color: #f87171;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 700;
            display: inline-block;
            border: 1px solid #b91c1c;
        }
        .badge-unknown {
            background-color: #374151;
            color: #9ca3af;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 700;
            display: inline-block;
            border: 1px solid #4b5563;
        }
        .explanation-box {
            background-color: #1e293b;
            border-left: 4px solid #3b82f6;
            padding: 16px;
            border-radius: 0 8px 8px 0;
            margin-top: 12px;
            font-style: italic;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown("<h1 style='text-align: center; margin-bottom: 25px;'>🕵️‍♂️ Explainable Fake News Detection System</h1>", unsafe_allow_html=True)
    
    # Load resources
    @st.cache_resource
    def load_baseline_model():
        model_path = "models/tfidf_logreg.pkl"
        vec_path = "models/tfidf_vectorizer.pkl"
        if os.path.exists(model_path) and os.path.exists(vec_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(vec_path, "rb") as f:
                vectorizer = pickle.load(f)
            return model, vectorizer
        return None, None
        
    @st.cache_resource
    def load_rag_components():
        index_path = "models/faiss_index.bin"
        meta_path = "models/chunk_metadata.pkl"
        if os.path.exists(index_path) and os.path.exists(meta_path):
            index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return index, metadata, model
        return None, None, None
        
    baseline_model, tfidf_vec = load_baseline_model()
    faiss_index, rag_meta, embed_model = load_rag_components()
    
    # Sidebar
    st.sidebar.markdown("### ⚙️ Pipeline Configuration")
    model_choice = st.sidebar.selectbox(
        "Select Detection Method",
        ["TF-IDF + Logistic Regression (Baseline)", "Local LLM Only (Qwen3)", "Local LLM + RAG (Evidence-Grounded)"]
    )
    
    # Ollama status diagnostic
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔌 LLM Backend Status")
    
    ollama_status = check_ollama_status()
    if ollama_status["running"]:
        st.sidebar.success("Ollama: Running (Port 11434)")
        if ollama_status["model_available"]:
            st.sidebar.info(f"Model: {ollama_status['selected_model']}")
            live_llm = True
            selected_llm_model = ollama_status["selected_model"]
        else:
            st.sidebar.warning("Target model 'qwen2.5:7b' not pulled.")
            st.sidebar.markdown("Using alternative: " + (ollama_status["available_models"][0] if ollama_status["available_models"] else "None"))
            if ollama_status["available_models"]:
                live_llm = True
                selected_llm_model = ollama_status["available_models"][0]
            else:
                st.sidebar.error("Please run `ollama pull qwen2.5:7b`")
                live_llm = False
                selected_llm_model = None
    else:
        st.sidebar.error("Ollama: Not Running")
        st.sidebar.warning("LLM modes will run in simulated (fallback) mode.")
        live_llm = False
        selected_llm_model = None
        
    # Main Tabs
    tab1, tab2 = st.tabs(["🎯 Classification Tool", "📈 Performance Dashboard"])
    
    # ==========================================
    # Tab 1: Classification Tool
    # ==========================================
    with tab1:
        st.subheader("Paste News Article for Analysis")
        input_text = st.text_area(
            "Enter the full text of the article or news headline below:",
            height=250,
            placeholder="Paste news text here..."
        )
        
        analyze_btn = st.button("Analyze Article")
        
        if analyze_btn:
            if not input_text.strip():
                st.warning("Please enter some text to analyze.")
            elif baseline_model is None or tfidf_vec is None:
                st.error("Baseline model files not found. Please train models first by running scripts.")
            else:
                st.markdown("---")
                
                # Perform baseline prediction first (to use in fallbacks if needed)
                base_label, base_conf = predict_baseline(input_text, baseline_model, tfidf_vec)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 🔍 Analysis Results")
                    
                    if model_choice == "TF-IDF + Logistic Regression (Baseline)":
                        label = base_label
                        
                        if label == "REAL":
                            st.markdown("Label: <span class='badge-real'>REAL NEWS</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("Label: <span class='badge-fake'>FAKE NEWS</span>", unsafe_allow_html=True)
                            
                        st.markdown(f"**Confidence Score:** `{base_conf * 100:.2f}%`")
                        st.info("The baseline model analyzes styling and keyword frequency (TF-IDF features) to determine the probability of authenticity.")
                        
                    elif model_choice == "Local LLM Only (Qwen3)":
                        with st.spinner("LLM is analyzing the article..."):
                            label, explanation = predict_llm(
                                input_text, 
                                model_name=selected_llm_model, 
                                live=live_llm, 
                                baseline_pred=base_label
                            )
                            
                        if label == "REAL":
                            st.markdown("Label: <span class='badge-real'>REAL NEWS</span>", unsafe_allow_html=True)
                        elif label == "FAKE":
                            st.markdown("Label: <span class='badge-fake'>FAKE NEWS</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("Label: <span class='badge-unknown'>UNKNOWN</span>", unsafe_allow_html=True)
                            
                        st.markdown("#### 💬 Natural Language Explanation:")
                        st.markdown(f"<div class='explanation-box'>{explanation}</div>", unsafe_allow_html=True)
                        
                    else: # LLM + RAG
                        if faiss_index is None or rag_meta is None or embed_model is None:
                            st.error("RAG FAISS index or embedding model not found. Build RAG index first.")
                        else:
                            with st.spinner("Retrieving evidence and running grounded LLM..."):
                                # 1. Retrieve chunks
                                query_vector = embed_model.encode([input_text])
                                faiss.normalize_L2(query_vector)
                                distances, indices = faiss_index.search(query_vector, 5)
                                
                                evidence = []
                                for idx_val in indices[0]:
                                    if idx_val < len(rag_meta):
                                        evidence.append(rag_meta[idx_val])
                                        
                                # 2. Run grounded prediction
                                label, explanation = predict_rag(
                                    input_text,
                                    model_name=selected_llm_model,
                                    evidence=evidence,
                                    live=live_llm,
                                    baseline_pred=base_label
                                )
                                
                            if label == "REAL":
                                st.markdown("Label: <span class='badge-real'>REAL NEWS</span>", unsafe_allow_html=True)
                            elif label == "FAKE":
                                st.markdown("Label: <span class='badge-fake'>FAKE NEWS</span>", unsafe_allow_html=True)
                            else:
                                st.markdown("Label: <span class='badge-unknown'>UNKNOWN</span>", unsafe_allow_html=True)
                                
                            st.markdown("#### 💬 Evidence-Grounded Explanation:")
                            st.markdown(f"<div class='explanation-box'>{explanation}</div>", unsafe_allow_html=True)
                            
                            st.markdown("---")
                            st.markdown("### 📚 Retrieved Evidence (ISOT Fact-check Corpus)")
                            for idx, ev in enumerate(evidence):
                                ev_label = ev['label']
                                badge_class = "badge-real" if ev_label == "REAL" else "badge-fake"
                                
                                similarity_score = distances[0][idx]
                                
                                with st.expander(f"Evidence {idx+1}: {ev['title']} (Cosine Similarity: {similarity_score:.4f})"):
                                    st.markdown(f"**Document Ground-Truth:** <span class='{badge_class}' style='padding: 2px 8px; font-size:0.75rem;'>{ev_label}</span>", unsafe_allow_html=True)
                                    st.markdown(f"**Retrieved Passage Chunk:**\n>{ev['text']}")
                                    
                with col2:
                    st.markdown("### 📊 Contrastive Probability")
                    # Show comparison to baseline classifier
                    st.markdown(f"**Baseline Verdict:** `{base_label}`")
                    st.markdown(f"**Baseline Confidence:** `{base_conf * 100:.2f}%`")
                    
                    if not live_llm and model_choice != "TF-IDF + Logistic Regression (Baseline)":
                        st.warning("⚠️ Running in **Simulated Mode** because Ollama was offline. Start Ollama and pull your model to enable live neural reasoning.")
                        
    # ==========================================
    # Tab 2: Performance Dashboard
    # ==========================================
    with tab2:
        st.subheader("Model Evaluation & Comparison")
        
        # Display key metrics cards
        col1, col2, col3 = st.columns(3)
        
        # Check if evaluation metrics exist
        comp_path = "results/comparison_table.csv"
        if os.path.exists(comp_path):
            comp_df = pd.read_csv(comp_path)
            
            with col1:
                # Baseline F1
                val = comp_df.loc[comp_df["Model"] == "TF-IDF + Logistic Regression", "F1-score"].values[0]
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val'>{val * 100:.1f}%</div>
                    <div class='metric-label'>Baseline Classifier F1</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                # LLM F1
                val = comp_df.loc[comp_df["Model"] == "Qwen (LLM Only)", "F1-score"].values[0]
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val'>{val * 100:.1f}%</div>
                    <div class='metric-label'>LLM-Only Classifier F1</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                # RAG F1
                val = comp_df.loc[comp_df["Model"] == "Qwen + RAG", "F1-score"].values[0]
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val'>{val * 100:.1f}%</div>
                    <div class='metric-label'>Grounded LLM + RAG F1</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.markdown("### 📊 Metrics Comparison")
                st.dataframe(comp_df, hide_index=True)
                
                # Show comparison chart
                f1_chart = "results/f1_comparison.png"
                if os.path.exists(f1_chart):
                    st.image(f1_chart, caption="Detailed Model Performance Comparison")
                    
            with col_right:
                st.markdown("### 🧩 Confusion Matrices")
                conf_matrices = "results/confusion_matrices.png"
                if os.path.exists(conf_matrices):
                    st.image(conf_matrices, caption="Confusion Matrices for each model configuration")
        else:
            st.warning("Evaluation comparison data not found. Please run evaluation.py first.")
            
        # Social Network Analysis (SNA) Visualizations
        st.markdown("---")
        st.markdown("### 🕸️ Social Network Analysis (Keyword Co-occurrence)")
        st.write("Visualizes how key terms associate in Real vs Fake news networks. Nodes represent frequent keywords, sized by PageRank centrality. Edges represent co-occurrence in the same article.")
        
        sna_real_path = "results/sna_real.png"
        sna_fake_path = "results/sna_fake.png"
        sna_comp_path = "results/sna_centrality_comparison.png"
        
        if os.path.exists(sna_real_path) and os.path.exists(sna_fake_path):
            col_real, col_fake = st.columns(2)
            with col_real:
                st.image(sna_real_path, use_column_width=True)
            with col_fake:
                st.image(sna_fake_path, use_column_width=True)
                
            if os.path.exists(sna_comp_path):
                st.image(sna_comp_path, caption="PageRank Centrality Comparison of top 10 keywords")
        else:
            st.warning("SNA visualization files not found. Please run sna_analysis.py first.")

if __name__ == "__main__":
    main()
