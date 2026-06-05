# 🕵️‍♂️ Explainable Fake News Detection System

A hybrid fake news detection application combining statistical baselines, local Large Language Models (LLMs), and Retrieval-Augmented Generation (RAG) to classify news articles and explain the reasoning behind the verdicts.

---

## 🚀 Key Features

* **Three-Tier Detection Pipeline:**
  1. **TF-IDF + Logistic Regression (Baseline):** A fast statistical classifier analyzing style and keyword frequencies.
  2. **Local LLM Only:** Context-aware zero-shot classification with natural language explanations.
  3. **Local LLM + RAG (Evidence-Grounded):** Fetches fact-checked evidence from the ISOT corpus using a FAISS vector index to back up classifications.
* **Interactive Web UI:** Clean, dark-mode Streamlit dashboard with pipeline selection, Ollama backend diagnostics, and real-time inference.
* **Performance Dashboard:** Evaluates metrics (Accuracy, Precision, Recall, F1-Score), displays confusion matrices, and features a **Social Network Analysis (SNA)** keyword co-occurrence graph highlighting structural word associations in Real vs. Fake news.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
* **Python 3.8+**
* **Ollama** installed on your system (Download from [ollama.com](https://ollama.com)).

### 2. Setup environment
Open your terminal inside the project directory and run:

```bash
# Install required Python dependencies
pip install -r requirements.txt
```
*(Note: If a requirements file is not present, make sure you have `streamlit`, `ollama`, `faiss-cpu`, `sentence-transformers`, `scikit-learn`, `pandas`, `numpy`, `networkx`, `matplotlib`, `seaborn`, and `beautifulsoup4` installed).*

### 3. Setup the Local LLM
Ensure Ollama is running (check your system tray), then download the required model:
```bash
ollama pull qwen2.5:7b
```

### 4. Running the Web Application
Start the Streamlit server:
```bash
python -m streamlit run app.py --server.headless true
```
Open your browser and navigate to **`http://localhost:8501`**.

---

## 📂 Project Structure & Pipelines

* **`app.py`**: The Streamlit web application interface and prediction routers.
* **`data_ingestion.py`**: Ingests the WELFake and ISOT datasets, cleans/normalizes the text, and prepares train/test splits.
* **`baseline_model.py`**: Trains the TF-IDF + Logistic Regression baseline model.
* **`llm_detector.py`**: Runs evaluations for the LLM-only classification pathway.
* **`rag_module.py`**: Implements document chunking, embedding generation (`all-MiniLM-L6-v2`), FAISS index construction, and RAG reasoning.
* **`sna_analysis.py`**: Extracts top keywords, computes PageRank/degree/betweenness centralities, and builds the keyword co-occurrence network visualizations.
* **`evaluation.py`**: Compiles prediction results across all configurations and exports the model comparison metrics.

---

## 🧠 Why a Baseline Might Incorrectly Label News
Statistical models like TF-IDF only analyze word frequencies:
* They are highly susceptible to **Intercept Bias** on short text inputs.
* They suffer from **Out-of-Vocabulary (OOV)** issues, completely ignoring specific entities (like names, places, and recent dates) that weren't present in the training set.

Switching to the **Local LLM** or **RAG** mode ensures semantic-level understanding and reference grounding, providing a much higher classification accuracy.
