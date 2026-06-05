import os
import json
from collections import Counter
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Basic English stopwords to filter out along with standard news terms
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
    # Common news words that don't add semantic value
    'said', 'would', 'one', 'people', 'also', 'new', 'time', 'year', 'two', 'like', 'first', 'even', 'could',
    'many', 'told', 'last', 'made', 'get', 'back', 'make', 'say', 'way', 'since', 'well', 'many', 'state'
])

def extract_keywords_and_cooccurrences(df, num_keywords=30, sample_size=3000):
    # Sample to speed up processing
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Tokenize and count word frequencies
    word_counts = Counter()
    tokenized_docs = []
    
    for text in sample_df["clean_text"].fillna(""):
        words = [w for w in text.split() if w.isalpha() and len(w) > 2 and w not in STOPWORDS]
        word_counts.update(words)
        tokenized_docs.append(set(words)) # use set to check co-occurrence in article
        
    # Get top keywords
    top_keywords = [word for word, count in word_counts.most_common(num_keywords)]
    
    # Calculate co-occurrences
    co_occur = Counter()
    for doc in tokenized_docs:
        # Filter doc words to only contain top keywords
        key_words_in_doc = list(doc.intersection(top_keywords))
        for i in range(len(key_words_in_doc)):
            for j in range(i + 1, len(key_words_in_doc)):
                w1, w2 = sorted([key_words_in_doc[i], key_words_in_doc[j]])
                co_occur[(w1, w2)] += 1
                
    return top_keywords, co_occur

def analyze_network(keywords, co_occurrences, label_name, results_dir):
    # Create networkx Graph
    G = nx.Graph()
    G.add_nodes_from(keywords)
    
    for (w1, w2), weight in co_occurrences.items():
        if weight > 2: # Filter weak connections to keep graph clean
            G.add_edge(w1, w2, weight=weight)
            
    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    
    # Compute centralities
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G, weight='weight')
    
    # Store metrics for saving
    metrics = []
    for node in G.nodes():
        metrics.append({
            "keyword": node,
            "degree": degree_centrality.get(node, 0.0),
            "betweenness": betweenness_centrality.get(node, 0.0),
            "pagerank": pagerank.get(node, 0.0)
        })
        
    metrics_df = pd.DataFrame(metrics).sort_values(by="pagerank", ascending=False)
    metrics_df.to_csv(os.path.join(results_dir, f"sna_{label_name}_metrics.csv"), index=False)
    
    # Visualize network
    plt.figure(figsize=(10, 8), dpi=150)
    plt.title(f"Keyword Co-occurrence Network ({label_name.upper()} News)", fontsize=16, fontweight='bold', pad=15)
    
    # Layout
    pos = nx.spring_layout(G, k=0.4, seed=42)
    
    # Node sizing based on PageRank
    node_sizes = [pagerank[node] * 6000 for node in G.nodes()]
    
    # Edge widths based on co-occurrence weights
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(weights) if weights else 1
    edge_widths = [(w / max_weight) * 4 for w in weights]
    
    # Node colors based on centrality
    node_colors = [degree_centrality[node] for node in G.nodes()]
    
    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.coolwarm, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", font_color="#1a1a1a")
    
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"sna_{label_name}.png"), bbox_inches="tight")
    plt.close()
    
    return metrics_df

def main():
    print("Starting Social Network Analysis (SNA)...")
    
    # Paths
    train_path = "data/cleaned/train.csv"
    results_dir = "results"
    
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(train_path):
        print("Cleaned training data not found. Please run data_ingestion.py first.")
        return
        
    df = pd.read_csv(train_path)
    
    # Filter real vs fake
    real_df = df[df["label"] == 0]
    fake_df = df[df["label"] == 1]
    
    print("Extracting keywords and co-occurrences for Real News...")
    real_keywords, real_cooccur = extract_keywords_and_cooccurrences(real_df, num_keywords=30)
    print("Analyzing Real News network...")
    real_metrics = analyze_network(real_keywords, real_cooccur, "real", results_dir)
    
    print("Extracting keywords and co-occurrences for Fake News...")
    fake_keywords, fake_cooccur = extract_keywords_and_cooccurrences(fake_df, num_keywords=30)
    print("Analyzing Fake News network...")
    fake_metrics = analyze_network(fake_keywords, fake_cooccur, "fake", results_dir)
    
    # Generate PageRank centrality comparison plot
    print("Generating centrality comparison bar charts...")
    plt.figure(figsize=(12, 6))
    
    # Top 10 REAL
    plt.subplot(1, 2, 1)
    sns.barplot(data=real_metrics.head(10), x="pagerank", y="keyword", hue="keyword", legend=False, palette="Blues_r")
    plt.title("Top 10 Keywords PageRank (REAL News)")
    plt.xlabel("PageRank Centrality")
    plt.ylabel("")
    
    # Top 10 FAKE
    plt.subplot(1, 2, 2)
    sns.barplot(data=fake_metrics.head(10), x="pagerank", y="keyword", hue="keyword", legend=False, palette="Oranges_r")
    plt.title("Top 10 Keywords PageRank (FAKE News)")
    plt.xlabel("PageRank Centrality")
    plt.ylabel("")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "sna_centrality_comparison.png"), bbox_inches="tight")
    plt.close()
    
    print("SNA complete! Visualizations and metrics saved to results/")

if __name__ == "__main__":
    main()
