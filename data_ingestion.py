import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Strip HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # 4. Remove emojis and non-ASCII/unusual characters (normalize to standard chars/punctuation)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # 5. Normalize whitespace (remove multiple spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # 6. Normalize case and strip
    text = text.strip().lower()
    
    return text

def main():
    print("Starting data ingestion and preprocessing...")
    
    # Paths
    raw_dir = "data/raw"
    cleaned_dir = "data/cleaned"
    rag_dir = "data/rag_corpus"
    
    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(rag_dir, exist_ok=True)
    
    # ==========================================
    # 1. Process WELFake Dataset (for model training)
    # ==========================================
    welfake_path = os.path.join(raw_dir, "WELFake_Dataset.csv")
    print(f"Loading WELFake from {welfake_path}...")
    welfake_df = pd.read_csv(welfake_path)
    
    # Inspect shape
    print(f"WELFake original shape: {welfake_df.shape}")
    
    # Drop rows where text is missing
    welfake_df = welfake_df.dropna(subset=["text"])
    
    # Fill missing titles with empty string
    welfake_df["title"] = welfake_df["title"].fillna("")
    
    # Clean text column
    print("Cleaning WELFake text (this may take a few minutes)...")
    welfake_df["clean_text"] = welfake_df["text"].apply(clean_text)
    welfake_df["clean_title"] = welfake_df["title"].apply(clean_text)
    
    # Drop rows where cleaned text is empty
    welfake_df = welfake_df[welfake_df["clean_text"] != ""]
    
    # Drop duplicates based on clean_text
    welfake_df = welfake_df.drop_duplicates(subset=["clean_text"])
    print(f"WELFake shape after cleaning and deduplication: {welfake_df.shape}")
    
    # Split into train/test (80% train, 20% test), stratified by label
    # Note: 0 = REAL, 1 = FAKE in this dataset version
    print("Splitting WELFake into stratified train/test sets...")
    train_df, test_df = train_test_split(
        welfake_df,
        test_size=0.2,
        random_state=42,
        stratify=welfake_df["label"]
    )
    
    train_path = os.path.join(cleaned_dir, "train.csv")
    test_path = os.path.join(cleaned_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved training data to {train_path} ({len(train_df)} rows)")
    print(f"Saved testing data to {test_path} ({len(test_df)} rows)")
    
    # ==========================================
    # 2. Process ISOT Dataset (for RAG corpus)
    # ==========================================
    fake_path = os.path.join(raw_dir, "Fake.csv")
    true_path = os.path.join(raw_dir, "True.csv")
    
    print(f"Loading ISOT Fake from {fake_path} and True from {true_path}...")
    isot_fake = pd.read_csv(fake_path)
    isot_true = pd.read_csv(true_path)
    
    # Assign labels: 0 = REAL, 1 = FAKE
    isot_fake["label"] = 1
    isot_true["label"] = 0
    
    # Merge datasets
    isot_df = pd.concat([isot_fake, isot_true], ignore_index=True)
    print(f"ISOT original shape: {isot_df.shape}")
    
    # Drop rows where text is missing
    isot_df = isot_df.dropna(subset=["text"])
    isot_df["title"] = isot_df["title"].fillna("")
    
    # Clean text column
    print("Cleaning ISOT text (this may take a few minutes)...")
    isot_df["clean_text"] = isot_df["text"].apply(clean_text)
    isot_df["clean_title"] = isot_df["title"].apply(clean_text)
    
    # Drop rows where cleaned text is empty
    isot_df = isot_df[isot_df["clean_text"] != ""]
    
    # Drop duplicates
    isot_df = isot_df.drop_duplicates(subset=["clean_text"])
    print(f"ISOT shape after cleaning and deduplication: {isot_df.shape}")
    
    isot_path = os.path.join(rag_dir, "isot.csv")
    isot_df.to_csv(isot_path, index=False)
    print(f"Saved RAG corpus to {isot_path} ({len(isot_df)} rows)")
    
    print("Data ingestion and preprocessing complete!")

if __name__ == "__main__":
    main()
