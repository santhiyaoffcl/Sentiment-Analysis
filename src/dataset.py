import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import config
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure word corpus is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def load_data(filepath, sample_size=config.SAMPLE_SIZE):
    """Loads the dataset, assigns columns, and samples it to speed up execution."""
    logging.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='latin-1', names=config.COLUMNS)
        logging.info(f"Loaded {len(df)} rows. Downsampling to {sample_size} rows...")
        
        # Keep only positive (4) and negative (0) sentiments, in case there are neutrals
        df = df[df['target'].isin([0, 4])]
        
        n_samples = min(sample_size, len(df))
        n_per_group = int(n_samples / 2)
        df_sampled = df.groupby('target', group_keys=False).sample(n=n_per_group, random_state=config.RANDOM_STATE)
        
        # Reset index and map target to 0 (Negative) and 1 (Positive)
        df_sampled = df_sampled.reset_index(drop=True)
        df_sampled['target'] = df_sampled['target'].map({0: 0, 4: 1})
        logging.info(f"Data sampling complete. New shape: {df_sampled.shape}")
        
        return df_sampled
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def preprocess_text(text):
    """Cleans text: lowercase, removes URLs, user mentions, non-alphabetic chars, and stopwords."""
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove User Mentions
    text = re.sub(r'@\w+', '', text)
    # Remove non-alphabet characters and replace with space
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stop words
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    return ' '.join(words)

def prepare_pipeline():
    """Runs the loading and cleaning pipeline and saves output."""
    df = load_data(config.RAW_DATA_PATH)
    
    logging.info("Preprocessing text data...")
    # Using simple apply since 150k shouldn't take too long (typically ~10 seconds)
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # Remove any empty rows after cleaning
    df = df[df['clean_text'].str.strip() != '']
    
    # Calculate word count for EDA
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    
    logging.info(f"Saving cleaned dataset to {config.CLEAN_DATA_PATH}...")
    df.to_csv(config.CLEAN_DATA_PATH, index=False)
    logging.info("Data preparation completed successfully.")

if __name__ == "__main__":
    prepare_pipeline()
