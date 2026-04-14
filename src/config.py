import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "trainingandtestdata")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# File paths
RAW_DATA_PATH = os.path.join(DATA_DIR, "training.1600000.processed.noemoticon.csv")
CLEAN_DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_sampled_data.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sampling and Model parameters
SAMPLE_SIZE = 150000
RANDOM_STATE = 42

# Data schema
COLUMNS = ["target", "id", "date", "flag", "user", "text"]

