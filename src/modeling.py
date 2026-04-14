import pandas as pd
import numpy as np
import logging
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_models():
    """Trains TF-IDF models and evaluates three classifiers."""
    logging.info(f"Loading cleaned dataset from {config.CLEAN_DATA_PATH}...")
    df = pd.read_csv(config.CLEAN_DATA_PATH)
    
    # Fill NaN values in clean_text
    df['clean_text'] = df['clean_text'].fillna('')
    
    X = df['clean_text']
    y = df['target']
    
    logging.info("Splitting dataset into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y)
    
    logging.info("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    logging.info(f"Feature set size: {X_train_vec.shape}")
    
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
        "Linear SVC": LinearSVC(random_state=config.RANDOM_STATE, dual=False)
    }
    
    best_f1 = 0
    best_model_name = ""
    best_model = None
    
    results = []
    
    for name, model in models.items():
        logging.info(f"Training {name}...")
        model.fit(X_train_vec, y_train)
        
        y_pred = model.predict(X_test_vec)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            "model": name,
            "accuracy": acc,
            "f1_score": f1
        })
        
        logging.info(f"{name} Results: Acc={acc:.4f}, F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    logging.info(f"Best model determined: {best_model_name} with F1 score: {best_f1:.4f}")
    
    logging.info("Saving best model and vectorizer...")
    joblib.dump(best_model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    
    logging.info("Models saved successfully.")
    
    # Save the evaluation results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{config.OUTPUT_DIR}/model_results.csv", index=False)
    
if __name__ == "__main__":
    train_models()
