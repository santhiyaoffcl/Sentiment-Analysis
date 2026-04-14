import pandas as pd
import numpy as np
import joblib
import logging
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_top_features(vectorizer, model, n=20):
    """Returns the top positive and negative features based on model coefficients.
       Assumes a linear model like Logistic Regression or Linear SVC.
    """
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        
        # Sort coefficients
        top_positive_indices = coefs.argsort()[-n:][::-1]
        top_negative_indices = coefs.argsort()[:n]
        
        positive_features = [(feature_names[i], coefs[i]) for i in top_positive_indices]
        negative_features = [(feature_names[i], coefs[i]) for i in top_negative_indices]
        
        return positive_features, negative_features
    else:
        logging.warning("Model does not have coefficients (e.g., Naive Bayes).")
        return [], []

def extract_insights():
    """Generates business insights from the trained model and data."""
    logging.info("Loading model and vectorizer for insight extraction...")
    try:
        model = joblib.load(config.MODEL_PATH)
        vectorizer = joblib.load(config.VECTORIZER_PATH)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return
        
    logging.info("Extracting top features...")
    pos_features, neg_features = get_top_features(vectorizer, model, n=30)
    
    # Save insights to CSV to be used by the Streamlit App
    pos_df = pd.DataFrame(pos_features, columns=["Keyword", "Coefficient"])
    neg_df = pd.DataFrame(neg_features, columns=["Keyword", "Coefficient"])
    
    pos_df.to_csv(f"{config.OUTPUT_DIR}/top_positive_keywords.csv", index=False)
    neg_df.to_csv(f"{config.OUTPUT_DIR}/top_negative_keywords.csv", index=False)
    
    logging.info("Successfully extracted and saved insights.")

if __name__ == "__main__":
    extract_insights()
