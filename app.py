import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib
import sys
import os
from streamlit_option_menu import option_menu

# Add src to path to import config
sys.path.append(os.path.abspath("src"))
import config
from dataset import preprocess_text

# Page config
st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="📊", layout="wide")
# Custom CSS for aesthetics
st.markdown("""
    <style>
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #0f172a, #020617);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(10px);
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #38bdf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif; 
        font-weight: 900;
        letter-spacing: -1px;
    }
    h2, h3 {color: #e2e8f0; font-family: 'Inter', sans-serif;}
    .reportview-container .main .block-container{padding-top: 2rem;}
    .stMetric {
        background: rgba(30, 41, 59, 0.4) !important; 
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 20px; 
        border-radius: 16px; 
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stMetric [data-testid="stMetricLabel"] p {color: #94a3b8 !important; font-size: 1.1rem !important;}
    .stMetric [data-testid="stMetricValue"] {color: #f8fafc !important; font-weight: 800 !important;}
    div.stButton > button {
        background: linear-gradient(45deg, #38bdf8, #818cf8) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.5) !important;
        transform: scale(1.02) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Glassmorphism layout template for Plotly graphs
GLASS_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e2e8f0')
)

# Cache data loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(config.CLEAN_DATA_PATH)
        # Ensure correct types
        df['target'] = df['target'].astype(int)
        df['target_label'] = df['target'].map({0: 'Negative', 1: 'Positive'})
        df['target_color'] = df['target_label']
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_resource
def load_models():
    try:
        model = joblib.load(config.MODEL_PATH)
        vectorizer = joblib.load(config.VECTORIZER_PATH)
        return model, vectorizer
    except Exception as e:
        return None, None

@st.cache_data
def load_insights():
    try:
        pos_df = pd.read_csv(f"{config.OUTPUT_DIR}/top_positive_keywords.csv")
        neg_df = pd.read_csv(f"{config.OUTPUT_DIR}/top_negative_keywords.csv")
        model_results = pd.read_csv(f"{config.OUTPUT_DIR}/model_results.csv")
        return pos_df, neg_df, model_results
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def main():
    st.title("💡 Sentiment Insights Engine")
    st.markdown("Analyze customer feedback, social media sentiment, and extract actionable insights through our AI-powered sentiment engine.")
    
    # Load all data
    df = load_data()
    model, vectorizer = load_models()
    pos_df, neg_df, model_results = load_insights()
    
    if df.empty or model is None:
        st.warning("⚠️ Data or Models not found. Please run the training pipeline first.")
        return
        
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #38bdf8; margin-bottom: 20px;'>Navigation</h2>", unsafe_allow_html=True)
        page = option_menu(
            menu_title=None,
            options=["Dashboard Overview", "Keyword Analysis", "Live Prediction"],
            icons=["house", "bar-chart", "activity"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#38bdf8", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color": "#e2e8f0", "--hover-color": "rgba(56, 189, 248, 0.1)"},
                "nav-link-selected": {"background-color": "rgba(56, 189, 248, 0.15)", "color": "#38bdf8", "border-right": "3px solid #38bdf8"},
            }
        )
    
    if page == "Dashboard Overview":
        st.header("1. Overview & Statistics")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Analyzed Records", f"{len(df):,}")
        
        pos_pct = (len(df[df['target'] == 1]) / len(df)) * 100
        col2.metric("Positive Sentiment", f"{pos_pct:.1f}%")
        
        avg_words = df['word_count'].mean()
        col3.metric("Avg. Words per Post", f"{avg_words:.1f}")
        
        st.markdown("---")
        
        # Row 2: Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            fig = px.pie(df, names='target_label', hole=0.4, color='target_label',
                         color_discrete_map={'Positive': '#10b981', 'Negative': '#ef4444'})
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), **GLASS_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Text Length vs Sentiment")
            fig = px.box(df, x='target_label', y='word_count', title="")
            fig.update_layout(showlegend=False, xaxis_title="Sentiment", yaxis_title="Word Count", **GLASS_LAYOUT)
            fig.update_traces(marker_color='#3b82f6')
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")
        
        st.header("2. Model Evaluation Results")
        if not model_results.empty:
            st.dataframe(model_results.style.format({"accuracy": "{:.4f}", "f1_score": "{:.4f}"}), use_container_width=True)
            
            fig = px.bar(model_results, x='model', y='f1_score', text_auto='.4f', 
                         title='F1 Score Comparison across Models')
            fig.update_layout(showlegend=False, **GLASS_LAYOUT)
            fig.update_traces(marker_color='#3b82f6')
            st.plotly_chart(fig, use_container_width=True)
            
    elif page == "Keyword Analysis":
        st.header("Top Drivers of Sentiment")
        st.markdown("These keywords have the highest influence on determining if a post is positive or negative.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Positive Keywords")
            if not pos_df.empty:
                fig = px.bar(pos_df.head(15), x='Coefficient', y='Keyword', orientation='h',
                             color='Coefficient', color_continuous_scale='Greens')
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, **GLASS_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
                
            st.subheader("Positive Word Cloud")
            pos_text = " ".join(df[df['target'] == 1]['clean_text'].dropna())
            if pos_text:
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(pos_text)
                fig_wc, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
                
        with col2:
            st.subheader("Top Negative Keywords")
            if not neg_df.empty:
                # Make coefficients positive for display purposes
                neg_df['Coefficient_Abs'] = neg_df['Coefficient'].abs()
                fig = px.bar(neg_df.head(15), x='Coefficient_Abs', y='Keyword', orientation='h',
                             color='Coefficient_Abs', color_continuous_scale='Reds')
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, **GLASS_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
                
            st.subheader("Negative Word Cloud")
            neg_text = " ".join(df[df['target'] == 0]['clean_text'].dropna())
            if neg_text:
                wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_text)
                fig_wc_neg, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc_neg)
                
    elif page == "Live Prediction":
        st.header("Test the Model")
        st.markdown("Enter custom text to see how the model analyzes the sentiment in real-time.")
        
        user_input = st.text_area("Enter your feedback/tweet:", height=150)
        
        if st.button("Analyze Sentiment", type="primary"):
            if user_input.strip() == "":
                st.warning("Please enter some text.")
            else:
                with st.spinner("Analyzing..."):
                    # Preprocess
                    clean_text = preprocess_text(user_input)
                    
                    # Vectorize
                    vec_text = vectorizer.transform([clean_text])
                    
                    # Predict
                    prediction = model.predict(vec_text)[0]
                    probabilities = model.predict_proba(vec_text)[0] if hasattr(model, 'predict_proba') else None
                    if not probabilities is not None and hasattr(model, 'decision_function'):
                        # Fake probability for LinearSVC using decision function
                        dist = model.decision_function(vec_text)[0]
                        prob_pos = 1 / (1 + np.exp(-dist))
                        probabilities = [1 - prob_pos, prob_pos]
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Result")
                        if prediction == 1:
                            st.success("Positive Sentiment 😊")
                        else:
                            st.error("Negative Sentiment 😞")
                            
                    with col2:
                        st.subheader("Confidence Scores")
                        if probabilities is not None:
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = probabilities[1] * 100,
                                title = {'text': "Confidence (Positive)"},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "#fca5a5"},
                                        {'range': [50, 100], 'color': "#86efac"}
                                    ]
                                }
                            ))
                            fig.update_layout(height=250, margin=dict(t=50, b=0, l=0, r=0), **GLASS_LAYOUT)
                            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
