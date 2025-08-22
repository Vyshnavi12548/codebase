# app.py
import random
import streamlit as st
import yfinance as yf
import requests
import joblib
import pandas as pd
import numpy as np
import shap
from transformers import pipeline
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Step 1: Set up the Streamlit UI and app title
# ----------------------------------------------------
st.set_page_config(page_title="Explainable Credit Intelligence Platform", layout="wide")
st.title("📈 Explainable Credit Intelligence Platform")
st.markdown("Enter a company name below to get a real-time, explainable credit score based on financial, news, and open-source data.")

# ----------------------------------------------------
# Step 2: Define API Keys (Replace with your own keys)
# ----------------------------------------------------
# NOTE: For a real app, use st.secrets for better security.
NEWS_API_KEY = "df692f2ed3d4467781325a999ea78221"
GITHUB_TOKEN = "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"

# ----------------------------------------------------
# Step 3: Implement Real-Time Data Ingestion Functions
# ----------------------------------------------------

def fetch_financial_data(ticker):
    """Fetches key financial metrics using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if 'marketCap' not in info or 'debtToEquity' not in info:
            st.warning(f"Could not fetch complete financial data for {ticker}. Please try another company.")
            return None
        
        financials = stock.financials.loc[['Total Revenue']]
        revenue_growth = 0
        if len(financials.columns) >= 2:
            current_revenue = financials.iloc[0, 0]
            previous_revenue = financials.iloc[0, 1]
            if previous_revenue > 0:
                revenue_growth = (current_revenue - previous_revenue) / previous_revenue
        
        return {
            'market_cap': info.get('marketCap', 0),
            'revenue_growth': revenue_growth,
            'debt_to_equity': info.get('debtToEquity', 0)
        }
    except Exception as e:
        st.error(f"Error fetching financial data: {e}")
        return None

# Cache the sentiment model to avoid reloading on every run
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_model = load_sentiment_model()

def get_news_sentiment(company_name):
    """Fetches news headlines and performs sentiment analysis."""
    try:
        url = f'https://newsapi.org/v2/everything?q="{company_name}"&sortBy=relevancy&language=en&apiKey={NEWS_API_KEY}'
        response = requests.get(url)
        data = response.json()
        
        if data.get('articles'):
            headlines = [article['title'] for article in data['articles'][:5]]
            sentiment_results = sentiment_model(headlines)
            sentiment_score = sum(1 if res['label'] == 'POSITIVE' else -1 for res in sentiment_results)
            return sentiment_score
        return 0
    except Exception as e:
        st.error(f"Error fetching news data: {e}")
        return 0

def fetch_github_data(org_name):
    """Fetches public GitHub repo data for a given organization."""
    try:
        url = f'https://api.github.com/orgs/{org_name}/repos'
        headers = {}
        if GITHUB_TOKEN:
            headers['Authorization'] = f'token {GITHUB_TOKEN}'
            
        response = requests.get(url, headers=headers)
        repos = response.json()
        
        if isinstance(repos, dict) and 'message' in repos:
            st.warning(f"GitHub API Error: {repos['message']}. Rate limit exceeded or organization not found.")
            return None
            
        total_stars = sum(repo.get('stargazers_count', 0) for repo in repos)
        total_forks = sum(repo.get('forks_count', 0) for repo in repos)
        
        return {
            'total_stars': total_stars,
            'total_forks': total_forks,
            'repo_count': len(repos)
        }
    except Exception as e:
        st.error(f"Error fetching GitHub data: {e}")
        return None

# ----------------------------------------------------
# Step 4: UI for User Input and Main Logic
# ----------------------------------------------------
ticker_map = {
    'tesla': {'ticker': 'TSLA', 'github_org': 'tesla'},
    'microsoft': {'ticker': 'MSFT', 'github_org': 'microsoft'},
    'apple': {'ticker': 'AAPL', 'github_org': 'apple'},
    'google': {'ticker': 'GOOGL', 'github_org': 'google'},
    'adobe': {'ticker': 'ADBE', 'github_org': 'adobe'},
    'snowflake': {'ticker': 'SNOW', 'github_org': 'Snowflake-Inc'},
    'meta': {'ticker': 'META', 'github_org': 'meta-os'},
    'netflix': {'ticker': 'NFLX', 'github_org': 'Netflix'},
    'nvidia': {'ticker': 'NVDA', 'github_org': 'NVIDIA'},
    'amazon': {'ticker': 'AMZN', 'github_org': 'amzn'},
    'intel': {'ticker': 'INTC', 'github_org': 'intel'},
}

col1, col2 = st.columns([3, 1])
with col1:
    company_name = st.text_input("Enter a company name (e.g., Tesla, Microsoft, Google):", "Microsoft")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_button = st.button("Get Score", type="primary")

if search_button and company_name:
    with st.spinner("Fetching and processing data..."):
        company_info = ticker_map.get(company_name.lower())

        if not company_info:
            st.warning(f"Company '{company_name}' not found in our database. Please try one of the suggestions.")
            st.stop()

        financial_data = fetch_financial_data(company_info['ticker'])
        news_sentiment = get_news_sentiment(company_name)
        github_data = fetch_github_data(company_info['github_org'])

        # Create feature vector for the model
        features = pd.DataFrame([{
            'revenue_growth': financial_data['revenue_growth'] if financial_data else 0,
            'debt_to_equity': financial_data['debt_to_equity'] if financial_data else 0,
            'news_sentiment': news_sentiment,
            'github_stars': github_data['total_stars'] if github_data else 0
        }])
        
        # ----------------------------------------------------
        # Step 5: Dynamic Scoring and Explainability Logic (with SHAP)
        # ----------------------------------------------------
        try:
            model = joblib.load('credit_score_model.pkl')
            score = model.predict(features)[0]

            # SHAP for advanced explainability
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features)
            
            # Use SHAP to generate a more detailed explanation
            st.subheader("Why This Score?")
            
            plt.style.use('dark_background')
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, features, show=False)
            st.pyplot(plt)
            
            st.subheader("Feature Contributions Summary")
            explanation_parts = []
            feature_contributions = []
            for i, val in enumerate(shap_values[0]):
                feature_name = features.columns[i]
                contribution_type = "Positive" if val > 0 else "Negative"
                explanation_parts.append(f"{contribution_type.lower()} contribution from {feature_name.replace('_', ' ')}")
                feature_contributions.append({'feature': feature_name.replace('_', ' '), 'contribution': contribution_type})

            st.write(f"The score is primarily influenced by {', '.join(explanation_parts)}.")

        except FileNotFoundError:
            st.error("The machine learning model file 'credit_score_model.pkl' was not found. Please run the train_model.py script first.")
            st.stop()

        # Generate mock history with real-time last score
        history_data = [
            {'date': '2025-01-01', 'score': random.randint(int(score) - 50, int(score) + 50)},
            {'date': '2025-02-01', 'score': random.randint(int(score) - 50, int(score) + 50)},
            {'date': '2025-03-01', 'score': random.randint(int(score) - 50, int(score) + 50)},
            {'date': '2025-04-01', 'score': random.randint(int(score) - 50, int(score) + 50)},
            {'date': '2025-05-01', 'score': random.randint(int(score) - 50, int(score) + 50)},
            {'date': '2025-06-01', 'score': score}
        ]
        
        # ----------------------------------------------------
        # Step 6: Display the results in the Streamlit UI
        # ----------------------------------------------------
        st.subheader("Credit Score")
        col_score, col_blank = st.columns([1, 4])
        with col_score:
            st.metric(label=f"Current Score for {company_name.capitalize()}", value=f"{score:.2f}")
        
        st.subheader("Feature Contributions")
        st.dataframe(feature_contributions)
        
        st.subheader("Score History")
        st.line_chart(history_data, x="date", y="score")

