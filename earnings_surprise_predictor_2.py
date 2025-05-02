# Imports
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Constants
FRED_API_KEY = "81a6f6b3dfe1b6086fcb2df55c5a8c2f"
FRED_SERIES_ID = "DGS10"
MODEL_PATH = "api_ready_model.pkl"
FALLBACK_PATH = "fallback_top100.csv"

# Load model and fallback
model = joblib.load(MODEL_PATH)
fallback_data = pd.read_csv(FALLBACK_PATH)


# Function to fetch latest interest rate from FRED
def get_fred_interest_rate():
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": FRED_SERIES_ID,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "limit": 1,
            "sort_order": "desc"
        }
        response = requests.get(url, params=params)
        data = response.json()
        return float(data["observations"][0]["value"])
    except Exception:
        return None

# Fetch live data from yfinance
def get_live_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="60d")
        eps = stock.info.get("earningsPerShare", None)
        rolling_vol = hist["Close"].pct_change().rolling(window=30).std().dropna()
        volatility = rolling_vol.iloc[-1] if not rolling_vol.empty else None
        price_reaction = hist["Close"].pct_change(periods=3).iloc[-1] if len(hist) > 3 else None
        return {
            "eps": eps,
            "rolling_volatility": volatility,
            "price_reaction": price_reaction
        }
    except Exception:
        return None

# Build feature set
def compute_features(eps, estimate, volatility, price_reaction, interest_rate):
    surprise_ratio = (eps - estimate) / abs(estimate) if eps is not None and estimate else None
    abs_surprise = abs(eps - estimate) if eps is not None and estimate else None
    return {
        "earnings_surprise_ratio": surprise_ratio,
        "absolute_surprise": abs_surprise,
        "rolling_volatility": volatility,
        "price_reaction": price_reaction,
        "interest_rate": interest_rate
    }

# Main prediction function
def predict_surprise(ticker, analyst_estimate=None):
    ticker = ticker.upper()
    live_data = get_live_data(ticker)
    interest_rate = get_fred_interest_rate()

    fallback_row = fallback_data[fallback_data["tic"] == ticker]
    fallback_exists = not fallback_row.empty
    fallback_label = fallback_row.iloc[0]["surprise_category"] if fallback_exists and "surprise_category" in fallback_row.columns else "N/A"

    if not live_data or not live_data.get("eps"):
        if not fallback_exists:
            raise ValueError(f"No fallback data available for ticker: {ticker}")
        fallback = fallback_row.iloc[0]
        features = {
            "earnings_surprise_ratio": fallback["earnings_surprise_ratio"],
            "absolute_surprise": fallback["absolute_surprise"],
            "rolling_volatility": fallback["rolling_volatility"],
            "price_reaction": fallback["price_reaction"],
            "interest_rate": fallback["interest_rate"]
        }
        source = "Fallback"
    else:
        estimate = analyst_estimate if analyst_estimate else 2.00
        features = compute_features(
            eps=live_data["eps"],
            estimate=estimate,
            volatility=live_data["rolling_volatility"],
            price_reaction=live_data["price_reaction"],
            interest_rate=interest_rate if interest_rate else 0.044
        )
        source = "Live API"

    # Predict using model
    ordered_features = [
        "earnings_surprise_ratio",
        "absolute_surprise",
        "price_reaction",
        "rolling_volatility",
        "interest_rate"
    ]
    df_features = pd.DataFrame([[features[feat] for feat in ordered_features]], columns=ordered_features)
    prediction = model.predict(df_features)[0]
    probabilities = model.predict_proba(df_features)[0].tolist()
    return prediction, features, source, fallback_label, probabilities

