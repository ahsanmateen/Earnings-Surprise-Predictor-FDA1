# AI-Based Earnings Surprise Predictor

This project is a data-driven tool that predicts whether a company will **beat**, **meet**, or **miss** earnings expectations. It uses a trained Random Forest model, integrates live financial data, and provides a visual dashboard for interpretation.

## 🔍 What It Does

- Fetches **live earnings data** from Yahoo Finance.
- Retrieves **macroeconomic indicators** (like interest rates) from FRED.
- Uses a trained **machine learning model** to classify earnings surprises.
- Displays **interactive visualizations** for predictions:
  - Radar chart of input features
  - Probability bar chart
  - Confidence gauge

## 📁 Folder Structure

```
project-folder/
│
├── ui_2.py                        # Main UI (Dash) file
├── earnings_surprise_predictor_2.py  # Model interface and feature logic
├── visualization.py              # Visualization functions (Plotly)
├── random_forest2.py             # (Optional) Model training script
│
├── fallback_top100.csv           # Fallback dataset for missing data
├── api_ready_model.pkl           # Trained Random Forest model
├── requirements.txt              # Required Python packages
└── README.md                     # This file
```

## ✅ How to Run

### 1. Install the required packages

```bash
pip install -r requirements.txt
```

### 2. Launch the dashboard

```bash
python ui_2.py
```

Open your browser and go to `http://127.0.0.1:8050/`

## ⚠️ Notes

- Make sure your internet connection is active (APIs fetch live data).
- If API data is unavailable, the model will fall back on `fallback_top100.csv`.
- The project uses a working FRED API key. If you're sharing this repository publicly, replace the key with a placeholder and mention it here.

## 📊 Dependencies

- dash  
- dash-bootstrap-components  
- pandas  
- plotly  
- yfinance  
- numpy  
- scikit-learn  
- joblib  
- requests  
- matplotlib  
- seaborn (only for training script)
