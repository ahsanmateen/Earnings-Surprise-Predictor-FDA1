import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd

from earnings_surprise_predictor_2 import predict_surprise
from visualization import (
    normalize_features,
    plot_radar_chart,
    plot_probability_bar,
    plot_prediction_gauge,
    generate_interpretation_text,
)

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    html.H1("Earnings Surprise Classifier", className="text-center mt-4"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Enter Stock Ticker (e.g., AAPL):"),
            dcc.Input(id="ticker-input", type="text", debounce=True, className="form-control"),
            html.Br(),
            html.Button("See Results", id="submit-button", n_clicks=0, className="btn btn-primary mt-2")
        ], width=4)
    ], className="mb-4"),

    dbc.Spinner([
        html.Div(id="prediction-output", className="mb-4"),
        dcc.Graph(id="radar-chart"),
        dcc.Graph(id="probability-bar"),
        dcc.Graph(id="gauge-chart")
    ])
], fluid=True)

# Callback
@app.callback(
    Output("prediction-output", "children"),
    Output("radar-chart", "figure"),
    Output("probability-bar", "figure"),
    Output("gauge-chart", "figure"),
    Input("submit-button", "n_clicks"),
    State("ticker-input", "value")
)
def update_output(n_clicks, ticker):
    if not ticker:
        return "Please enter a ticker.", go.Figure(), go.Figure(), go.Figure()

    try:
        # Get prediction 
        pred_label, feats, source, fallback_label, proba = predict_surprise(ticker)
        confidence = max(proba)
        interpretation = generate_interpretation_text(feats, pred_label)
        interpretation_text = html.P(interpretation, style={"fontStyle": "italic", "marginTop": "1rem"})


        # Visualizations
        normalized_feats = normalize_features(feats)
        radar_fig = plot_radar_chart(normalized_feats)
        bar_fig = plot_probability_bar(proba)
        gauge_fig = plot_prediction_gauge(pred_label, confidence)

        # UI Text
        pred_text = html.H4(f"Predicted Surprise Category: {pred_label}", className="text-success")
        source_text = html.P(f"Prediction Source: {source}", className="text-muted")
        
        return [pred_text, source_text, interpretation_text], radar_fig, bar_fig, gauge_fig


    except Exception as e:
        return f"Error: {str(e)}", go.Figure(), go.Figure(), go.Figure()

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
