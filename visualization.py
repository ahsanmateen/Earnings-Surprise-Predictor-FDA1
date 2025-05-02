import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# ========== RADAR CHART ==========
def plot_radar_chart(features: dict):
    """
    Generates a radar chart from normalized feature inputs (0–1 scale)
    """
    labels = list(features.keys())
    values = list(features.values())

    # Close loop
    labels.append(labels[0])
    values.append(values[0])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

    fig = go.Figure(
        data=[go.Scatterpolar(r=values, theta=labels, fill='toself', name='Feature Profile')]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Feature Profile (Normalized)",
        showlegend=False
    )
    return fig

# ========== PROBABILITY BAR CHART ==========
def plot_probability_bar(probabilities: list):
    """
    Plots model prediction probabilities for Miss, Meet, Beat.
    Input: list of 3 floats [p_miss, p_meet, p_beat]
    """
    labels = ["Miss", "Meet", "Beat"]
    fig = go.Figure(
        [go.Bar(x=labels, y=probabilities, marker_color=['red', 'gold', 'green'])]
    )
    fig.update_layout(
        title="Prediction Class Probabilities",
        yaxis=dict(title="Probability", range=[0, 1]),
        xaxis=dict(title="Predicted Class")
    )
    return fig

# ========== GAUGE / SPEEDOMETER ==========
def plot_prediction_gauge(predicted_label: str, confidence: float):
    """
    Draws a semicircular gauge showing model's top prediction and its confidence.
    """
    color_map = {"Miss": "red", "Meet": "gold", "Beat": "green"}
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%"},
        title={"text": f"Confidence in Prediction: {predicted_label}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color_map.get(predicted_label, "blue")},
            "steps": [
                {"range": [0, 33], "color": "#ffcccc"},
                {"range": [33, 66], "color": "#fff4cc"},
                {"range": [66, 100], "color": "#ccffcc"}
            ]
        }
    ))
    fig.update_layout(height=300)
    return fig


# Optional utility for normalizing inputs
def normalize_features(raw_features: dict):
    """
    Normalize fallback/live input features to [0, 1] scale for radar use.
    """
    bounds = {
        "earnings_surprise_ratio": (-20, 20),
        "absolute_surprise": (0, 5),
        "price_reaction": (-0.1, 0.1),
        "rolling_volatility": (0.005, 0.05),
        "interest_rate": (0.03, 0.06)
    }
    return {
        k: max(0, min(1, (v - bounds[k][0]) / (bounds[k][1] - bounds[k][0])))
        for k, v in raw_features.items()
    }

def generate_interpretation_text(features: dict, predicted_label: str) -> str:
    """
    Returns a concise explanation of the model's likely reasoning based on key input features.
    """
    phrases = []

    # Interpret price reaction
    if features.get("price_reaction") is not None:
        pr = features["price_reaction"]
        if pr > 0.05:
            phrases.append("a strong positive price reaction")
        elif pr < -0.05:
            phrases.append("a sharp negative price reaction")
        else:
            phrases.append("a muted price reaction")

    # Interpret earnings surprise ratio
    if features.get("earnings_surprise_ratio") is not None:
        es = abs(features["earnings_surprise_ratio"])
        if es > 0.2:
            phrases.append("a large earnings deviation")
        else:
            phrases.append("a modest earnings deviation")

    # Interpret volatility
    if features.get("rolling_volatility") is not None:
        vol = features["rolling_volatility"]
        if vol > 0.04:
            phrases.append("high volatility")
        elif vol < 0.01:
            phrases.append("low volatility")

    reason = ", ".join(phrases)
    return f"The prediction of '{predicted_label}' was influenced by {reason}."


# ========== TEST BLOCK (for direct execution) ==========
if __name__ == "__main__":
    raw_features = {
        "earnings_surprise_ratio": -4.25,
        "absolute_surprise": 3.72,
        "price_reaction": -0.04,
        "rolling_volatility": 0.015,
        "interest_rate": 0.044
    }
    probabilities = [0.02, 0.12, 0.86]
    predicted_label = "Beat"
    confidence = max(probabilities)

    norm_features = normalize_features(raw_features)

    # Show all charts in browser
    plot_radar_chart(norm_features).show()
    plot_probability_bar(probabilities).show()
    plot_prediction_gauge(predicted_label, confidence).show()


