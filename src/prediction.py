"""Prediction helpers for assigning new customers to trained segments."""

import pandas as pd


def predict_customer(pipeline, features, values):
    """Return a segment index for one customer using the fitted preprocessing pipeline."""
    row = pd.DataFrame([{feature: values[feature] for feature in features}])
    return int(pipeline.predict(row)[0])
