"""Reusable K-Means training, evaluation, and model persistence."""

from pathlib import Path
from typing import Dict, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _preprocessor(data: pd.DataFrame, features: Sequence[str]) -> ColumnTransformer:
    numeric_features = [
        column for column in features if pd.api.types.is_numeric_dtype(data[column])
    ]
    categorical_features = [
        column for column in features if column not in numeric_features
    ]
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_kmeans(
    data: pd.DataFrame,
    features: Sequence[str],
    n_clusters: int,
    random_state: int = 42,
) -> Dict:
    """Fit a scaled, imputed K-Means pipeline and evaluate candidate cluster counts."""
    features = list(features)
    if len(features) < 2:
        raise ValueError("Select at least two features to train a segmentation model.")
    if len(data) < 3:
        raise ValueError("At least three customer rows are needed for clustering.")
    if any(feature not in data.columns for feature in features):
        raise ValueError("A selected feature is not available in the current dataset.")
    if any(not data[feature].notna().any() for feature in features):
        raise ValueError("Selected features must contain at least one observed value.")
    if not 2 <= n_clusters < len(data):
        raise ValueError("The number of clusters must be at least 2 and below row count.")

    distinct_rows = int(data[features].drop_duplicates().shape[0])
    if distinct_rows < n_clusters:
        raise ValueError(
            "The selected features contain fewer distinct customer profiles than the "
            "requested cluster count. Choose different features or a smaller number of clusters."
        )
    preprocessor = _preprocessor(data, features)
    transformed = preprocessor.fit_transform(data[features])
    if transformed.shape[1] == 0:
        raise ValueError("The selected columns could not be converted into model features.")

    max_k = min(10, len(data) - 1, distinct_rows)
    baseline = KMeans(n_clusters=1, random_state=random_state, n_init=10)
    baseline.fit(transformed)
    candidates = [
        {
            "clusters": 1,
            "inertia": float(baseline.inertia_),
            "silhouette": float("nan"),
        }
    ]
    for k in range(2, max_k + 1):
        candidate = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = candidate.fit_predict(transformed)
        score = (
            float(
                silhouette_score(
                    transformed,
                    labels,
                    sample_size=min(len(data), 5000),
                    random_state=random_state,
                )
            )
            if len(np.unique(labels)) > 1
            else float("nan")
        )
        candidates.append(
            {
                "clusters": k,
                "inertia": float(candidate.inertia_),
                "silhouette": score,
            }
        )

    estimator = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = estimator.fit_predict(transformed)
    if len(np.unique(labels)) != n_clusters:
        raise ValueError(
            "The selected features contain fewer distinct groups than the requested "
            "cluster count. Choose different features or a smaller number of clusters."
        )
    silhouette = float(
        silhouette_score(
            transformed,
            labels,
            sample_size=min(len(data), 5000),
            random_state=random_state,
        )
    )
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("kmeans", estimator)]
    )
    return {
        "pipeline": pipeline,
        "labels": labels,
        "features": features,
        "n_clusters": n_clusters,
        "inertia": float(estimator.inertia_),
        "silhouette": silhouette,
        "evaluation": pd.DataFrame(candidates),
    }


def save_model(result: Dict, path: str) -> Path:
    """Persist the fitted model and its input schema using Joblib."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": result["pipeline"],
            "features": result["features"],
            "n_clusters": result["n_clusters"],
            "silhouette": result["silhouette"],
        },
        destination,
    )
    return destination
