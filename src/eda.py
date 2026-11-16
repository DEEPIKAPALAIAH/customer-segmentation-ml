"""Exploratory-analysis summaries used by the dashboard."""

import pandas as pd


def numeric_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return a readable numeric summary, including missingness."""
    numeric = data.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe().T
    summary["missing"] = numeric.isna().sum()
    return summary.reset_index(names="feature")


def categorical_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return category counts for non-numeric fields."""
    rows = []
    for column in data.select_dtypes(exclude="number").columns:
        rows.append(
            {
                "feature": column,
                "unique_values": int(data[column].nunique(dropna=True)),
                "most_common": (
                    str(data[column].mode(dropna=True).iloc[0])
                    if not data[column].mode(dropna=True).empty
                    else "Not available"
                ),
                "missing": int(data[column].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate pairwise numeric feature correlations."""
    numeric = data.select_dtypes(include="number")
    return numeric.corr() if numeric.shape[1] >= 2 else pd.DataFrame()
