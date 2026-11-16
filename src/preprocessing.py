"""Dataset validation, column discovery, and preprocessing diagnostics."""

import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd


ALIASES = {
    "customer_id": {"customerid", "customer_id", "clientid", "id"},
    "gender": {"gender", "genre", "sex"},
    "age": {"age", "customerage"},
    "income": {
        "annualincome",
        "annualincomek",
        "annualincomeusd",
        "income",
        "yearlyincome",
    },
    "spending": {
        "spendingscore",
        "spendingscore1100",
        "spendingscore1to100",
        "score",
        "purchase_score",
    },
}


def _normalise_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def detect_columns(data: pd.DataFrame) -> Dict[str, str]:
    """Map familiar customer fields to their actual uploaded column names."""
    names = {_normalise_name(column): str(column) for column in data.columns}
    detected = {}
    for role, aliases in ALIASES.items():
        match = next((names[alias] for alias in aliases if alias in names), None)
        if match is not None:
            detected[role] = match
    return detected


def prepare_dataset(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict, Dict[str, str]]:
    """Clean column names, remove unusable rows/columns, and report data quality."""
    if data.empty:
        raise ValueError("The dataset has no rows. Upload a CSV with customer records.")

    cleaned = data.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    if any(not column for column in cleaned.columns):
        raise ValueError("Every CSV column must have a name.")
    if cleaned.columns.duplicated().any():
        duplicates = cleaned.columns[cleaned.columns.duplicated()].tolist()
        raise ValueError(f"Column names must be unique. Duplicated: {', '.join(duplicates)}.")

    original_rows = len(cleaned)
    original_columns = len(cleaned.columns)
    missing_before = int(cleaned.isna().sum().sum())
    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    empty_columns = cleaned.columns[cleaned.isna().all()].tolist()
    if empty_columns:
        cleaned = cleaned.drop(columns=empty_columns)
    if cleaned.empty or len(cleaned.columns) < 2:
        raise ValueError(
            "At least two usable columns and one customer row are required. "
            "Check that the CSV contains customer data."
        )

    detected = detect_columns(cleaned)
    for role, column in list(detected.items()):
        if role in {"age", "income", "spending"}:
            converted = pd.to_numeric(cleaned[column], errors="coerce")
            if converted.notna().sum() > 0:
                cleaned[column] = converted

    # Convert number-like text columns so uploaded CSVs can use numeric fields.
    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            converted = pd.to_numeric(cleaned[column], errors="coerce")
            non_missing = cleaned[column].notna().sum()
            if non_missing and converted.notna().sum() / non_missing >= 0.9:
                cleaned[column] = converted

    numeric_columns = cleaned.select_dtypes(include=np.number).columns
    infinite_values = int(np.isinf(cleaned[numeric_columns]).sum().sum())
    if infinite_values:
        cleaned[numeric_columns] = cleaned[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )
    newly_empty = cleaned.columns[cleaned.isna().all()].tolist()
    if newly_empty:
        cleaned = cleaned.drop(columns=newly_empty)

    report = {
        "rows_before": original_rows,
        "rows_after": len(cleaned),
        "columns_before": original_columns,
        "columns_after": len(cleaned.columns),
        "duplicate_rows_removed": duplicate_rows,
        "missing_values_before": missing_before,
        "missing_values_remaining": int(cleaned.isna().sum().sum()),
        "non_finite_values_handled": infinite_values,
        "empty_columns_removed": empty_columns + newly_empty,
    }
    return cleaned, report, detect_columns(cleaned)


def get_feature_columns(data: pd.DataFrame, detected: Dict[str, str]) -> list:
    """Return usable non-ID features, excluding columns with no observed values."""
    id_column = detected.get("customer_id")
    return [
        column
        for column in data.columns
        if column != id_column and data[column].notna().any()
    ]


def outlier_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Count numeric outliers using the standard 1.5-IQR rule."""
    records = []
    for column in data.select_dtypes(include=np.number).columns:
        values = data[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        records.append(
            {
                "Feature": column,
                "Outliers": int(((values < lower) | (values > upper)).sum()),
                "Lower bound": float(lower),
                "Upper bound": float(upper),
            }
        )
    return pd.DataFrame(records)
