# Customer Intelligence — Segmentation & Analytics

A practical, interactive customer analytics application built with Streamlit and scikit-learn. It explores demographic and purchasing data, trains a real K-Means segmentation model, and turns the resulting cluster profiles into business recommendations. It supports the included Mall Customers dataset and user-provided CSV files.

## Problem statement

Customer data is often available as a flat file but difficult to translate into useful actions. This project uses unsupervised learning to discover groups with similar observed characteristics, compare their spending and income patterns, and provide a repeatable starting point for targeted marketing.

## Objectives

- Inspect data quality, missing values, duplicates, numeric summaries, and outliers.
- Explore customer demographics and behavior with interactive Plotly charts.
- Train and evaluate K-Means using scaled numeric and encoded categorical features.
- Compare cluster counts using inertia (Elbow method) and silhouette score.
- Create data-derived segment profiles, customer-level assignments, and recommendations.
- Export customer assignments, profile summaries, evaluation metrics, and an interactive chart.
- Predict a segment for an individual customer using the fitted model pipeline.

## Technology

Python 3.10+, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Joblib, Streamlit 1.48+, and Plotly.

## Dataset

`data/Mall_Customers.csv` is the 200-row Mall Customer Segmentation sample, mirrored from [ywchiu/python_for_data_science](https://github.com/ywchiu/python_for_data_science/tree/master/Data). The included fields are customer ID, gender, age, annual income (thousands), and spending score (1–100). The dataset is suitable for demonstrating unsupervised learning; it is small and should not be treated as representative of a real market.

The app also accepts CSV uploads. It recognizes common variants of customer ID, gender/genre, age, annual income, and spending score names. At least two usable columns and three rows are needed for model comparison and training. Numeric values are median-imputed and standardized; categorical values are mode-imputed and one-hot encoded in the model pipeline. Duplicate records are removed, empty columns are dropped, and the original missing values are reported.

## Methodology

1. **Validate and inspect:** normalize column whitespace, drop duplicate rows and fully empty columns, infer familiar field names, and report missingness, summaries, and IQR outlier counts.
2. **Select features:** choose two or more available non-ID features in the sidebar. Categorical and numerical features can be mixed.
3. **Preprocess:** impute missing values, standardize numeric fields, and one-hot encode categoricals. The same fitted transformations are used for customer predictions.
4. **Cluster:** fit K-Means with a reproducible random seed. Compare inertia and silhouette scores for feasible values of *k* and interactively adjust the selected cluster count.
5. **Interpret:** compare centroid coordinates in transformed space and compute customer counts, shares, demographic averages, behavior descriptions, and marketing actions from the current dataset.
6. **Export:** download assignments, segment summaries, evaluation results, and an interactive HTML chart.

Segment labels are assigned from each cluster's actual average income and spending relative to dataset averages; cluster numbers themselves are not manually assigned. When those fields are unavailable, clusters receive neutral numbered labels.

## Installation

On Windows PowerShell, from the project directory:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.10 or later is supported. If your preferred Python version is installed, replace `-3.14` with its version selector.

## Run the application

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open the local URL printed by Streamlit. Use the sidebar to select a data source, model features, cluster count, and dashboard page. The fitted model is saved to `models/customer_segmentation.joblib` when the directory is writable; the file is ignored by Git.

## Dashboard

- **Overview:** live customer and segment metrics and cluster distribution.
- **Customer analytics:** numeric distributions, gender mix, income/spending and age/spending comparisons, spending patterns, and correlation heatmap where fields exist.
- **Segmentation:** cluster visualization, centroids, segment profiles, Elbow curve, silhouette comparison, and fit metrics.
- **Customer explorer:** search/filter customer records, inspect individual assignments, and predict a segment for a new customer.
- **Business insights:** computed segment takeaways, marketing recommendations, and a retention opportunity.
- **Export center:** CSV downloads of assignments, profiles, and evaluation results, plus an interactive chart export.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover common schema detection, duplicate handling, model fitting, segment totals, and predictions through the fitted preprocessing pipeline.

## Results and business interpretation

The silhouette score and cluster assignments are computed at runtime from the selected data and features; they can change with uploads or settings. Review those metrics with the Elbow curve and cluster profiles rather than treating a single score as a definitive answer. The bundled dataset commonly demonstrates groups with different income/spending combinations, but actual profiles are always recalculated by the application.

## Screenshots

Add current screenshots of the running dashboard here before publishing, for example:

```text
screenshots/
  overview.png
  segmentation.png
```

## Future enhancements

- Add data drift and cluster stability analysis across repeated samples.
- Support business-specific segment naming and campaign outcome tracking.
- Add exportable PDF reports and scheduled data refreshes.
- Expand privacy controls and deployment-specific authentication.

## Limitations

- K-Means favors roughly spherical clusters and requires users to choose a cluster count.
- Silhouette is a geometric metric and does not establish commercial value or causality.
- Automatic segment names and marketing suggestions are heuristic summaries of averages, not a substitute for domain review or controlled experiments.
- CSV columns that are completely empty are removed; unsupported file formats are not accepted.
