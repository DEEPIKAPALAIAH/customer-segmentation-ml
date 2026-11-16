"""Streamlit customer segmentation and analytics application."""

from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.clustering import save_model, train_kmeans
from src.eda import categorical_summary, correlation_matrix, numeric_summary
from src.insights import business_insights, profile_segments
from src.prediction import predict_customer
from src.preprocessing import get_feature_columns, outlier_summary, prepare_dataset


ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "data" / "Mall_Customers.csv"
MODEL_PATH = ROOT / "models" / "customer_segmentation.joblib"

st.set_page_config(
    page_title="Customer Intelligence | Segmentation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_styles():
    style_path = ROOT / "assets" / "style.css"
    if style_path.exists():
        st.markdown(f"<style>{style_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _money(value):
    return f"{value:,.1f}" if pd.notna(value) else "—"


def _metric_card(label, value, note):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{escape(str(label))}</div>'
        f'<div class="metric-value">{escape(str(value))}</div>'
        f'<div class="metric-note">{escape(str(note))}</div></div>',
        unsafe_allow_html=True,
    )


def _chart_interpretation(text):
    st.markdown(
        f'<p class="chart-note">{escape(str(text))}</p>', unsafe_allow_html=True
    )


def _display_source():
    st.sidebar.markdown("### Data source")
    source = st.sidebar.radio(
        "Choose customer data",
        ["Mall Customers sample", "Upload a CSV"],
        label_visibility="collapsed",
    )
    if source == "Mall Customers sample":
        if not DEFAULT_DATA.exists():
            st.error("The bundled sample file is missing. Add data/Mall_Customers.csv or upload a CSV.")
            st.stop()
        return pd.read_csv(DEFAULT_DATA), "Mall Customers sample"

    uploaded = st.sidebar.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is None:
        st.info(
            "Upload a CSV containing at least two usable customer features. "
            "Commonly detected fields include Age, Gender, Annual Income, and Spending Score."
        )
        st.stop()
    try:
        return pd.read_csv(uploaded), uploaded.name
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error(f"Could not read this CSV: {error}")
        st.info("Save the file as a comma-separated UTF-8 CSV and try again.")
        st.stop()


@st.cache_data(show_spinner="Training K-Means and evaluating cluster counts…")
def _fit_cached(data, features, n_clusters):
    return train_kmeans(data, features, n_clusters)


def _create_plots(data, detected, labels, profiles):
    work = data.copy()
    work["cluster"] = labels
    segment_names = profiles.set_index("cluster")["segment"].to_dict()
    work["segment"] = work["cluster"].map(segment_names)
    age = detected.get("age")
    gender = detected.get("gender")
    income = detected.get("income")
    spending = detected.get("spending")
    return work, age, gender, income, spending


def _overview(data, profiles, income, spending):
    st.markdown("## Overview")
    st.caption("A live view of customer composition and the current segmentation run.")
    first, second, third, fourth = st.columns(4)
    with first:
        _metric_card("TOTAL CUSTOMERS", f"{len(data):,}", "Unique customer records")
    with second:
        _metric_card("SEGMENTS FOUND", str(len(profiles)), "Learned by K-Means")
    with third:
        value = _money(data[income].mean()) if income else "—"
        _metric_card("AVG. ANNUAL INCOME", value, income or "No income field detected")
    with fourth:
        value = _money(data[spending].mean()) if spending else "—"
        _metric_card("AVG. SPENDING SCORE", value, spending or "No spending field detected")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### Segment distribution")
        fig = px.bar(
            profiles,
            x="segment",
            y="customers",
            color="segment",
            text="customers",
            labels={"segment": "Customer segment", "customers": "Customers"},
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Customers")
        st.plotly_chart(fig, width="stretch")
        _chart_interpretation(
            f"{profiles.loc[profiles['customers'].idxmax(), 'segment']} is the largest "
            "observed group. Counts are calculated from the current dataset."
        )
    with right:
        st.markdown("### Customer mix")
        fig = px.pie(
            profiles,
            names="segment",
            values="customers",
            hole=0.66,
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig.update_traces(textposition="inside", textinfo="percent")
        fig.update_layout(legend_title_text="Segment")
        st.plotly_chart(fig, width="stretch")
        _chart_interpretation(
            "The chart shows each segment's share of the customers included in this run."
        )


def _analytics(data, work, age, gender, income, spending):
    st.markdown("## Customer analytics")
    st.caption("Explore demographics, spending patterns, and relationships in the source data.")
    numeric_columns = list(data.select_dtypes(include="number").columns)
    selected_columns = st.multiselect(
        "Features to include in the numeric comparison",
        numeric_columns,
        default=numeric_columns,
        key="analytics_features",
    )
    if not selected_columns:
        st.info("Select one or more numeric fields to view their distributions.")
    else:
        cols = st.columns(min(2, len(selected_columns)))
        for index, column in enumerate(selected_columns):
            fig = px.histogram(
                data,
                x=column,
                nbins=24,
                marginal="box",
                color_discrete_sequence=["#7b8cff"],
                title=f"{column} distribution",
            )
            fig.update_layout(bargap=0.08)
            with cols[index % len(cols)]:
                st.plotly_chart(fig, width="stretch")
                _chart_interpretation(
                    f"Observed values range from {_money(data[column].min())} to "
                    f"{_money(data[column].max())}; the mean is {_money(data[column].mean())}."
                )

    if gender and gender in data:
        st.markdown("### Gender distribution")
        gender_counts = data[gender].fillna("Unknown").astype(str).value_counts().rename_axis(gender).reset_index(name="Customers")
        fig = px.bar(gender_counts, x=gender, y="Customers", color=gender, text="Customers")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
        _chart_interpretation(
            f"The most common observed category is {gender_counts.iloc[0][gender]}."
        )

    if income and spending:
        st.markdown("### Income vs. spending")
        fig = px.scatter(
            work,
            x=income,
            y=spending,
            color="segment",
            hover_data=[column for column in (age, gender) if column],
            opacity=0.82,
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        st.plotly_chart(fig, width="stretch")
        corr = data[[income, spending]].corr().iloc[0, 1]
        _chart_interpretation(
            f"The observed Pearson correlation is {corr:.2f}. This describes association, "
            "not a causal relationship."
        )

    if age and spending:
        st.markdown("### Age vs. spending")
        fig = px.scatter(
            work,
            x=age,
            y=spending,
            color="segment",
            hover_data=[column for column in (income, gender) if column],
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        st.plotly_chart(fig, width="stretch")
        _chart_interpretation(
            f"The plot compares {age} and {spending} for each customer; colors show "
            "the model-assigned segment."
        )

    if spending:
        st.markdown("### Purchasing patterns")
        fig = px.box(
            work,
            x="segment",
            y=spending,
            color="segment",
            points="all",
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        _chart_interpretation(
            "Spending-score distributions reveal variation within segments as well as "
            "differences between segment averages."
        )
    else:
        st.info("A spending-score field was not detected. Purchasing-pattern charts are unavailable.")

    corr = correlation_matrix(data)
    if not corr.empty:
        st.markdown("### Numeric feature correlations")
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        st.plotly_chart(fig, width="stretch")
        _chart_interpretation(
            "Correlations range from -1 to 1; values near either extreme indicate stronger "
            "linear association, not causation."
        )


def _segmentation(data, result, profiles, work, features):
    st.markdown("## Segmentation")
    st.caption("Inspect cluster quality, customer profiles, and learned cluster centers.")
    metric_a, metric_b, metric_c = st.columns(3)
    with metric_a:
        _metric_card("SILHOUETTE SCORE", f"{result['silhouette']:.3f}", "Higher is better; range -1 to 1")
    with metric_b:
        _metric_card("INERTIA", f"{result['inertia']:,.1f}", "Within-cluster squared distance")
    with metric_c:
        _metric_card("FEATURES USED", str(len(features)), ", ".join(features))

    x_options = [
        column
        for column in features
        if pd.api.types.is_numeric_dtype(data[column])
    ] or list(features)
    y_options = [column for column in x_options if column != x_options[0]]
    if not y_options:
        y_options = list(features)
    x_col, y_col = st.columns(2)
    with x_col:
        x_feature = st.selectbox("Horizontal axis", x_options, key="cluster_x")
    with y_col:
        y_feature = st.selectbox("Vertical axis", y_options, key="cluster_y")
    if x_feature == y_feature:
        st.warning("Choose different fields for the horizontal and vertical axes.")
    else:
        fig = px.scatter(
            work,
            x=x_feature,
            y=y_feature,
            color="segment",
            hover_data=[column for column in features if column not in (x_feature, y_feature)],
            color_discrete_sequence=px.colors.qualitative.Prism,
            title="Customer clusters",
        )
        centers = result["pipeline"].named_steps["kmeans"].cluster_centers_
        numeric = [
            feature
            for feature in features
            if pd.api.types.is_numeric_dtype(data[feature])
        ]
        if x_feature in numeric and y_feature in numeric:
            preprocessor = result["pipeline"].named_steps["preprocessor"]
            names = preprocessor.get_feature_names_out()
            center_frame = pd.DataFrame(centers, columns=names)
            x_name, y_name = f"numeric__{x_feature}", f"numeric__{y_feature}"
            if x_name in center_frame and y_name in center_frame:
                scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
                numeric_names = list(
                    preprocessor.transformers_[0][2]
                )
                fig.add_trace(
                    go.Scatter(
                        x=scaler.inverse_transform(centers[:, [numeric_names.index(x) for x in numeric]])[:, numeric.index(x_feature)],
                        y=scaler.inverse_transform(centers[:, [numeric_names.index(x) for x in numeric]])[:, numeric.index(y_feature)],
                        mode="markers",
                        marker={"symbol": "x", "size": 15, "color": "#ffffff", "line": {"width": 2}},
                        name="Cluster centroids",
                    )
                )
        st.plotly_chart(fig, width="stretch")
        _chart_interpretation(
            "Points are customers colored by their fitted cluster. Cross markers show "
            "centroids when both axes are numeric model features."
        )

    st.markdown("### Cluster-count evaluation")
    evaluation = result["evaluation"]
    left, right = st.columns(2)
    with left:
        fig = px.line(evaluation, x="clusters", y="inertia", markers=True, title="Elbow method")
        fig.update_layout(xaxis_title="Number of clusters (k)", yaxis_title="Inertia")
        st.plotly_chart(fig, width="stretch")
        _chart_interpretation(
            "Look for a bend where additional clusters yield diminishing reductions in inertia; "
            "the elbow is a diagnostic rather than a guaranteed optimum."
        )
    with right:
        silhouette_evaluation = evaluation[evaluation["clusters"] >= 2]
        fig = px.line(
            silhouette_evaluation,
            x="clusters",
            y="silhouette",
            markers=True,
            title="Silhouette by cluster count",
        )
        fig.update_layout(xaxis_title="Number of clusters (k)", yaxis_title="Silhouette")
        st.plotly_chart(fig, width="stretch")
        _chart_interpretation(
            "Silhouette compares within-cluster cohesion to between-cluster separation; "
            "compare scores alongside interpretability and business context."
        )

    st.markdown("### Segment profiles")
    display = profiles.rename(
        columns={
            "segment": "Segment",
            "customers": "Customers",
            "percentage": "Share (%)",
            "average_age": "Avg. age",
            "average_income": "Avg. income",
            "average_spending": "Avg. spending",
            "characteristics": "Behavioral characteristics",
            "recommendation": "Marketing recommendation",
        }
    ).drop(columns=["cluster"])
    st.dataframe(display, width="stretch", hide_index=True)

    st.markdown("### Standardized cluster centroids")
    st.caption("Centroids are shown in the transformed feature space used by K-Means.")
    centers = result["pipeline"].named_steps["kmeans"].cluster_centers_
    feature_names = result["pipeline"].named_steps["preprocessor"].get_feature_names_out()
    center_frame = pd.DataFrame(centers, columns=feature_names)
    center_frame.insert(0, "Cluster", range(len(center_frame)))
    st.dataframe(center_frame.round(3), width="stretch", hide_index=True)


def _explorer(data, work, profiles, detected, features, result):
    st.markdown("## Customer explorer")
    st.caption("Search and inspect real customer rows and their model-assigned segments.")
    all_segments = ["All segments"] + profiles["segment"].tolist()
    selected_segment = st.selectbox("Filter by segment", all_segments)
    query = st.text_input("Search customer records", placeholder="Search any value in a row…")
    view = work.copy()
    if selected_segment != "All segments":
        view = view[view["segment"] == selected_segment]
    if query:
        matches = view.astype(str).apply(
            lambda column: column.str.contains(query, case=False, na=False)
        ).any(axis=1)
        view = view[matches]
    id_column = detected.get("customer_id")
    st.caption(f"{len(view):,} matching customers")
    st.dataframe(view, width="stretch", hide_index=True)

    if not view.empty:
        selection = st.selectbox(
            "Select a customer for details",
            view.index.tolist(),
            format_func=lambda index: (
                str(data.at[index, id_column]) if id_column else f"Customer row {index + 1}"
            ),
        )
        row = data.loc[selection]
        details = st.columns(min(4, max(1, len(data.columns))))
        for index, column in enumerate(data.columns):
            with details[index % len(details)]:
                _metric_card(column, str(row[column]) if pd.notna(row[column]) else "—", "Customer detail")
        cluster = int(work.at[selection, "cluster"])
        st.success(f"Assigned segment: {work.at[selection, 'segment']} (cluster {cluster})")

    st.markdown("### Predict a new customer")
    st.caption("Enter the same feature types used to train this model.")
    values = {}
    input_columns = st.columns(min(3, len(features)))
    for index, feature in enumerate(features):
        series = data[feature]
        with input_columns[index % len(input_columns)]:
            if pd.api.types.is_numeric_dtype(series):
                low = float(series.min())
                high = float(series.max())
                default = float(series.median())
                step = max((high - low) / 100, 0.1)
                values[feature] = st.number_input(
                    feature,
                    min_value=min(low, default),
                    max_value=max(high, default),
                    value=default,
                    step=step,
                    key=f"prediction_{feature}",
                )
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())
                if not options:
                    st.warning(f"{feature} has no available values.")
                    continue
                values[feature] = st.selectbox(
                    feature, options, key=f"prediction_{feature}"
                )
    if len(values) == len(features) and st.button("Predict customer segment", type="primary"):
        try:
            cluster = predict_customer(result["pipeline"], features, values)
            profile = profiles.loc[profiles["cluster"] == cluster].iloc[0]
            st.success(f"Predicted segment: {profile['segment']}")
            st.caption(
                "This is an unsupervised similarity assignment, not a probability or a guarantee "
                "of future purchasing behavior."
            )
        except (ValueError, KeyError) as error:
            st.error(f"Prediction could not be completed: {error}")


def _business_insights(profiles):
    st.markdown("## Business insights")
    st.caption("Recommendations are derived from the customer groups found in this run.")
    for insight in business_insights(profiles):
        st.markdown(f"- {insight}")
    st.markdown("### Recommended actions by segment")
    for _, profile in profiles.iterrows():
        with st.expander(
            f"{profile['segment']} · {profile['customers']} customers "
            f"({profile['percentage']:.1f}%)"
        ):
            st.write(profile["characteristics"])
            st.markdown(f"**Suggested action:** {profile['recommendation']}")
    if profiles["average_spending"].notna().any():
        low_spend = profiles.loc[profiles["average_spending"].idxmin()]
        st.markdown("### Retention opportunity")
        st.info(
            f"{low_spend['segment']} has the lowest average spending score "
            f"({low_spend['average_spending']:.1f}). Consider testing a small, measurable "
            "re-engagement campaign before scaling."
        )
    else:
        st.info("Spending behavior cannot be assessed because no spending-score field was detected.")


def _exports(work, profiles, evaluation, features):
    st.markdown("## Export center")
    st.caption("Download the current run's actual results for reporting and further analysis.")
    customer_csv = work.to_csv(index=False).encode("utf-8")
    summary = profiles.to_csv(index=False).encode("utf-8")
    evaluation_csv = evaluation.to_csv(index=False).encode("utf-8")
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download segmented customers",
            customer_csv,
            file_name="segmented_customers.csv",
            mime="text/csv",
            width="stretch",
        )
    with right:
        st.download_button(
            "Download segment summaries",
            summary,
            file_name="customer_segment_summary.csv",
            mime="text/csv",
            width="stretch",
        )
    st.download_button(
        "Download model evaluation",
        evaluation_csv,
        file_name="cluster_evaluation.csv",
        mime="text/csv",
    )
    try:
        if len(features) >= 2:
            chart = px.scatter(
                work,
                x=features[0],
                y=features[1],
                color="segment",
            )
            st.download_button(
                "Download customer-cluster chart (HTML)",
                chart.to_html(include_plotlyjs="cdn").encode("utf-8"),
                file_name="customer_clusters.html",
                mime="text/html",
            )
        else:
            st.info("Chart export requires at least two numeric columns in the current dataset.")
    except (IndexError, ValueError):
        st.info("Chart export is unavailable for the current dataset.")


def main():
    _load_styles()
    st.markdown(
        '<div class="hero"><div class="eyebrow">CUSTOMER INTELLIGENCE · ML WORKSPACE</div>'
        '<h1>Know your customers.<br><span>Grow with intention.</span></h1>'
        '<p>Explore behavior, discover meaningful segments, and turn customer data into action.</p></div>',
        unsafe_allow_html=True,
    )
    raw_data, source_name = _display_source()
    try:
        data, report, detected = prepare_dataset(raw_data)
    except ValueError as error:
        st.error(str(error))
        st.info(
            "Required: at least two usable customer features and one data row. "
            "Common columns such as Age, Gender, Annual Income, and Spending Score "
            "are detected automatically."
        )
        st.stop()

    features_available = get_feature_columns(data, detected)
    if len(features_available) < 2:
        st.error(
            "This dataset has fewer than two usable clustering features. "
            "Add numeric or categorical customer columns and upload it again."
        )
        st.stop()

    st.sidebar.markdown(f"**Source:** {source_name}")
    st.sidebar.caption(f"{len(data):,} rows · {len(data.columns)} usable columns")
    with st.sidebar.expander("Data quality"):
        st.write(
            f"Removed duplicates: **{report['duplicate_rows_removed']}**  \n"
            f"Missing values retained for model imputation: **{report['missing_values_remaining']}**  \n"
            f"Non-finite values treated as missing: **{report['non_finite_values_handled']}**  \n"
            f"Empty columns removed: **{len(report['empty_columns_removed'])}**"
        )
        outliers = outlier_summary(data)
        if not outliers.empty:
            st.dataframe(outliers, width="stretch", hide_index=True)
    with st.sidebar.expander("Dataset statistics"):
        before, after = st.columns(2)
        before.metric("Rows before", report["rows_before"])
        after.metric("Rows after", report["rows_after"])
        st.caption("Numeric statistics before cleaning")
        raw_numeric_summary = numeric_summary(raw_data)
        if not raw_numeric_summary.empty:
            st.dataframe(raw_numeric_summary, width="stretch", hide_index=True)
        st.caption("Numeric statistics after cleaning")
        st.dataframe(numeric_summary(data), width="stretch", hide_index=True)
        categories = categorical_summary(data)
        if not categories.empty:
            st.dataframe(categories, width="stretch", hide_index=True)

    preferred = [
        detected[key]
        for key in ("age", "income", "spending")
        if key in detected and detected[key] in features_available
    ]
    if len(preferred) < 2:
        preferred = features_available[: min(3, len(features_available))]
    selected_features = st.sidebar.multiselect(
        "Clustering features",
        options=features_available,
        default=preferred,
        help="Numerical features are imputed and standardized; categorical features are imputed and one-hot encoded.",
    )
    if len(data) < 3:
        st.error("At least three customer rows are required to compare and train clusters.")
        st.stop()
    max_cluster_count = min(10, len(data) - 1)
    if max_cluster_count == 2:
        cluster_count = 2
        st.sidebar.caption("Using 2 clusters; three customers do not allow a larger valid choice.")
    else:
        cluster_count = st.sidebar.slider(
            "Number of clusters",
            min_value=2,
            max_value=max_cluster_count,
            value=min(5, max_cluster_count),
            help="The Elbow and Silhouette charts compare several cluster counts.",
        )
    if len(selected_features) < 2:
        st.warning("Select at least two clustering features in the sidebar.")
        st.stop()
    try:
        result = _fit_cached(data, tuple(selected_features), cluster_count)
    except (ValueError, TypeError) as error:
        st.error(f"Model training failed: {error}")
        st.info("Check that the selected columns contain usable numeric or categorical values.")
        st.stop()

    data = data.reset_index(drop=True)
    profiles = profile_segments(data, result["labels"], detected)
    work, age, gender, income, spending = _create_plots(
        data, detected, result["labels"], profiles
    )
    try:
        save_model(result, str(MODEL_PATH))
    except (OSError, ValueError) as error:
        st.warning(f"The model is active but could not be saved to disk: {error}")

    navigation = st.sidebar.radio(
        "Workspace",
        [
            "Overview",
            "Customer analytics",
            "Segmentation",
            "Customer explorer",
            "Business insights",
            "Export center",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Silhouette **{result['silhouette']:.3f}** · "
        f"{len(selected_features)} features · {cluster_count} clusters"
    )

    if navigation == "Overview":
        _overview(data, profiles, income, spending)
    elif navigation == "Customer analytics":
        _analytics(data, work, age, gender, income, spending)
    elif navigation == "Segmentation":
        _segmentation(
            data,
            result,
            profiles,
            work,
            selected_features,
        )
    elif navigation == "Customer explorer":
        _explorer(data, work, profiles, detected, selected_features, result)
    elif navigation == "Business insights":
        _business_insights(profiles)
    else:
        _exports(work, profiles, result["evaluation"], selected_features)


if __name__ == "__main__":
    main()
