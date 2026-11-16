"""Data-driven customer profiles and marketing recommendations."""

import numpy as np
import pandas as pd


def _mean_or_nan(series):
    return float(series.mean()) if series.notna().any() else float("nan")


def profile_segments(
    data: pd.DataFrame, labels, detected: dict
) -> pd.DataFrame:
    """Summarize each fitted cluster and derive its label from relative behavior."""
    work = data.copy()
    work["cluster"] = labels
    total = len(work)
    age = detected.get("age")
    income = detected.get("income")
    spending = detected.get("spending")

    overall_income = _mean_or_nan(work[income]) if income else float("nan")
    overall_spending = _mean_or_nan(work[spending]) if spending else float("nan")
    lower_income = (
        float(work[income].quantile(0.35))
        if income and work[income].notna().any()
        else float("nan")
    )
    lower_spending = (
        float(work[spending].quantile(0.35))
        if spending and work[spending].notna().any()
        else float("nan")
    )
    has_business_metrics = income is not None and spending is not None
    profiles = []

    for cluster, group in work.groupby("cluster", sort=True):
        avg_age = _mean_or_nan(group[age]) if age else float("nan")
        avg_income = _mean_or_nan(group[income]) if income else float("nan")
        avg_spending = _mean_or_nan(group[spending]) if spending else float("nan")

        if has_business_metrics and np.isfinite(avg_income) and np.isfinite(avg_spending):
            high_income = avg_income >= overall_income
            high_spending = avg_spending >= overall_spending
            if high_income and high_spending:
                segment = "Premium Customers"
                recommendation = "Reward loyalty and protect retention with exclusive benefits."
            elif high_spending:
                segment = "High-Spending Customers"
                recommendation = "Use cross-sell offers and timely replenishment campaigns."
            elif high_income:
                segment = "Under-engaged Affluent"
                recommendation = "Test premium discovery offers and personalized reactivation."
            elif avg_income < lower_income and avg_spending < lower_spending:
                segment = "Budget-Conscious Customers"
                recommendation = "Use value bundles, clear savings, and low-cost loyalty rewards."
            else:
                segment = "Potential Customers"
                recommendation = "Nurture with relevant recommendations and first-purchase incentives."
        else:
            segment = f"Customer Segment {int(cluster) + 1}"
            recommendation = (
                "Run a focused campaign informed by this segment's observed feature profile."
            )

        behaviors = []
        if np.isfinite(avg_age) and age:
            behaviors.append(
                f"Average age is {avg_age:.1f} years"
            )
        if income and np.isfinite(avg_income):
            relative = "above" if avg_income >= overall_income else "below"
            behaviors.append(f"income is {relative} the dataset average")
        if spending and np.isfinite(avg_spending):
            relative = "above" if avg_spending >= overall_spending else "below"
            behaviors.append(f"spending score is {relative} the dataset average")
        if not behaviors:
            behaviors.append("profile is based on the selected clustering features")

        profiles.append(
            {
                "cluster": int(cluster),
                "segment": segment,
                "customers": int(len(group)),
                "percentage": len(group) / total * 100,
                "average_age": avg_age,
                "average_income": avg_income,
                "average_spending": avg_spending,
                "characteristics": "; ".join(behaviors) + ".",
                "recommendation": recommendation,
            }
        )
    result = pd.DataFrame(profiles)
    duplicate_names = result["segment"].duplicated(keep=False)
    result.loc[duplicate_names, "segment"] = result.loc[duplicate_names].apply(
        lambda row: f"{row['segment']} · Cluster {int(row['cluster'])}", axis=1
    )
    return result


def business_insights(profiles: pd.DataFrame) -> list:
    """Generate concise insights from the computed segment profiles."""
    if profiles.empty:
        return ["No customer segments are available for this dataset."]
    insights = []
    largest = profiles.loc[profiles["customers"].idxmax()]
    insights.append(
        f"{largest['segment']} is the largest group, representing "
        f"{largest['percentage']:.1f}% of customers."
    )
    if profiles["average_spending"].notna().any():
        highest_spend = profiles.loc[profiles["average_spending"].idxmax()]
        insights.append(
            f"{highest_spend['segment']} has the highest average spending score "
            f"({highest_spend['average_spending']:.1f})."
        )
    if profiles["average_income"].notna().any():
        highest_income = profiles.loc[profiles["average_income"].idxmax()]
        insights.append(
            f"{highest_income['segment']} has the highest average income "
            f"({highest_income['average_income']:.1f})."
        )
    premium = profiles[profiles["segment"] == "Premium Customers"]
    if not premium.empty:
        insights.append(
            "Prioritize retention for Premium Customers with recognition and "
            "high-relevance benefits."
        )
    return insights
