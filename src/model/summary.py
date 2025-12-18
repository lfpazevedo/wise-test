"""
Executive Summary Generator

Generates plain-English reports summarizing the econometric analysis results.
Designed for non-technical stakeholders.
"""

import pandas as pd
from typing import Optional


def generate_executive_summary(
    adl_results: dict,
    granger_results: dict,
    seasonal_results: dict,
    verbose: bool = True,
) -> str:
    """
    Generate a plain-English executive summary of the analysis.

    Parameters
    ----------
    adl_results : dict
        Results from run_adl_forecast()
    granger_results : dict
        Results from run_granger_test()
    seasonal_results : dict
        Results from test_joint_seasonality()
    verbose : bool
        Whether to print the summary

    Returns
    -------
    str
        The executive summary text
    """
    lines = []

    lines.append("")
    lines.append("#" * 60)
    lines.append("EXECUTIVE SUMMARY: S&P 500 PREDICTABILITY ANALYSIS")
    lines.append("#" * 60)

    # 1. Market Structure
    lines.append("")
    lines.append("1. MARKET BEHAVIOR:")

    # Check if random walk detected
    sig_vars = adl_results.get("inference_table", pd.DataFrame())
    if not sig_vars.empty:
        sig_predictors = sig_vars[sig_vars["p-value"] < 0.05].index.tolist()
        non_const_sig = [v for v in sig_predictors if str(v).lower() != "const"]
    else:
        non_const_sig = []

    if len(non_const_sig) == 0 and adl_results.get("differenced", False):
        lines.append("   ✓ The S&P 500 follows a 'Random Walk with Drift'.")
        lines.append(
            "   → Best prediction: Today's price + average growth (~0.7%/month)."
        )
        lines.append("   → Past returns do NOT predict future returns.")
    else:
        lines.append("   ⚠ Some predictors show significance.")
        lines.append(
            f"   → Significant variables: {', '.join(map(str, non_const_sig))}"
        )

    # Fat tails warning
    jb_pvalue = adl_results.get("best_model", {}).get("jb_pvalue", 1.0)
    if jb_pvalue < 0.05:
        lines.append("")
        lines.append("   ⚠ RISK ALERT: 'Fat Tails' detected in residuals.")
        lines.append("   → Extreme market events (crashes/rallies) occur MORE often")
        lines.append("     than standard statistical models predict.")
        lines.append("   → Confidence intervals may UNDERESTIMATE actual risk.")

    # 2. Sunspot Hypothesis
    lines.append("")
    lines.append("2. SUNSPOT CORRELATION:")

    granger_causes = granger_results.get("granger_causes", False)
    min_pvalue = granger_results.get("min_pvalue", 1.0)

    # Check if X_lag is significant in ADL
    adl_supports = (
        any("X_lag" in str(v) or "x_lag" in str(v) for v in sig_predictors)
        if not sig_vars.empty
        else False
    )

    if granger_causes and adl_supports:
        lines.append("   Status: ✓ CONFIRMED")
        lines.append(f"   → Granger Test p-value: {min_pvalue:.4f} (Significant)")
        lines.append("   → Sunspots may have predictive power for S&P 500.")
    elif granger_causes or adl_supports:
        lines.append("   Status: ⚠ MIXED EVIDENCE")
        lines.append(f"   → Granger Test p-value: {min_pvalue:.4f}")
        lines.append("   → Results are inconclusive. Interpret with caution.")
    else:
        lines.append("   Status: ✗ BUSTED")
        lines.append(f"   → Granger Test p-value: {min_pvalue:.4f} (Not Significant)")
        lines.append(
            "   → Solar activity provides ZERO edge in forecasting market direction."
        )

    # 3. Seasonality
    lines.append("")
    lines.append("3. SEASONAL PATTERNS (Calendar Effects):")

    has_seasonality = seasonal_results.get("has_seasonality", False)
    seas_pvalue = seasonal_results.get("p_value", 1.0)

    if has_seasonality:
        lines.append("   Status: ✓ DETECTED")
        lines.append(f"   → Joint F-test p-value: {seas_pvalue:.4f} (Significant)")
        sig_months = seasonal_results.get("tested_months", [])
        if sig_months:
            lines.append(f"   → Significant months: {', '.join(sig_months)}")
    else:
        lines.append("   Status: ✗ NO SIGNIFICANT SEASONALITY")
        lines.append(f"   → Joint F-test p-value: {seas_pvalue:.4f}")
        lines.append("   → 'Santa Claus Rally' - Not statistically significant")
        lines.append("   → 'Sell in May' - Not statistically significant")
        lines.append("   → 'September Effect' - Not statistically significant")

    # 4. Forecast
    lines.append("")
    lines.append("4. FORECAST:")
    forecast = adl_results.get("forecast", None)
    last_actual = adl_results.get("last_actual", None)

    if forecast and last_actual:
        pct_change = ((forecast - last_actual) / last_actual) * 100
        lines.append(f"   → Last Observed Price: {last_actual:,.2f}")
        lines.append(f"   → Next Period Forecast: {forecast:,.2f}")
        lines.append(f"   → Implied Change: {pct_change:+.2f}%")
        lines.append("")
        lines.append("   ⚠ CAVEAT: Given the 'Random Walk' nature, this forecast")
        lines.append(
            "   represents the AVERAGE expectation, not a confident prediction."
        )

    # 5. Recommendation
    lines.append("")
    lines.append("5. RECOMMENDATION:")

    if not granger_causes and not has_seasonality and len(non_const_sig) == 0:
        lines.append("   ✗ DO NOT trade based on sunspots or calendar effects.")
        lines.append("   → The market appears efficient with respect to these signals.")
        lines.append("   → Any observed correlations are likely SPURIOUS.")
    else:
        lines.append("   ⚠ Some signals show statistical significance.")
        lines.append("   → Further investigation recommended before trading.")
        lines.append(
            "   → Consider economic significance, not just statistical significance."
        )

    lines.append("")
    lines.append("#" * 60)

    summary_text = "\n".join(lines)

    if verbose:
        print(summary_text)

    return summary_text


if __name__ == "__main__":
    # Example usage with mock data
    mock_adl = {
        "forecast": 6758.40,
        "last_actual": 6721.43,
        "differenced": True,
        "y_logged": True,
        "best_model": {"jb_pvalue": 0.0001},
        "inference_table": pd.DataFrame(
            {"Coefficient": [0.0068, 0.035, 0.005], "p-value": [0.04, 0.72, 0.29]},
            index=["const", "Y_lag1", "X_lag1"],
        ),
    }

    mock_granger = {"granger_causes": False, "min_pvalue": 0.1989}

    mock_seasonal = {"has_seasonality": False, "p_value": 0.5699}

    generate_executive_summary(mock_adl, mock_granger, mock_seasonal)
