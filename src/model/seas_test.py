"""
Seasonality Testing Module

Tests for monthly seasonal patterns in time series data using dummy variables.
Avoids the Dummy Variable Trap by dropping one month (baseline) when a constant is present.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Optional, Tuple


def create_seasonal_dummies(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    baseline_month: int = 1,
    prefix: str = "M",
) -> Tuple[pd.DataFrame, list]:
    """
    Create monthly seasonal dummy variables.

    To avoid the Dummy Variable Trap when using an intercept (constant),
    we create N-1 dummies (11 for monthly data), dropping the baseline month.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex or a date column
    date_col : str, optional
        Name of date column. If None, uses the index
    baseline_month : int
        Month to drop as baseline (1=January, 12=December)
        Default is January (1), so coefficients represent deviation from January
    prefix : str
        Prefix for dummy column names (e.g., 'M' gives M_2, M_3, etc.)

    Returns
    -------
    Tuple[pd.DataFrame, list]
        - DataFrame with seasonal dummy columns added
        - List of seasonal column names
    """
    work_df = df.copy()

    # Extract month from index or column
    if date_col:
        work_df["_month"] = pd.to_datetime(work_df[date_col]).dt.month
    else:
        work_df["_month"] = work_df.index.month

    # Create dummies for all months
    all_dummies = pd.get_dummies(work_df["_month"], prefix=prefix)

    # Drop the baseline month to avoid Dummy Variable Trap
    baseline_col = f"{prefix}_{baseline_month}"
    if baseline_col in all_dummies.columns:
        all_dummies = all_dummies.drop(columns=[baseline_col])

    # Convert to int (0/1)
    all_dummies = all_dummies.astype(int)

    # Get column names
    seasonal_cols = all_dummies.columns.tolist()

    # Join back
    work_df = pd.concat([work_df, all_dummies], axis=1)

    # Drop temp column
    work_df = work_df.drop(columns=["_month"])

    return work_df, seasonal_cols


def test_joint_seasonality(model, seasonal_cols: list, verbose: bool = True) -> dict:
    """
    Test if all seasonal dummies are jointly significant (F-test).

    H0: All seasonal coefficients = 0 (No seasonality)
    H1: At least one seasonal coefficient ≠ 0 (Seasonality exists)

    Parameters
    ----------
    model : statsmodels OLS results
        Fitted OLS model containing seasonal dummies
    seasonal_cols : list
        List of seasonal dummy column names
    verbose : bool
        Whether to print results

    Returns
    -------
    dict
        F-test results including F-statistic, p-value, and interpretation
    """
    # Filter to columns that exist in the model
    model_vars = model.model.exog_names if hasattr(model.model, "exog_names") else []
    existing_seasonal = [col for col in seasonal_cols if col in model_vars]

    if not existing_seasonal:
        return {
            "error": "No seasonal dummies found in model",
            "f_stat": None,
            "p_value": None,
            "has_seasonality": None,
        }

    # Build hypothesis string: "M_2=0, M_3=0, ..., M_12=0"
    hypothesis = ", ".join([f"{col}=0" for col in existing_seasonal])

    try:
        f_test = model.f_test(hypothesis)
        f_stat = float(f_test.fvalue)
        p_value = float(f_test.pvalue)
        has_seasonality = p_value < 0.05

        if verbose:
            print("\n" + "=" * 60)
            print("JOINT SEASONALITY TEST (F-TEST)")
            print("=" * 60)
            print(f"H0: All monthly dummies = 0 (No seasonality)")
            print(f"H1: At least one month ≠ 0 (Seasonality exists)")
            print()
            print(f"F-statistic: {f_stat:.4f}")
            print(f"p-value: {p_value:.4f}")
            print()

            if has_seasonality:
                print("✓ REJECT H0: Significant monthly seasonality detected.")
                print(
                    "  At least one month has a statistically different return pattern."
                )
            else:
                print("✗ FAIL TO REJECT H0: No significant seasonality.")
                print("  Monthly patterns are not statistically distinguishable.")

        return {
            "f_stat": f_stat,
            "p_value": p_value,
            "has_seasonality": has_seasonality,
            "tested_months": existing_seasonal,
        }

    except Exception as e:
        if verbose:
            print(f"Error in joint seasonality test: {e}")
        return {
            "error": str(e),
            "f_stat": None,
            "p_value": None,
            "has_seasonality": None,
        }


def analyze_individual_months(
    model,
    seasonal_cols: list,
    baseline_month: int = 1,
    use_robust: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Analyze individual monthly effects and identify significant months.

    Parameters
    ----------
    model : statsmodels OLS results
        Fitted model with seasonal dummies
    seasonal_cols : list
        List of seasonal dummy column names
    baseline_month : int
        The baseline month that was dropped (for interpretation)
    use_robust : bool
        Whether to use HC3 robust standard errors
    verbose : bool
        Whether to print results

    Returns
    -------
    pd.DataFrame
        Table of monthly effects with significance indicators
    """
    month_names = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }

    if use_robust:
        results = model.get_robustcov_results(cov_type="HC3")
    else:
        results = model

    # Get model variable names
    model_vars = model.model.exog_names if hasattr(model.model, "exog_names") else []

    # Build results table
    rows = []

    for col in seasonal_cols:
        if col in model_vars:
            idx = model_vars.index(col)
            # FIX: Use .iloc[] for integer-based positioning to avoid FutureWarning
            coef = results.params.iloc[idx]
            pval = results.pvalues.iloc[idx]
            tstat = results.tvalues.iloc[idx]

            # Extract month number from column name (e.g., 'M_12' -> 12)
            month_num = int(col.split("_")[-1])

            sig = (
                "***"
                if pval < 0.01
                else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            )

            rows.append(
                {
                    "Month": month_names.get(month_num, f"Month {month_num}"),
                    "Month_Num": month_num,
                    "Coefficient": coef,
                    "t-stat": tstat,
                    "p-value": pval,
                    "Significant": sig,
                }
            )

    df_results = pd.DataFrame(rows).sort_values("Month_Num")

    if verbose:
        print("\n" + "=" * 60)
        print("INDIVIDUAL MONTHLY EFFECTS")
        print("=" * 60)
        print(
            f"Baseline Month: {month_names.get(baseline_month, f'Month {baseline_month}')}"
        )
        print("Coefficients represent deviation from baseline.\n")

        # Format for display
        display_df = df_results[
            ["Month", "Coefficient", "t-stat", "p-value", "Significant"]
        ].copy()
        print(display_df.to_string(index=False))

        # Highlight significant months
        sig_months = df_results[df_results["p-value"] < 0.05]
        if not sig_months.empty:
            print("\n✓ SIGNIFICANT MONTHLY EFFECTS (p < 0.05):")
            for _, row in sig_months.iterrows():
                direction = "higher" if row["Coefficient"] > 0 else "lower"
                print(
                    f"  {row['Month']}: {row['Coefficient']:.4f} ({direction} than {month_names.get(baseline_month)})"
                )
        else:
            print("\n✗ No individually significant months at p < 0.05")

    return df_results


def check_famous_effects(monthly_results: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Check for famous calendar anomalies in finance.

    Known effects:
    - January Effect: Higher returns in January
    - Santa Claus Rally: Higher returns in December
    - Sell in May: Lower returns May-October
    - September Effect: Lower returns in September (historically worst month)

    Parameters
    ----------
    monthly_results : pd.DataFrame
        Output from analyze_individual_months()
    verbose : bool
        Whether to print analysis

    Returns
    -------
    dict
        Dictionary of detected effects
    """
    effects = {}

    if verbose:
        print("\n" + "=" * 60)
        print("FAMOUS CALENDAR ANOMALIES CHECK")
        print("=" * 60)

    # December Effect (Santa Claus Rally)
    dec_row = monthly_results[monthly_results["Month_Num"] == 12]
    if not dec_row.empty:
        dec_coef = dec_row["Coefficient"].values[0]
        dec_pval = dec_row["p-value"].values[0]
        dec_sig = dec_pval < 0.05
        effects["december_effect"] = {
            "coefficient": dec_coef,
            "p_value": dec_pval,
            "significant": dec_sig,
            "direction": "positive" if dec_coef > 0 else "negative",
        }
        if verbose:
            status = "✓ CONFIRMED" if dec_sig and dec_coef > 0 else "✗ Not significant"
            print(f"\nSanta Claus Rally (December): {status}")
            print(f"  Coefficient: {dec_coef:.4f}, p-value: {dec_pval:.4f}")

    # September Effect (historically worst month)
    sep_row = monthly_results[monthly_results["Month_Num"] == 9]
    if not sep_row.empty:
        sep_coef = sep_row["Coefficient"].values[0]
        sep_pval = sep_row["p-value"].values[0]
        sep_sig = sep_pval < 0.05
        effects["september_effect"] = {
            "coefficient": sep_coef,
            "p_value": sep_pval,
            "significant": sep_sig,
            "direction": "positive" if sep_coef > 0 else "negative",
        }
        if verbose:
            status = "✓ CONFIRMED" if sep_sig and sep_coef < 0 else "✗ Not significant"
            print(f"\nSeptember Effect (Worst Month): {status}")
            print(f"  Coefficient: {sep_coef:.4f}, p-value: {sep_pval:.4f}")

    # Sell in May (May-October underperformance)
    summer_months = [5, 6, 7, 8, 9, 10]
    summer_rows = monthly_results[monthly_results["Month_Num"].isin(summer_months)]
    if not summer_rows.empty:
        avg_summer_coef = summer_rows["Coefficient"].mean()
        any_sig = (summer_rows["p-value"] < 0.05).any()
        effects["sell_in_may"] = {
            "avg_coefficient": avg_summer_coef,
            "any_significant": any_sig,
            "direction": "underperform" if avg_summer_coef < 0 else "outperform",
        }
        if verbose:
            status = (
                "⚠ Partially confirmed"
                if any_sig and avg_summer_coef < 0
                else "✗ Not significant"
            )
            print(f"\nSell in May (May-Oct): {status}")
            print(f"  Average May-Oct coefficient: {avg_summer_coef:.4f}")

    return effects


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)

    # Create synthetic monthly data
    dates = pd.date_range("2015-01-01", periods=120, freq="MS")

    # Simulate returns with December seasonality
    returns = np.random.randn(120) * 0.04 + 0.007  # ~0.7% monthly drift

    # Add December effect (+2% extra)
    for i, d in enumerate(dates):
        if d.month == 12:
            returns[i] += 0.02

    df = pd.DataFrame(
        {"returns": returns, "exog": np.random.randn(120)},  # Random exogenous variable
        index=dates,
    )

    print("=" * 60)
    print("SEASONALITY TEST - Example with Synthetic Data")
    print("=" * 60)
    print("True data: December has +2% extra return\n")

    # Create seasonal dummies
    df_seasonal, seasonal_cols = create_seasonal_dummies(df, baseline_month=1)

    print(
        f"Created {len(seasonal_cols)} seasonal dummies: {seasonal_cols[:3]}...{seasonal_cols[-3:]}"
    )

    # Fit model
    Y = df_seasonal["returns"]
    X = df_seasonal[seasonal_cols].copy()
    X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit()

    # Test joint seasonality
    joint_result = test_joint_seasonality(model, seasonal_cols)

    # Analyze individual months
    monthly_df = analyze_individual_months(model, seasonal_cols, baseline_month=1)

    # Check famous effects
    effects = check_famous_effects(monthly_df)
