"""
Daily Returns Analysis Module

Analyzes daily S&P 500 returns in relation to sunspot activity.
Provides robustness checks at daily frequency to complement monthly analysis.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import jarque_bera, spearmanr, pearsonr
from typing import Optional


def calculate_daily_returns(
    df: pd.DataFrame, price_col: str = "SP500_Close"
) -> pd.DataFrame:
    """
    Calculate daily log returns from price data.

    Parameters
    ----------
    df : DataFrame
        DataFrame with price data
    price_col : str
        Column name for price data

    Returns
    -------
    DataFrame
        Original data with added return columns
    """
    work_df = df.copy()

    # Simple returns
    work_df["return_simple"] = work_df[price_col].pct_change()

    # Log returns (preferred for statistical analysis)
    work_df["return_log"] = np.log(work_df[price_col] / work_df[price_col].shift(1))

    return work_df


def analyze_daily_sunspot_correlation(
    df: pd.DataFrame,
    return_col: str = "return_log",
    sunspot_col: str = "monthly_total_sunspot_number",
    lags: list = None,
    verbose: bool = True,
) -> dict:
    """
    Analyze correlation between daily returns and sunspot activity at various lags.

    Parameters
    ----------
    df : DataFrame
        DataFrame with returns and sunspot data
    return_col : str
        Column name for returns
    sunspot_col : str
        Column name for sunspot data
    lags : list
        List of lags to test (in days). Default [0, 1, 5, 21, 63, 126, 252]
        representing same-day, 1-day, 1-week, 1-month, 3-months, 6-months, 1-year
    verbose : bool
        Whether to print results

    Returns
    -------
    dict
        Correlation analysis results
    """
    if lags is None:
        lags = [0, 1, 5, 21, 63, 126, 252]  # Standard trading day intervals

    work_df = df[[return_col, sunspot_col]].dropna()

    if verbose:
        print("\n" + "=" * 60)
        print("DAILY RETURNS vs SUNSPOT CORRELATION ANALYSIS")
        print("=" * 60)
        print(f"Observations: {len(work_df)}")
        print()

    results = []

    for lag in lags:
        # Lagged sunspot data
        sunspot_lagged = work_df[sunspot_col].shift(lag)

        # Clean data
        valid_idx = ~(work_df[return_col].isna() | sunspot_lagged.isna())
        returns = work_df.loc[valid_idx, return_col]
        sunspots = sunspot_lagged[valid_idx]

        if len(returns) < 30:
            continue

        # Pearson correlation (linear)
        pearson_r, pearson_p = pearsonr(returns, sunspots)

        # Spearman correlation (rank-based, robust to outliers)
        spearman_r, spearman_p = spearmanr(returns, sunspots)

        results.append(
            {
                "lag_days": lag,
                "lag_desc": _lag_description(lag),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "n_obs": len(returns),
                "significant": pearson_p < 0.05 or spearman_p < 0.05,
            }
        )

    if verbose and results:
        print(
            f"{'Lag':<12} | {'Pearson r':>10} | {'p-value':>10} | {'Spearman r':>10} | {'p-value':>10}"
        )
        print("-" * 65)
        for r in results:
            sig = "✓" if r["significant"] else ""
            print(
                f"{r['lag_desc']:<12} | {r['pearson_r']:>10.4f} | {r['pearson_p']:>10.4f} | "
                f"{r['spearman_r']:>10.4f} | {r['spearman_p']:>10.4f} {sig}"
            )

        # Summary
        sig_lags = [r for r in results if r["significant"]]
        print()
        if sig_lags:
            print(f"✓ Significant correlations found at {len(sig_lags)} lag(s)")
        else:
            print("✗ No significant correlations at any lag")

    return {
        "correlations": results,
        "any_significant": any(r["significant"] for r in results),
        "max_abs_correlation": (
            max(abs(r["pearson_r"]) for r in results) if results else 0
        ),
    }


def _lag_description(lag: int) -> str:
    """Convert lag in days to human-readable description."""
    if lag == 0:
        return "Same day"
    elif lag == 1:
        return "1 day"
    elif lag <= 5:
        return f"{lag} days"
    elif lag <= 21:
        return f"~{lag//5} week(s)"
    elif lag <= 63:
        return f"~{lag//21} month(s)"
    elif lag <= 252:
        return f"~{lag//63} quarter(s)"
    else:
        return f"~{lag//252} year(s)"


def test_daily_return_predictability(
    df: pd.DataFrame,
    return_col: str = "return_log",
    sunspot_col: str = "monthly_total_sunspot_number",
    ar_lags: int = 5,
    sunspot_lags: int = 21,
    verbose: bool = True,
) -> dict:
    """
    Test if sunspots add predictive power beyond autoregressive terms.

    Uses nested model comparison:
    - Restricted: Returns ~ AR(p)
    - Unrestricted: Returns ~ AR(p) + Sunspot lags

    Parameters
    ----------
    df : DataFrame
        DataFrame with returns and sunspot data
    return_col : str
        Column name for returns
    sunspot_col : str
        Column name for sunspot data
    ar_lags : int
        Number of autoregressive lags
    sunspot_lags : int
        Number of sunspot lags to include
    verbose : bool
        Whether to print results

    Returns
    -------
    dict
        Nested model comparison results
    """
    work_df = df[[return_col, sunspot_col]].copy()

    # Create lagged features
    for i in range(1, ar_lags + 1):
        work_df[f"ret_lag{i}"] = work_df[return_col].shift(i)

    for i in range(1, sunspot_lags + 1):
        work_df[f"sun_lag{i}"] = work_df[sunspot_col].shift(i)

    work_df = work_df.dropna()

    if len(work_df) < ar_lags + sunspot_lags + 10:
        return {"error": "Insufficient data"}

    Y = work_df[return_col]

    # Restricted model (AR only)
    ar_cols = [f"ret_lag{i}" for i in range(1, ar_lags + 1)]
    X_restricted = sm.add_constant(work_df[ar_cols])
    model_restricted = sm.OLS(Y, X_restricted).fit()

    # Unrestricted model (AR + Sunspots)
    sun_cols = [f"sun_lag{i}" for i in range(1, sunspot_lags + 1)]
    X_unrestricted = sm.add_constant(work_df[ar_cols + sun_cols])
    model_unrestricted = sm.OLS(Y, X_unrestricted).fit()

    # F-test for nested models
    f_stat = ((model_restricted.ssr - model_unrestricted.ssr) / sunspot_lags) / (
        model_unrestricted.ssr / model_unrestricted.df_resid
    )

    from scipy.stats import f as f_dist

    f_pvalue = 1 - f_dist.cdf(f_stat, sunspot_lags, model_unrestricted.df_resid)

    sunspots_add_value = f_pvalue < 0.05

    if verbose:
        print("\n" + "=" * 60)
        print("NESTED MODEL COMPARISON (Daily Frequency)")
        print("=" * 60)
        print(f"H0: Sunspot lags do not improve prediction")
        print(f"H1: Sunspot lags add predictive power")
        print()
        print(f"Restricted Model: AR({ar_lags})")
        print(f"  R²: {model_restricted.rsquared:.4f}")
        print(f"  AIC: {model_restricted.aic:.2f}")
        print()
        print(f"Unrestricted Model: AR({ar_lags}) + Sunspot lags (1-{sunspot_lags})")
        print(f"  R²: {model_unrestricted.rsquared:.4f}")
        print(f"  AIC: {model_unrestricted.aic:.2f}")
        print()
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {f_pvalue:.4f}")
        print()

        if sunspots_add_value:
            print(
                "✓ REJECT H0: Sunspots add significant predictive power at daily frequency"
            )
        else:
            print(
                "✗ FAIL TO REJECT H0: Sunspots do NOT add predictive power at daily frequency"
            )

    return {
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
        "sunspots_add_value": sunspots_add_value,
        "r2_restricted": model_restricted.rsquared,
        "r2_unrestricted": model_unrestricted.rsquared,
        "aic_restricted": model_restricted.aic,
        "aic_unrestricted": model_unrestricted.aic,
        "model_restricted": model_restricted,
        "model_unrestricted": model_unrestricted,
    }


def run_daily_robustness_check(
    df: pd.DataFrame,
    price_col: str = "SP500_Close",
    sunspot_col: str = "monthly_total_sunspot_number",
    verbose: bool = True,
) -> dict:
    """
    Run complete daily frequency robustness analysis.

    Parameters
    ----------
    df : DataFrame
        Daily data with price and sunspot columns
    price_col : str
        Column name for price data
    sunspot_col : str
        Column name for sunspot data
    verbose : bool
        Whether to print results

    Returns
    -------
    dict
        Complete daily analysis results
    """
    if verbose:
        print("\n" + "=" * 60)
        print("DAILY FREQUENCY ROBUSTNESS ANALYSIS")
        print("=" * 60)

    # Calculate returns
    df_returns = calculate_daily_returns(df, price_col)

    # Basic statistics
    returns = df_returns["return_log"].dropna()

    if verbose:
        print(f"\nDaily Return Statistics:")
        print(f"  Mean: {returns.mean()*100:.4f}%")
        print(f"  Std Dev: {returns.std()*100:.4f}%")
        print(f"  Skewness: {returns.skew():.4f}")
        print(f"  Kurtosis: {returns.kurtosis():.4f}")

        # Jarque-Bera test
        jb_stat, jb_p = jarque_bera(returns)
        print(
            f"  Jarque-Bera p-value: {jb_p:.4f} ({'Normal' if jb_p > 0.05 else 'Non-Normal'})"
        )

    # Correlation analysis
    corr_results = analyze_daily_sunspot_correlation(
        df_returns, "return_log", sunspot_col, verbose=verbose
    )

    # Predictability test
    pred_results = test_daily_return_predictability(
        df_returns, "return_log", sunspot_col, verbose=verbose
    )

    # Summary verdict
    if verbose:
        print("\n" + "=" * 60)
        print("DAILY ANALYSIS VERDICT")
        print("=" * 60)

        evidence_for = []
        evidence_against = []

        if corr_results.get("any_significant"):
            evidence_for.append("Significant correlations found")
        else:
            evidence_against.append("No significant correlations")

        if pred_results.get("sunspots_add_value"):
            evidence_for.append("Sunspots improve AR model")
        else:
            evidence_against.append("Sunspots don't improve prediction")

        if evidence_for:
            print("Evidence FOR sunspot-market relationship:")
            for e in evidence_for:
                print(f"  ✓ {e}")

        if evidence_against:
            print("Evidence AGAINST sunspot-market relationship:")
            for e in evidence_against:
                print(f"  ✗ {e}")

    return {
        "return_stats": {
            "mean": returns.mean(),
            "std": returns.std(),
            "skew": returns.skew(),
            "kurtosis": returns.kurtosis(),
        },
        "correlation_analysis": corr_results,
        "predictability_test": pred_results,
        "supports_hypothesis": corr_results.get("any_significant")
        or pred_results.get("sunspots_add_value", False),
    }


if __name__ == "__main__":
    # Example usage
    from src.data.processing.df_daily import get_daily_merged_data

    print("Loading daily data...")
    daily_df = get_daily_merged_data()
    daily_df = daily_df.set_index("date")

    results = run_daily_robustness_check(daily_df)
