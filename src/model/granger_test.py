"""
Granger Causality Test Module

Tests whether one time series (X) has predictive power for another (Y).
H0: X does NOT Granger-cause Y
H1: X Granger-causes Y (past values of X help predict Y)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Optional


def run_granger_test(
    df: pd.DataFrame,
    target_col: str,
    exog_col: str,
    max_lags: int = 4,
    significance: float = 0.05,
    verbose: bool = True
) -> dict:
    """
    Performs Granger Causality Test on STATIONARY data.
    
    IMPORTANT: Input data must be stationary (differenced if needed).
    Running Granger on non-stationary data yields spurious results.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data
    target_col : str
        Column name of the target variable (Y) - what we're trying to predict
    exog_col : str
        Column name of the exogenous variable (X) - potential predictor
    max_lags : int
        Maximum number of lags to test (tests 1 to max_lags)
    significance : float
        Significance level for hypothesis testing (default 0.05)
    verbose : bool
        Whether to print detailed output
    
    Returns
    -------
    dict
        Dictionary containing test results and interpretation
    """
    if verbose:
        print("\n" + "=" * 60)
        print("GRANGER CAUSALITY TEST")
        print("=" * 60)
        print(f"Question: Does {exog_col} Granger-cause {target_col}?")
        print(f"H0: {exog_col} does NOT help predict {target_col}")
        print(f"H1: {exog_col} has predictive power for {target_col}")
        print()
    
    # 1. Prepare Data
    # Statsmodels expects column order: [Target, Predictor]
    data = df[[target_col, exog_col]].dropna()
    
    if len(data) < max_lags + 2:
        msg = f"Error: Not enough data points ({len(data)}) for Granger Test with {max_lags} lags."
        if verbose:
            print(msg)
        return {'error': msg, 'results': None}
    
    if verbose:
        print(f"Observations used: {len(data)}")
        print(f"Testing lags: 1 to {max_lags}")
        print()
    
    # 2. Run Test
    # Removed verbose=False to fix deprecation warning
    try:
        test_result = grangercausalitytests(data, maxlag=max_lags)
    except Exception as e:
        if verbose:
            print(f"Error running Granger test: {e}")
        return {'error': str(e), 'results': None}
    
    # 3. Parse Results
    results_list = []
    
    if verbose:
        print(f"{'Lag':<5} | {'F-stat':<12} | {'p-value':<12} | {'Conclusion':<20}")
        print("-" * 55)
    
    for lag in range(1, max_lags + 1):
        # Dictionary structure: result[lag][0]['ssr_ftest']
        # ssr_ftest returns (F-stat, p-value, df_denom, df_num)
        f_stat, p_value, df_denom, df_num = test_result[lag][0]['ssr_ftest']
        
        is_significant = p_value < significance
        conclusion = "SIGNIFICANT ✓" if is_significant else "Not significant"
        
        if verbose:
            print(f"{lag:<5} | {f_stat:<12.4f} | {p_value:<12.4f} | {conclusion:<20}")
        
        results_list.append({
            'lag': lag,
            'f_stat': f_stat,
            'p_value': p_value,
            'significant': is_significant
        })
    
    # 4. Overall Interpretation
    significant_lags = [r['lag'] for r in results_list if r['significant']]
    min_pvalue = min(r['p_value'] for r in results_list)
    best_lag = [r['lag'] for r in results_list if r['p_value'] == min_pvalue][0]
    
    granger_causes = len(significant_lags) > 0
    
    if verbose:
        print()
        print("=" * 60)
        print("GRANGER CAUSALITY VERDICT")
        print("=" * 60)
        
        if granger_causes:
            print(f"✓ REJECT H0 at α={significance}")
            print(f"  {exog_col} Granger-causes {target_col}")
            print(f"  Significant at lag(s): {significant_lags}")
            print(f"  Best lag: {best_lag} (p={min_pvalue:.4f})")
            print()
            print("  INTERPRETATION: There IS statistical evidence that past values")
            print(f"  of {exog_col} help predict future values of {target_col}.")
        else:
            print(f"✗ FAIL TO REJECT H0 at α={significance}")
            print(f"  {exog_col} does NOT Granger-cause {target_col}")
            print(f"  Lowest p-value: {min_pvalue:.4f} at lag {best_lag}")
            print()
            print("  INTERPRETATION: No statistical evidence that past values")
            print(f"  of {exog_col} help predict future values of {target_col}.")
            print("  This is consistent with the Efficient Market Hypothesis.")
    
    return {
        'granger_causes': granger_causes,
        'significant_lags': significant_lags,
        'best_lag': best_lag,
        'min_pvalue': min_pvalue,
        'results': results_list,
        'target': target_col,
        'predictor': exog_col
    }


def run_bidirectional_granger_test(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    max_lags: int = 4,
    significance: float = 0.05,
    verbose: bool = True
) -> dict:
    """
    Run Granger Causality in both directions to check for:
    - A -> B (Does A predict B?)
    - B -> A (Does B predict A?)
    
    This helps identify the direction of causality or detect feedback loops.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("BIDIRECTIONAL GRANGER CAUSALITY TEST")
        print("=" * 60)
    
    # Test A -> B
    if verbose:
        print(f"\n--- Testing: {col_a} → {col_b} ---")
    result_a_to_b = run_granger_test(
        df, target_col=col_b, exog_col=col_a, 
        max_lags=max_lags, significance=significance, verbose=verbose
    )
    
    # Test B -> A
    if verbose:
        print(f"\n--- Testing: {col_b} → {col_a} ---")
    result_b_to_a = run_granger_test(
        df, target_col=col_a, exog_col=col_b,
        max_lags=max_lags, significance=significance, verbose=verbose
    )
    
    # Summary
    a_causes_b = result_a_to_b.get('granger_causes', False)
    b_causes_a = result_b_to_a.get('granger_causes', False)
    
    if verbose:
        print("\n" + "=" * 60)
        print("BIDIRECTIONAL SUMMARY")
        print("=" * 60)
        
        if a_causes_b and b_causes_a:
            print(f"⟷ FEEDBACK LOOP: {col_a} ↔ {col_b}")
            print("  Both variables Granger-cause each other.")
        elif a_causes_b:
            print(f"→ UNIDIRECTIONAL: {col_a} → {col_b}")
            print(f"  {col_a} predicts {col_b}, but not vice versa.")
        elif b_causes_a:
            print(f"← UNIDIRECTIONAL: {col_a} ← {col_b}")
            print(f"  {col_b} predicts {col_a}, but not vice versa.")
        else:
            print(f"✗ NO CAUSALITY: {col_a} ⊥ {col_b}")
            print("  Neither variable Granger-causes the other.")
    
    return {
        f'{col_a}_causes_{col_b}': result_a_to_b,
        f'{col_b}_causes_{col_a}': result_b_to_a,
        'relationship': 'feedback' if (a_causes_b and b_causes_a) 
                        else 'a_causes_b' if a_causes_b 
                        else 'b_causes_a' if b_causes_a 
                        else 'none'
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    n = 200
    
    # Create synthetic stationary data
    x = np.random.randn(n)
    # y has some dependence on lagged x (true Granger causality)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.3 * y[t-1] + 0.2 * x[t-1] + 0.1 * x[t-2] + np.random.randn()
    
    df = pd.DataFrame({'Y': y, 'X': x})
    
    print("=" * 60)
    print("GRANGER CAUSALITY TEST - Example with Synthetic Data")
    print("=" * 60)
    print("True relationship: Y[t] = 0.3*Y[t-1] + 0.2*X[t-1] + 0.1*X[t-2] + noise")
    print("Expected: X should Granger-cause Y")
    
    result = run_granger_test(df, target_col='Y', exog_col='X', max_lags=4)
    
    print("\n" + "=" * 60)
    print("Test with No Causality (Independent Series)")
    print("=" * 60)
    
    df_independent = pd.DataFrame({
        'Y': np.random.randn(n),
        'X': np.random.randn(n)
    })
    
    result_none = run_granger_test(df_independent, target_col='Y', exog_col='X', max_lags=4)
