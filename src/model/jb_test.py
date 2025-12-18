from typing import Union, Any
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera


def perform_jarque_bera_test(
    x: Union[pd.Series, np.ndarray, list]
) -> dict[str, Any]:
    """
    Jarque-Bera test for normality.

    The Jarque-Bera test checks whether the sample data has skewness and kurtosis
    matching a normal distribution.

    Null Hypothesis (H0): The data is normally distributed.
    Alternative Hypothesis (Ha): The data is not normally distributed.

    Parameters
    ----------
    x : array_like, 1d
        The data series to test (typically model residuals).

    Returns
    -------
    dict
        Dictionary containing:
        - 'statistic': The Jarque-Bera test statistic
        - 'p_value': The p-value for the hypothesis test
        - 'is_normal': Boolean indicating if data appears normal (p > 0.05)
        - 'interpretation': String describing the result
    """
    # Clean the data
    if isinstance(x, pd.Series):
        clean_x = x.dropna().values
    elif isinstance(x, list):
        clean_x = np.array([v for v in x if v is not None and not np.isnan(v)])
    else:
        clean_x = x[~np.isnan(x)]
    
    if len(clean_x) < 4:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'is_normal': None,
            'interpretation': 'Insufficient data for Jarque-Bera test (need at least 4 observations)'
        }
    
    statistic, p_value = jarque_bera(clean_x)
    is_normal = p_value > 0.05
    
    if is_normal:
        interpretation = f"Residuals appear normally distributed (p={p_value:.4f} > 0.05). OLS inference is valid."
    else:
        interpretation = f"Residuals are NOT normal (p={p_value:.4f} <= 0.05). Likely fat tails. t-statistics may be unreliable."
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'is_normal': is_normal,
        'interpretation': interpretation
    }


if __name__ == "__main__":
    import statsmodels.api as sm
    
    # Example: Test residuals from a simple regression
    np.random.seed(42)
    
    # Generate sample data
    n = 100
    x = np.random.randn(n)
    y = 2 + 3 * x + np.random.randn(n)  # Normal errors
    
    # Fit regression
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    print("=== Jarque-Bera Test Example ===")
    print("\nCase 1: Normal residuals")
    result = perform_jarque_bera_test(model.resid)
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Generate non-normal data (heavy tails)
    print("\nCase 2: Non-normal residuals (t-distribution with df=3)")
    heavy_tail_resid = np.random.standard_t(df=3, size=100)
    result2 = perform_jarque_bera_test(heavy_tail_resid)
    for key, value in result2.items():
        print(f"  {key}: {value}")
