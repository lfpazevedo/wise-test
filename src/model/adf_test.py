from typing import Union, Optional, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def perform_adf_test(
    x: Union[pd.Series, np.ndarray, list],
    maxlag: Optional[int] = None,
    regression: str = 'c',
    autolag: Optional[str] = 'AIC',
    store: bool = False,
    regresults: bool = False
) -> Any:
    """
    Augmented Dickey-Fuller unit root test.

    The Augmented Dickey-Fuller test can be used to test for a unit root in a 
    univariate process in the presence of serial correlation.

    Parameters
    ----------
    x : array_like, 1d
        The data series to test.
    maxlag : int, optional
        Maximum lag which is included in test, default value of 12*(nobs/100)^{1/4} 
        is used when None.
    regression : {'c','ct','ctt','n'}, optional
        Constant and trend order to include in regression.
        'c' : constant only (default).
        'ct' : constant and trend.
        'ctt' : constant, and linear and quadratic trend.
        'n' : no constant, no trend.
    autolag : {'AIC', 'BIC', 't-stat', None}, optional
        Method to use when automatically determining the lag length among the values 0, 1, ..., maxlag.
    store : bool, optional
        If True, then a result instance is returned additionally to the adf statistic. Default is False.
    regresults : bool, optional
        If True, the full regression results are returned. Default is False.

    Returns
    -------
    adf : float
        The test statistic.
    pvalue : float
        MacKinnon’s approximate p-value based on MacKinnon (1994, 2010).
    usedlag : int
        The number of lags used.
    nobs : int
        The number of observations used for the ADF regression and calculation of the critical values.
    critical_values : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 % levels. Based on MacKinnon (2010).
    icbest : float
        The maximized information criterion if autolag is not None.
    resstore : ResultStore, optional
        A dummy class with results attached as attributes.
    """
    return adfuller(
        x,
        maxlag=maxlag,
        regression=regression,
        autolag=autolag,
        store=store,
        regresults=regresults
    )
