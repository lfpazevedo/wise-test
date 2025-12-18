import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.compat import lzip

def perform_breusch_pagan_test(fitted_model):
    """
    Performs the Breusch-Pagan test for heteroscedasticity.
    
    The null hypothesis (H0): Homoscedasticity is present.
    The alternative hypothesis (Ha): Heteroscedasticity exists.
    
    Args:
        fitted_model: A fitted statsmodels regression object.
        
    Returns:
        list: A list of tuples containing the test statistics and p-values.
    """
    names = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    
    # Get the test result
    # het_breuschpagan requires residuals and the exogenous variables (independent variables)
    test_result = sms.het_breuschpagan(fitted_model.resid, fitted_model.model.exog)
    
    return lzip(names, test_result)

if __name__ == "__main__":
    # Create a dataset
    dataframe = pd.DataFrame({'rating': [92, 84, 87, 82, 98, 94, 75, 80, 83, 89],
                              'points': [27, 30, 15, 26, 27, 20, 16, 18, 19, 20],
                              'runs': [5000, 7000, 5102, 8019, 1200, 7210, 6200, 9214, 4012, 3102],
                              'wickets': [110, 120, 110, 80, 90, 119, 116, 100, 90, 76]})

    print("Dataset:")
    print(dataframe.head())
    print("-" * 30)

    # Fit regression model
    # rating as response variable; points, runs, and wickets as explanatory variables
    fit = smf.ols('rating ~ points+runs+wickets', data=dataframe).fit()
    
    # Perform the test
    results = perform_breusch_pagan_test(fit)
    
    print("\nBreusch-Pagan Test Results:")
    for name, value in results:
        print(f"{name}: {value}")
        
    # Interpretation helper
    p_value = dict(results)['p-value']
    print("\nInterpretation:")
    if p_value > 0.05:
        print(f"p-value ({p_value:.4f}) > 0.05. We cannot reject the null hypothesis.")
        print("We do not have enough proof to say that heteroscedasticity is present.")
    else:
        print(f"p-value ({p_value:.4f}) <= 0.05. We reject the null hypothesis.")
        print("Heteroscedasticity is likely present.")
