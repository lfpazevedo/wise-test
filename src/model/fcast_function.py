import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import jarque_bera
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import re


def check_variance_stability(
    series: pd.Series, name: str, significance: float = 0.05
) -> tuple[pd.Series, bool]:
    """
    Check if variance is stable using BP test on trend residuals.
    If heteroscedasticity is detected, apply log transformation.

    Returns:
        Tuple of (transformed_series, was_logged)
    """
    clean = series.dropna()
    if len(clean) < 4:
        print(f"Variance Check {name}: Insufficient data, skipping")
        return series, False

    # Fit trend model: Y ~ Time
    y = clean.values
    x = pd.DataFrame({"time": np.arange(len(y))})
    x = sm.add_constant(x, has_constant="add")  # Explicitly add constant

    model = sm.OLS(y, x).fit()

    # Breusch-Pagan test on trend residuals
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    p_value = bp_test[1]

    print(f"BP Test {name} (Level): p-value = {p_value:.4f}")

    if p_value < significance:
        print(f"  >> Heteroscedasticity detected. Applying LOG transformation.")

        # Check for zeros or negatives
        has_zeros_or_negatives = (series <= 0).any()

        if has_zeros_or_negatives:
            # Use log1p(x) = log(1+x) to avoid -inf at zero
            print(
                f"     Note: {name} contains zeros/negatives. Using np.log1p() to avoid -inf."
            )
            transformed = np.log1p(series)
        else:
            transformed = np.log(series)

        return transformed, True
    else:
        print(f"  >> Variance stable. Using LEVEL.")
        return series, False


def check_stationarity(
    series: pd.Series, name: str, significance: float = 0.05
) -> tuple[float, bool]:
    """
    Perform ADF test and return p-value and stationarity status.
    """
    clean = series.dropna()
    if len(clean) < 4:
        print(f"ADF Test {name}: Insufficient data")
        return 1.0, False

    result = adfuller(clean)
    p_value = result[1]
    is_stationary = p_value <= significance
    print(
        f"ADF Test {name}: p-value = {p_value:.4f} ({'Stationary' if is_stationary else 'Non-Stationary'})"
    )
    return p_value, is_stationary


def check_normality(residuals: np.ndarray) -> tuple[float, bool, str]:
    """
    Perform Jarque-Bera test for normality on residuals.

    Returns:
        Tuple of (p_value, is_normal, interpretation)
    """
    if len(residuals) < 4:
        return np.nan, False, "Insufficient data"

    statistic, p_value = jarque_bera(residuals)
    is_normal = p_value > 0.05

    if is_normal:
        interpretation = "Normal"
    else:
        interpretation = "Non-Normal (Fat Tails)"

    return p_value, is_normal, interpretation


def create_lag_features(
    df: pd.DataFrame, target_col: str, exog_col: str, y_lags: int, x_lags: int
) -> tuple[pd.DataFrame, list]:
    """
    Create lagged features for ADL model.
    """
    temp_df = pd.DataFrame(index=df.index)
    temp_df["Y"] = df[target_col].values

    cols = []
    for i in range(1, y_lags + 1):
        col_name = f"Y_lag{i}"
        temp_df[col_name] = df[target_col].shift(i).values
        cols.append(col_name)

    for i in range(1, x_lags + 1):
        col_name = f"X_lag{i}"
        temp_df[col_name] = df[exog_col].shift(i).values  # Lagged exogenous
        cols.append(col_name)

    return temp_df, cols


def get_inference_table(model, use_robust: bool = False) -> pd.DataFrame:
    """
    Safely extracts model statistics into a DataFrame.
    Handles both Pandas Series (named) and NumPy Arrays (unnamed) inputs.
    """
    if use_robust:
        robust_model = model.get_robustcov_results(cov_type="HC3")
        params = robust_model.params
        bse = robust_model.bse
        tvalues = robust_model.tvalues
        pvalues = robust_model.pvalues
    else:
        params = model.params
        bse = model.bse
        tvalues = model.tvalues
        pvalues = model.pvalues

    # Construct DataFrame WITHOUT .values to handle both Series and arrays
    inference_df = pd.DataFrame(
        {"Coefficient": params, "Std.Error": bse, "t-stat": tvalues, "p-value": pvalues}
    )

    # Add significance stars
    inference_df["Significant"] = [
        "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        for p in inference_df["p-value"]
    ]

    # Ensure proper variable names from model
    if hasattr(model, "model") and hasattr(model.model, "exog_names"):
        inference_df.index = model.model.exog_names

    return inference_df


def fit_adl_model(
    df: pd.DataFrame, target_col: str, exog_col: str, y_lags: int, x_lags: int
):
    """
    Fit a single ADL(p,q) model and return results.
    """
    temp_df, cols = create_lag_features(df, target_col, exog_col, y_lags, x_lags)
    temp_df = temp_df.dropna()

    if len(temp_df) < len(cols) + 2:
        return None

    # Keep as DataFrame for named columns
    X = temp_df[cols].copy()
    Y = temp_df["Y"].copy()

    # Add constant (returns DataFrame with 'const' column)
    X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(Y, X).fit()

    # Breusch-Pagan test for heteroscedasticity
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    bp_pvalue = bp_test[1]

    # Jarque-Bera test for normality
    jb_pvalue, is_normal, normality_status = check_normality(model.resid)

    # In-sample RMSE
    rmse = np.sqrt(mean_squared_error(Y, model.predict(X)))

    return {
        "model": model,
        "y_lags": y_lags,
        "x_lags": x_lags,
        "aic": model.aic,
        "bic": model.bic,
        "rmse": rmse,
        "bp_pvalue": bp_pvalue,
        "heteroscedasticity": "Present" if bp_pvalue < 0.05 else "Absent",
        "jb_pvalue": jb_pvalue,
        "normality": normality_status,
        "cols": cols,
    }


def run_rolling_forecast_cv(
    df: pd.DataFrame,
    target_col: str,
    exog_col: str,
    max_lags: int = 4,
    test_size_ratio: float = 0.25,
    min_train_obs: int = 60,
    verbose: bool = True,
) -> dict:
    """
    Performs Rolling Window Out-of-Sample Cross-Validation for ADL model selection.

    This is the CORRECT way to select forecasting models - by evaluating actual
    out-of-sample predictive performance, not in-sample fit statistics (AIC/BIC).

    Parameters
    ----------
    df : DataFrame
        Transformed (stationary) data with target and exogenous columns
    target_col : str
        Name of target variable column
    exog_col : str
        Name of exogenous variable column
    max_lags : int
        Maximum lags to test for both Y and X (tests 1 to max_lags)
    test_size_ratio : float
        Proportion of data to use for rolling evaluation (default 0.25 = last 25%)
    min_train_obs : int
        Minimum observations required in training window
    verbose : bool
        Whether to print progress and results

    Returns
    -------
    dict
        Contains best model parameters, all model results, and forecast errors
    """
    if verbose:
        print("\n" + "=" * 60)
        print("ROLLING WINDOW OUT-OF-SAMPLE CROSS-VALIDATION")
        print("=" * 60)
        print("Selecting model based on TRUE predictive performance,")
        print("NOT in-sample fit statistics (AIC/BIC).")
        print()

    # Prepare clean data
    data = df[[target_col, exog_col]].copy().dropna()
    n_obs = len(data)

    # Calculate test window
    test_size = max(int(n_obs * test_size_ratio), 12)  # At least 12 months
    train_end_idx = n_obs - test_size

    if train_end_idx < min_train_obs:
        if verbose:
            print(
                f"ERROR: Insufficient data. Need at least {min_train_obs} training obs."
            )
            print(f"       Available: {train_end_idx}")
        return None

    if verbose:
        print(f"Total observations: {n_obs}")
        print(f"Initial training window: {train_end_idx} obs")
        print(f"Rolling test window: {test_size} periods")
        print(f"Testing models: ADL(1,1) to ADL({max_lags},{max_lags})")
        print()

    # Grid of models to test
    models_grid = [
        (p, q) for p in range(1, max_lags + 1) for q in range(1, max_lags + 1)
    ]

    # Store forecast errors for each model
    model_errors = {f"ADL({p},{q})": [] for p, q in models_grid}
    model_forecasts = {f"ADL({p},{q})": [] for p, q in models_grid}
    actuals = []

    # Rolling forecast loop
    if verbose:
        print("Running rolling forecasts", end="")

    for i in range(train_end_idx, n_obs):
        # Expanding window: train on 0...i-1, forecast i
        train_data = data.iloc[:i]
        actual_val = data.iloc[i][target_col]
        actuals.append(actual_val)

        if verbose and (i - train_end_idx) % 10 == 0:
            print(".", end="", flush=True)

        # Test each model specification
        for p, q in models_grid:
            model_name = f"ADL({p},{q})"

            # Construct features for training window
            temp_df = pd.DataFrame(index=train_data.index)
            temp_df["target"] = train_data[target_col].values

            features = []
            # Y lags
            for lag in range(1, p + 1):
                col = f"Y_lag{lag}"
                temp_df[col] = train_data[target_col].shift(lag).values
                features.append(col)
            # X lags
            for lag in range(1, q + 1):
                col = f"X_lag{lag}"
                temp_df[col] = train_data[exog_col].shift(lag).values
                features.append(col)

            temp_df = temp_df.dropna()

            if len(temp_df) < max(p, q) + 5:
                continue  # Skip if window too small

            try:
                X_train = sm.add_constant(temp_df[features], has_constant="add")
                y_train = temp_df["target"]

                model = sm.OLS(y_train, X_train).fit()

                # Construct input for 1-step ahead forecast
                # We need lagged values available at forecast time
                pred_input = {"const": 1.0}

                for lag in range(1, p + 1):
                    pred_input[f"Y_lag{lag}"] = train_data[target_col].iloc[-lag]

                for lag in range(1, q + 1):
                    pred_input[f"X_lag{lag}"] = train_data[exog_col].iloc[-lag]

                # Create prediction DataFrame with correct column order
                pred_df = pd.DataFrame([pred_input])[["const"] + features]
                forecast = model.predict(pred_df)[0]

                # Store forecast and error
                error = actual_val - forecast
                model_errors[model_name].append(error)
                model_forecasts[model_name].append(forecast)

            except Exception:
                # Handle collinearity or numerical issues
                pass

    if verbose:
        print(" Done!")
        print()

    # Calculate aggregated metrics
    results = []

    if verbose:
        print(
            f"{'Model':<12} | {'RMSFE':<10} | {'MAE':<10} | {'Dir Acc%':<10} | {'# Fcsts':<8}"
        )
        print("-" * 60)

    best_rmsfe = float("inf")
    best_model_name = ""
    best_model_params = (1, 1)

    for name, errors in model_errors.items():
        if len(errors) < 5:
            continue  # Skip models with too few valid forecasts

        errors_arr = np.array(errors)
        forecasts_arr = np.array(model_forecasts[name])
        actuals_arr = np.array(actuals[: len(errors)])

        # Root Mean Squared Forecast Error
        rmsfe = np.sqrt(np.mean(errors_arr**2))

        # Mean Absolute Error
        mae = np.mean(np.abs(errors_arr))

        # Direction Accuracy (crucial for trading)
        actual_direction = np.sign(actuals_arr)
        pred_direction = np.sign(forecasts_arr)
        direction_acc = np.mean(actual_direction == pred_direction) * 100

        if verbose:
            print(
                f"{name:<12} | {rmsfe:<10.6f} | {mae:<10.6f} | {direction_acc:<10.1f} | {len(errors):<8}"
            )

        results.append(
            {
                "Model": name,
                "RMSFE": rmsfe,
                "MAE": mae,
                "Direction_Accuracy": direction_acc,
                "N_Forecasts": len(errors),
                "errors": errors,
                "forecasts": forecasts_arr.tolist(),
            }
        )

        if rmsfe < best_rmsfe:
            best_rmsfe = rmsfe
            best_model_name = name
            best_model_params = tuple(map(int, re.findall(r"\d+", name)))

    if verbose:
        print("-" * 60)
        print(f"\n🏆 BEST MODEL (by RMSFE): {best_model_name}")
        print(f"   Out-of-Sample RMSFE: {best_rmsfe:.6f}")
        print(f"   Evaluated on {test_size} recursive 1-step forecasts")
        print()
        print("   WHY THIS MATTERS:")
        print("   - AIC/BIC reward in-sample fit → prone to overfitting")
        print("   - RMSFE measures actual forecasting accuracy")
        print("   - This model would have performed best 'live'")

    return {
        "best_model": best_model_name,
        "best_params": {"y_lags": best_model_params[0], "x_lags": best_model_params[1]},
        "best_rmsfe": best_rmsfe,
        "all_results": pd.DataFrame(results).drop(
            columns=["errors", "forecasts"], errors="ignore"
        ),
        "detailed_results": results,
        "actuals": actuals,
        "test_size": test_size,
    }


def fit_final_model(
    df: pd.DataFrame,
    target_col: str,
    exog_col: str,
    y_lags: int,
    x_lags: int,
) -> dict:
    """
    Fit the selected ADL model on FULL data for final inference and forecasting.

    This is called AFTER model selection via rolling CV.
    """
    temp_df, cols = create_lag_features(df, target_col, exog_col, y_lags, x_lags)
    temp_df = temp_df.dropna()

    X = sm.add_constant(temp_df[cols], has_constant="add")
    Y = temp_df["Y"]

    model = sm.OLS(Y, X).fit()

    # Diagnostics
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    bp_pvalue = bp_test[1]
    jb_pvalue, _, normality_status = check_normality(model.resid)

    return {
        "model": model,
        "y_lags": y_lags,
        "x_lags": x_lags,
        "aic": model.aic,
        "bic": model.bic,
        "bp_pvalue": bp_pvalue,
        "heteroscedasticity": "Present" if bp_pvalue < 0.05 else "Absent",
        "jb_pvalue": jb_pvalue,
        "normality": normality_status,
        "cols": cols,
    }


def forecast_next_period(
    df: pd.DataFrame, model_result: dict, target_col: str, exog_col: str
) -> float:
    """
    Generate forecast for the next period using lagged values.
    """
    model = model_result["model"]
    y_lags = model_result["y_lags"]
    x_lags = model_result["x_lags"]
    cols = model_result["cols"]

    # Build input dict with proper column order
    input_dict = {}

    for i in range(1, y_lags + 1):
        idx = -i
        if abs(idx) <= len(df):
            input_dict[f"Y_lag{i}"] = df[target_col].iloc[idx]
        else:
            input_dict[f"Y_lag{i}"] = np.nan

    for i in range(1, x_lags + 1):
        idx = -i
        if abs(idx) <= len(df):
            input_dict[f"X_lag{i}"] = df[exog_col].iloc[idx]
        else:
            input_dict[f"X_lag{i}"] = np.nan

    # Create DataFrame and add constant
    input_df = pd.DataFrame([input_dict])[cols]
    input_df = sm.add_constant(input_df, has_constant="add")

    forecast = model.predict(input_df)[0]
    return forecast


def plot_model_diagnostics(residuals, title="Model Residuals Diagnostics"):
    """
    Generates a 4-panel plot to visualize:
    1. Volatility Clustering (Time Series)
    2. Fat Tails (Histogram + KDE)
    3. Normality Deviation (Q-Q Plot)
    4. Remaining Autocorrelation (ACF)

    Parameters
    ----------
    residuals : array-like
        Model residuals to analyze
    title : str
        Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # 1. Residuals over Time (Check for Volatility Clustering)
    sns.lineplot(data=residuals, ax=axes[0, 0], color="blue", alpha=0.6)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0, 0].set_title(
        "Residuals over Time\n(Volatility Clustering Check)", fontsize=11
    )
    axes[0, 0].set_ylabel("Residual Error")
    axes[0, 0].set_xlabel("Time Period")
    axes[0, 0].grid(alpha=0.3)

    # 2. Histogram vs Normal Distribution (Check for Fat Tails)
    sns.histplot(
        residuals, kde=True, ax=axes[0, 1], color="green", stat="density", alpha=0.6
    )
    # Overlay standard normal curve
    xmin, xmax = axes[0, 1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, residuals.mean(), residuals.std())
    axes[0, 1].plot(x, p, "k", linewidth=2, label="Normal Distribution", linestyle="--")
    axes[0, 1].set_title("Distribution of Residuals\n(Fat Tails Check)", fontsize=11)
    axes[0, 1].set_xlabel("Residual Value")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Q-Q Plot (Normality Check)
    sm.qqplot(residuals, line="45", ax=axes[1, 0], fit=True)
    axes[1, 0].set_title("Q-Q Plot\n(Normality Deviation Check)", fontsize=11)
    axes[1, 0].grid(alpha=0.3)

    # 4. ACF Plot (Autocorrelation Check)
    plot_acf(residuals, ax=axes[1, 1], lags=20, alpha=0.05)
    axes[1, 1].set_title(
        "Autocorrelation of Residuals\n(White Noise Check)", fontsize=11
    )
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_cv_results(cv_results: dict, title: str = "Rolling CV Model Comparison"):
    """
    Visualize cross-validation results.
    """
    if cv_results is None:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: RMSFE comparison
    results_df = cv_results["all_results"].sort_values("RMSFE")
    colors = [
        "green" if m == cv_results["best_model"] else "steelblue"
        for m in results_df["Model"]
    ]

    axes[0].barh(results_df["Model"], results_df["RMSFE"], color=colors)
    axes[0].set_xlabel("Root Mean Squared Forecast Error (RMSFE)")
    axes[0].set_title("Out-of-Sample Forecasting Performance\n(Lower is Better)")
    axes[0].invert_yaxis()

    # Highlight best
    best_row = results_df[results_df["Model"] == cv_results["best_model"]].iloc[0]
    axes[0].axvline(best_row["RMSFE"], color="green", linestyle="--", alpha=0.7)

    # Plot 2: Direction Accuracy
    axes[1].barh(results_df["Model"], results_df["Direction_Accuracy"], color=colors)
    axes[1].set_xlabel("Direction Accuracy (%)")
    axes[1].set_title("Directional Forecasting Accuracy\n(Higher is Better)")
    axes[1].axvline(50, color="red", linestyle="--", alpha=0.5, label="Random (50%)")
    axes[1].invert_yaxis()
    axes[1].legend()

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def run_adl_forecast(
    df: pd.DataFrame,
    target_col: str = "SP500_Close_EOM",
    exog_col: str = "monthly_total_sunspot_number",
    max_lags: int = 4,
    apply_differencing: bool = True,
    check_variance: bool = True,
    show_diagnostics: bool = True,
    cv_test_ratio: float = 0.25,
    verbose: bool = True,
) -> dict:
    """
    Run ADL forecast pipeline with PROPER out-of-sample model selection.

    KEY CHANGE: Model selection now uses Rolling Window Cross-Validation
    instead of in-sample AIC/BIC. This selects the model that actually
    forecasts best, not the one that fits historical data best.

    Parameters
    ----------
    df : DataFrame
        Input data containing target and exogenous variables
    target_col : str
        Name of the target column (endogenous variable)
    exog_col : str
        Name of the exogenous variable column
    max_lags : int
        Maximum number of lags for ADL model
    apply_differencing : bool
        Whether to apply differencing to achieve stationarity
    check_variance : bool
        Whether to check and stabilize variance using log transformation
    show_diagnostics : bool
        Whether to display diagnostic plots
    cv_test_ratio : float
        Proportion of data for rolling CV evaluation (default 0.25)
    verbose : bool
        Whether to print detailed output
    """
    if verbose:
        print("=" * 60)
        print("ADL FORECAST WITH OUT-OF-SAMPLE MODEL SELECTION")
        print("=" * 60)
        print(f"Y (Endogenous): {target_col}")
        print(f"X (Exogenous): {exog_col}")
        print(f"Observations: {len(df)}")
        print()

    # Store original data
    original_df = df[[target_col, exog_col]].copy()
    work_df = df[[target_col, exog_col]].copy()

    # Track transformations
    y_logged = False
    x_logged = False
    differenced = False

    # STEP 1: Variance Stabilization
    if check_variance:
        if verbose:
            print("--- STEP 1: Variance Stabilization (BP Test) ---")
        work_df[target_col], y_logged = check_variance_stability(
            work_df[target_col], "Y"
        )
        work_df[exog_col], x_logged = check_variance_stability(work_df[exog_col], "X")
        if verbose:
            print()

    # STEP 2: Stationarity checks
    if verbose:
        print("--- STEP 2: Stationarity Tests (ADF) ---")
    p_y, stat_y = check_stationarity(
        work_df[target_col], "Y" + (" (Log)" if y_logged else "")
    )
    p_x, stat_x = check_stationarity(
        work_df[exog_col], "X" + (" (Log)" if x_logged else "")
    )

    if apply_differencing and (not stat_y or not stat_x):
        if verbose:
            print("\n>> Applying first differences for stationarity...")
        work_df = work_df.diff().dropna()
        differenced = True

        if verbose:
            print("\n--- Post-Differencing ADF Tests ---")
            check_stationarity(work_df[target_col], "ΔY")
            check_stationarity(work_df[exog_col], "ΔX")
            print()

    # STEP 3: Rolling CV Model Selection (THE KEY IMPROVEMENT)
    if verbose:
        print("--- STEP 3: Out-of-Sample Model Selection (Rolling CV) ---")

    cv_results = run_rolling_forecast_cv(
        work_df,
        target_col,
        exog_col,
        max_lags=max_lags,
        test_size_ratio=cv_test_ratio,
        verbose=verbose,
    )

    if cv_results is None:
        print("ERROR: Cross-validation failed.")
        return None

    best_y_lags = cv_results["best_params"]["y_lags"]
    best_x_lags = cv_results["best_params"]["x_lags"]

    # STEP 4: Fit final model on full data
    if verbose:
        print("\n--- STEP 4: Final Model (Full Sample) ---")
        print(f"Fitting ADL({best_y_lags},{best_x_lags}) on all data for inference...")

    best_result = fit_final_model(
        work_df, target_col, exog_col, best_y_lags, best_x_lags
    )

    # STEP 5: Statistical Inference
    if verbose:
        print("\n--- STEP 5: Statistical Inference ---")

    use_robust = best_result["bp_pvalue"] < 0.05

    if use_robust and verbose:
        print("WARNING: Heteroscedasticity detected. Using HC3 Robust SE.")

    inference_table = get_inference_table(best_result["model"], use_robust=use_robust)

    if verbose:
        print(inference_table.to_string())

        if best_result["jb_pvalue"] < 0.05:
            print(
                "\nNOTE: Non-normal residuals (fat tails). t-stats may be unreliable."
            )

    # STEP 6: Generate forecast
    if verbose:
        print("\n--- STEP 6: Forecast ---")

    forecast_transformed = forecast_next_period(
        work_df, best_result, target_col, exog_col
    )

    last_actual = original_df[target_col].dropna().iloc[-1]

    if y_logged and differenced:
        forecast_value = last_actual * np.exp(forecast_transformed)
        pct_change = (np.exp(forecast_transformed) - 1) * 100
    elif differenced:
        forecast_value = last_actual + forecast_transformed
        pct_change = (
            (forecast_transformed / last_actual) * 100 if last_actual != 0 else 0
        )
    elif y_logged:
        forecast_value = np.exp(forecast_transformed)
        pct_change = ((forecast_value - last_actual) / last_actual) * 100
    else:
        forecast_value = forecast_transformed
        pct_change = (
            ((forecast_value - last_actual) / last_actual) * 100
            if last_actual != 0
            else 0
        )

    if verbose:
        print(f"Last Actual: {last_actual:.2f}")
        print(f"Next Period Forecast: {forecast_value:.2f}")
        print(f"Implied Change: {pct_change:+.2f}%")

    # STEP 7: Diagnostics
    if show_diagnostics:
        if verbose:
            print("\n--- STEP 7: Diagnostics ---")
        plot_model_diagnostics(
            best_result["model"].resid,
            title=f"ADL({best_y_lags},{best_x_lags}) Diagnostics",
        )
        plot_cv_results(cv_results, "Rolling CV: Model Selection")
        plt.show()

    # Build results table for display
    results_table = cv_results["all_results"].copy()
    results_table = results_table.sort_values("RMSFE")

    return {
        "results_table": results_table,
        "cv_results": cv_results,
        "best_model": best_result,
        "inference_table": inference_table,
        "used_robust_se": use_robust,
        "forecast": forecast_value,
        "forecast_transformed": forecast_transformed,
        "y_logged": y_logged,
        "x_logged": x_logged,
        "differenced": differenced,
        "target_col": target_col,
        "exog_col": exog_col,
        "last_actual": last_actual,
        "rmsfe": cv_results["best_rmsfe"],
        "direction_accuracy": (
            results_table[results_table["Model"] == cv_results["best_model"]][
                "Direction_Accuracy"
            ].values[0]
            if len(results_table) > 0
            else None
        ),
    }
