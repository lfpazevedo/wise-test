from src.data.processing.df_monthly import get_monthly_merged_data
from src.data.processing.df_daily import get_daily_merged_data
from src.model.fcast_function import run_adl_forecast
from src.model.granger_test import run_granger_test
from src.model.daily_analysis import run_daily_robustness_check
from src.model.seas_test import (
    create_seasonal_dummies,
    test_joint_seasonality,
    analyze_individual_months,
    check_famous_effects,
)
from src.model.summary import generate_executive_summary
import numpy as np
import statsmodels.api as sm


def run_full_analysis_pipeline(
    daily_df=None, monthly_df=None, verbose=True, show_plots=False
):
    """
    Orchestrates the full analysis pipeline.
    Can be used by CLI (verbose=True) or Web App (verbose=True/False + return values).

    Returns:
        dict: Dictionary containing all analysis results and models.
    """
    pipeline_results = {}

    # ============================================================
    # DAILY FREQUENCY ROBUSTNESS CHECK
    # ============================================================
    if verbose:
        print("=" * 60)
        print("LOADING DAILY DATA FOR ROBUSTNESS ANALYSIS")
        print("=" * 60)

    if daily_df is None:
        daily_df = get_daily_merged_data()
        daily_df = daily_df.set_index("date")

    if verbose:
        print(f"Daily observations: {len(daily_df)}")
        print(f"Date range: {daily_df.index.min()} to {daily_df.index.max()}")

    # Run daily robustness check
    daily_results = run_daily_robustness_check(
        daily_df,
        price_col="SP500_Close",
        sunspot_col="monthly_total_sunspot_number",
        verbose=verbose,
    )
    pipeline_results["daily_results"] = daily_results

    # ============================================================
    # MONTHLY ANALYSIS (Primary)
    # ============================================================
    if verbose:
        print("\n")
        print("=" * 60)
        print("LOADING MONTHLY DATA FOR PRIMARY ANALYSIS")
        print("=" * 60)

    # Get merged monthly data
    if monthly_df is None:
        merged_df = get_monthly_merged_data()
    else:
        merged_df = monthly_df.copy()

    pipeline_results["merged_df"] = merged_df

    if verbose:
        print("\nMerged Monthly Data Sample:")
        print(merged_df.head())
        print(merged_df.tail())
        print(f"\nMerged Monthly DataFrame shape: {merged_df.shape}")

    # Prepare data for forecasting (set date as index)
    if "date" in merged_df.columns:
        forecast_df = merged_df.set_index("date").copy()
    else:
        forecast_df = merged_df.copy()

    # Run ADL forecast with PROPER out-of-sample model selection
    if verbose:
        print("\n")
    results = run_adl_forecast(
        df=forecast_df,
        target_col="SP500_Close_EOM",
        exog_col="monthly_total_sunspot_number",
        max_lags=4,
        apply_differencing=True,
        check_variance=True,
        cv_test_ratio=0.25,  # Use last 25% for rolling CV
        show_diagnostics=show_plots,
        verbose=verbose,
    )
    pipeline_results["adl_results"] = results

    if results:
        if verbose:
            print("\n" + "=" * 60)
            print("FINAL FORECAST RESULT")
            print("=" * 60)

            # Show transformation summary
            transforms = []
            if results["y_logged"]:
                transforms.append("Log")
            if results["differenced"]:
                transforms.append("Differenced")

            if transforms:
                print(f"Transformations: {' → '.join(transforms)}")

            print(
                f"Best Model: {results['cv_results']['best_model']} (selected by Rolling CV)"
            )
            print(
                f"Robust SE Used: {'Yes (HC3)' if results['used_robust_se'] else 'No (Standard)'}"
            )

            # Show OOS validation metrics
            print(f"Out-of-Sample RMSFE: {results['rmsfe']:.6f}")
            if results.get("direction_accuracy"):
                print(f"Direction Accuracy: {results['direction_accuracy']:.1f}%")

            print(f"Last Actual: {results['last_actual']:.2f}")
            print(f"Next Period Forecast: {results['forecast']:.2f}")

            # Show significant predictors
            sig_vars = results["inference_table"][
                results["inference_table"]["p-value"] < 0.05
            ].index.tolist()
            if sig_vars:
                print(
                    f"Significant Predictors (p<0.05): {', '.join(map(str, sig_vars))}"
                )
            else:
                print("Significant Predictors (p<0.05): None")

            # ECONOMETRIC INTERPRETATION
            print("\n" + "=" * 60)
            print("ECONOMETRIC INSIGHTS")
            print("=" * 60)

            # Check for Random Walk pattern
            non_const_sig = [v for v in sig_vars if str(v).lower() != "const"]
            if len(non_const_sig) == 0 and results["differenced"]:
                print("\n✓ RANDOM WALK DETECTED:")
                if "const" in results["inference_table"].index:
                    drift = results["inference_table"].loc["const", "Coefficient"]
                    if results["y_logged"]:
                        drift_pct = (np.exp(drift) - 1) * 100
                        print(f"  Model: ΔLn(S&P500) ≈ {drift:.4f} + noise")
                        print(
                            f"  Interpretation: Market trends up ~{drift_pct:.2f}% monthly on average."
                        )
                    else:
                        print(f"  Model: ΔS&P500 ≈ {drift:.4f} + noise")

                print("  Past returns do NOT predict future returns.")
                print("  Sunspots do NOT predict future returns.")
                print("  → Consistent with Efficient Market Hypothesis (EMH)")

            # Fat Tails Warning
            if results["best_model"]["jb_pvalue"] < 0.05:
                print("\n⚠ FAT TAILS WARNING:")
                print("  Residuals show non-normal distribution (fat tails).")
                print("  This means 'Black Swan' events (crashes/rallies) occur")
                print("  more frequently than a normal distribution predicts.")
                print("  → Forecast confidence intervals may underestimate risk.")

        else:
            # Recompute sig_vars for silent mode logic if needed below (or just skip)
            sig_vars = results["inference_table"][
                results["inference_table"]["p-value"] < 0.05
            ].index.tolist()

        # RUN GRANGER CAUSALITY TEST
        if verbose:
            print("\n")

        # Reconstruct the transformed dataframe
        work_df = forecast_df[[results["target_col"], results["exog_col"]]].copy()

        if results["y_logged"]:
            if (forecast_df[results["target_col"]] <= 0).any():
                work_df[results["target_col"]] = np.log1p(
                    work_df[results["target_col"]]
                )
            else:
                work_df[results["target_col"]] = np.log(work_df[results["target_col"]])

            if (forecast_df[results["exog_col"]] <= 0).any():
                work_df[results["exog_col"]] = np.log1p(work_df[results["exog_col"]])
            else:
                work_df[results["exog_col"]] = np.log(work_df[results["exog_col"]])

        if results["differenced"]:
            work_df = work_df.diff().dropna()

        work_df_clean = work_df.replace([np.inf, -np.inf], np.nan).dropna()
        pipeline_results["work_df_clean"] = work_df_clean

        if len(work_df_clean) < 10:
            if verbose:
                print("=" * 60)
                print("GRANGER CAUSALITY TEST - SKIPPED")
                print("=" * 60)
                print(f"ERROR: Insufficient clean data ({len(work_df_clean)} rows)")
            granger_result = {"granger_causes": False}
        else:
            granger_result = run_granger_test(
                df=work_df_clean,
                target_col=results["target_col"],
                exog_col=results["exog_col"],
                max_lags=4,
                significance=0.05,
                verbose=verbose,
            )
        pipeline_results["granger_result"] = granger_result

        # Final verdict combining ADL and Granger results
        if verbose:
            print("\n" + "=" * 60)
            print("FINAL HYPOTHESIS TEST VERDICT")
            print("=" * 60)
            print("Question: Do Sunspots predict S&P 500 movements?")
            print()

        adl_supports = any("X_lag" in str(v) or "x_lag" in str(v) for v in sig_vars)
        granger_supports = (
            granger_result.get("granger_causes", False)
            if "error" not in granger_result
            else False
        )
        daily_supports = daily_results.get("supports_hypothesis", False)

        if verbose:
            if adl_supports and granger_supports:
                print("✓ HYPOTHESIS SUPPORTED:")
                print("  Both ADL coefficients AND Granger test show significance.")
                print("  Sunspots have predictive power for S&P 500.")
            elif adl_supports or granger_supports:
                print("⚠ MIXED EVIDENCE:")
                if adl_supports:
                    print("  ADL shows sunspot lags are significant,")
                    print("  BUT Granger test does not confirm causality.")
                else:
                    print("  Granger test shows causality,")
                    print("  BUT ADL coefficients are not significant.")
                print("  → Weak evidence, interpret with caution.")
            else:
                print("✗ HYPOTHESIS REJECTED:")
                print("  Neither ADL nor Granger test show significance.")
                print("  No statistical evidence that Sunspots predict S&P 500.")
                print("  → The observed correlation is likely spurious.")

            # Add daily frequency findings
            print()
            print("Daily Frequency Robustness Check:")
            if daily_supports:
                print("  ⚠ Some evidence at daily frequency (investigate further)")
            else:
                print("  ✓ Consistent with monthly results - no predictive power")

        # ============================================================
        # SEASONALITY ANALYSIS
        # ============================================================
        if verbose:
            print("\n")
            print("=" * 60)
            print("SEASONALITY ANALYSIS")
            print("=" * 60)
            print("Testing for monthly calendar effects (e.g., December Rally)")

        seasonal_df, seasonal_cols = create_seasonal_dummies(
            work_df_clean, baseline_month=1, prefix="M"
        )
        pipeline_results["seasonal_df"] = seasonal_df
        pipeline_results["seasonal_cols"] = seasonal_cols

        Y_seas = seasonal_df[results["target_col"]]
        X_seas = seasonal_df[seasonal_cols].copy()
        X_seas = sm.add_constant(X_seas)

        try:
            seas_model = sm.OLS(Y_seas, X_seas).fit()

            from statsmodels.stats.diagnostic import het_breuschpagan

            bp_test = het_breuschpagan(seas_model.resid, seas_model.model.exog)
            use_robust_seas = bp_test[1] < 0.05

            joint_result = test_joint_seasonality(
                seas_model, seasonal_cols, verbose=verbose
            )
            pipeline_results["seasonal_joint_result"] = joint_result

            monthly_df = analyze_individual_months(
                seas_model,
                seasonal_cols,
                baseline_month=1,
                use_robust=use_robust_seas,
                verbose=verbose,
            )
            pipeline_results["seasonal_monthly_df"] = monthly_df

            effects = check_famous_effects(monthly_df, verbose=verbose)
            pipeline_results["seasonal_effects"] = effects

            if verbose:
                print("\n" + "=" * 60)
                print("SEASONALITY VERDICT")
                print("=" * 60)

                if joint_result.get("has_seasonality", False):
                    print("✓ Monthly seasonality IS statistically significant.")
                    sig_months = monthly_df[monthly_df["p-value"] < 0.05][
                        "Month"
                    ].tolist()
                    if sig_months:
                        print(f"  Significant months: {', '.join(sig_months)}")
                else:
                    print("✗ No significant monthly seasonality detected.")
                    print("  Calendar effects do not explain S&P 500 returns.")

        except Exception as e:
            if verbose:
                print(f"Error in seasonality analysis: {e}")
            joint_result = {"has_seasonality": False, "p_value": None}
            pipeline_results["seasonal_joint_result"] = joint_result

    # ============================================================
    # EXECUTIVE SUMMARY
    # ============================================================
    summary_text = generate_executive_summary(
        adl_results=results,
        granger_results=granger_result,
        seasonal_results=joint_result,
        verbose=verbose,
    )
    pipeline_results["executive_summary"] = summary_text

    return pipeline_results


def main():
    run_full_analysis_pipeline(verbose=True, show_plots=False)


if __name__ == "__main__":
    main()

