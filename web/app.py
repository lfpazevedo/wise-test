import sys
import os
import pandas as pd
import numpy as np
from flask import Flask
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# Add project root to path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import run_full_analysis_pipeline
from src.data.processing.df_daily import get_daily_merged_data
from src.data.processing.df_monthly import get_monthly_merged_data
from web.plots import (
    create_level_chart,
    create_log_scale_chart,
    create_monthly_chart_with_avg,
    create_diagnostic_plots,
    create_seasonality_chart,
)

# 1. SETUP & DATA LOADING
# ---------------------------------------------------------
server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname="/",
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True,
)
app.title = "Market Myth Buster"


def load_data():
    daily_df = get_daily_merged_data()
    monthly_df = get_monthly_merged_data()

    if not isinstance(daily_df.index, pd.DatetimeIndex):
        if "date" in daily_df.columns:
            daily_df = daily_df.set_index("date")
    daily_df = daily_df.sort_index()

    if not isinstance(monthly_df.index, pd.DatetimeIndex):
        if "date" in monthly_df.columns:
            monthly_df = monthly_df.set_index("date")
    monthly_df = monthly_df.sort_index()

    return daily_df, monthly_df


daily_df, monthly_df = load_data()


# 3. DASH LAYOUT
# ---------------------------------------------------------
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H1(
                                "🌞 S&P 500 vs Sunspots", className="text-center mb-2"
                            ),
                            html.H5(
                                "Econometric Myth Buster Dashboard",
                                className="text-center text-muted",
                            ),
                        ]
                    ),
                    width=12,
                )
            ],
            className="mb-4 mt-3",
        ),
        # Main Tabs
        dbc.Tabs(
            [
                # TAB 1: DATA EXPLORATION
                dbc.Tab(
                    label="📊 1. Data Exploration",
                    children=[
                        html.Br(),
                        html.H4("Daily Data", className="text-primary"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        figure=create_level_chart(
                                            "Daily",
                                            daily_df,
                                            "SP500_Close",
                                            "monthly_total_sunspot_number",
                                        )
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        figure=create_log_scale_chart(
                                            "Daily",
                                            daily_df,
                                            "SP500_Close",
                                            "monthly_total_sunspot_number",
                                        )
                                    ),
                                    width=6,
                                ),
                            ]
                        ),
                        html.Hr(),
                        html.H4("Monthly Data", className="text-primary"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        figure=create_monthly_chart_with_avg(
                                            monthly_df,
                                            "SP500_Close_EOM",
                                            "SP500_Close_Avg",
                                            "monthly_total_sunspot_number",
                                        )
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        figure=create_log_scale_chart(
                                            "Monthly",
                                            monthly_df,
                                            "SP500_Close_EOM",
                                            "monthly_total_sunspot_number",
                                        )
                                    ),
                                    width=6,
                                ),
                            ]
                        ),
                        html.Br(),
                        dbc.Card(
                            [
                                dbc.CardHeader(html.H5("📈 Data Summary")),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.P(
                                                            f"Daily observations: {len(daily_df):,}"
                                                        ),
                                                        html.P(
                                                            f"Date range: {daily_df.index.min().strftime('%Y-%m-%d')} to {daily_df.index.max().strftime('%Y-%m-%d')}"
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.P(
                                                            f"Monthly observations: {len(monthly_df):,}"
                                                        ),
                                                        html.P(
                                                            f"Date range: {monthly_df.index.min().strftime('%Y-%m')} to {monthly_df.index.max().strftime('%Y-%m')}"
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ],
                ),
                # TAB 2: ECONOMETRIC ANALYSIS
                dbc.Tab(
                    label="🔬 2. Econometric Analysis",
                    children=[
                        html.Br(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "▶️ Run Full Analysis",
                                        id="run-btn",
                                        color="primary",
                                        size="lg",
                                        className="w-100",
                                    ),
                                    width={"size": 4, "offset": 4},
                                )
                            ]
                        ),
                        html.Br(),
                        # Loading spinner
                        dcc.Loading(
                            id="loading",
                            type="circle",
                            children=[
                                # Step 0: Variance Stabilization
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5(
                                                "📊 Step 0: Variance Stabilization (Breusch-Pagan Test)"
                                            )
                                        ),
                                        dbc.CardBody(id="variance-results"),
                                    ],
                                    className="mb-3",
                                ),
                                # Step 1: Stationarity Tests
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5(
                                                "📉 Step 1: Stationarity Tests (ADF)"
                                            )
                                        ),
                                        dbc.CardBody(id="stationarity-results"),
                                    ],
                                    className="mb-3",
                                ),
                                # Step 2: Model Comparison
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("🔄 Step 2: ADL Model Comparison")
                                        ),
                                        dbc.CardBody(id="model-comparison"),
                                    ],
                                    className="mb-3",
                                ),
                                # Step 3: Best Model Summary
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5(
                                                "🏆 Step 3: Best Model - OLS Regression Summary"
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.Pre(
                                                    id="ols-summary",
                                                    style={
                                                        "whiteSpace": "pre",
                                                        "fontFamily": "monospace",
                                                        "fontSize": "11px",
                                                        "backgroundColor": "#f8f9fa",
                                                        "padding": "15px",
                                                        "overflowX": "auto",
                                                        "maxHeight": "400px",
                                                        "overflowY": "auto",
                                                    },
                                                )
                                            ]
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                # Step 4: Granger Causality
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("🔗 Step 4: Granger Causality Test")
                                        ),
                                        dbc.CardBody(id="granger-results"),
                                    ],
                                    className="mb-3",
                                ),
                                # Step 5: Seasonality
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("📅 Step 5: Seasonality Analysis")
                                        ),
                                        dbc.CardBody(id="seasonality-results"),
                                    ],
                                    className="mb-3",
                                ),
                                # Step 6: Diagnostics
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5("🔍 Step 6: Model Diagnostics")
                                        ),
                                        dbc.CardBody(
                                            [dcc.Graph(id="diagnostic-plots")]
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                            ],
                        ),
                    ],
                ),
                # TAB 3: RESULTS & FORECAST
                dbc.Tab(
                    label="📋 3. Results & Forecast",
                    children=[
                        html.Br(),
                        # Key Metrics Cards
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("📈 ADL Forecast"),
                                            dbc.CardBody(
                                                html.H4(
                                                    id="res-forecast",
                                                    className="text-primary",
                                                )
                                            ),
                                        ]
                                    ),
                                    width=3,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("🔗 Granger Causality"),
                                            dbc.CardBody(
                                                html.H4(
                                                    id="res-granger",
                                                    className="text-danger",
                                                )
                                            ),
                                        ]
                                    ),
                                    width=3,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("📊 Heteroscedasticity"),
                                            dbc.CardBody(
                                                html.H4(
                                                    id="res-bp",
                                                    className="text-warning",
                                                )
                                            ),
                                        ]
                                    ),
                                    width=3,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("📐 Normality"),
                                            dbc.CardBody(
                                                html.H4(
                                                    id="res-jb", className="text-info"
                                                )
                                            ),
                                        ]
                                    ),
                                    width=3,
                                ),
                            ],
                            className="mb-4",
                        ),
                        # Executive Summary
                        dbc.Card(
                            [
                                dbc.CardHeader(html.H5("📝 Executive Summary")),
                                dbc.CardBody(
                                    html.Pre(
                                        id="res-summary",
                                        style={
                                            "whiteSpace": "pre-wrap",
                                            "fontFamily": "monospace",
                                            "backgroundColor": "#f8f9fa",
                                            "padding": "20px",
                                            "fontSize": "13px",
                                        },
                                    )
                                ),
                            ]
                        ),
                    ],
                ),
            ]
        ),
    ],
    fluid=True,
)


# 4. CALLBACKS
# ---------------------------------------------------------
@app.callback(
    [
        Output("variance-results", "children"),
        Output("stationarity-results", "children"),
        Output("model-comparison", "children"),
        Output("ols-summary", "children"),
        Output("granger-results", "children"),
        Output("seasonality-results", "children"),
        Output("diagnostic-plots", "figure"),
        Output("res-forecast", "children"),
        Output("res-granger", "children"),
        Output("res-bp", "children"),
        Output("res-jb", "children"),
        Output("res-summary", "children"),
    ],
    [Input("run-btn", "n_clicks")],
    prevent_initial_call=True,
)
def run_full_analysis(n_clicks):
    from io import StringIO
    import sys

    # Capture stdout to parse for variance/stationarity logs (legacy requirement)
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    # Run the centralized pipeline
    # Use existing daily_df/monthly_df to avoid reloading
    pipeline_results = run_full_analysis_pipeline(
        daily_df=daily_df, monthly_df=monthly_df, verbose=True, show_plots=False
    )

    full_output = mystdout.getvalue()
    sys.stdout = old_stdout

    # Extract results from pipeline dictionary
    adl_results = pipeline_results.get("adl_results", {})
    granger_results = pipeline_results.get("granger_result", {})
    seasonal_results = pipeline_results.get("seasonal_joint_result", {})
    monthly_effects = pipeline_results.get("seasonal_monthly_df", pd.DataFrame())
    famous_effects = pipeline_results.get("seasonal_effects", {})
    summary_text = pipeline_results.get("executive_summary", "")

    # Parse variance stabilization section from captured stdout
    variance_lines = []
    stationarity_lines = []

    in_variance = False
    in_stationarity = False

    for line in full_output.split("\n"):
        # Variance stabilization section
        if "Variance Stabilization" in line or "BP Test" in line:
            in_variance = True
            in_stationarity = False

        # Stationarity section
        if "Stationarity" in line and "Variance" not in line:
            in_variance = False
            in_stationarity = True

        # Model comparison marks end of stationarity
        if "Model Comparison" in line or "Rolling CV" in line:
            in_stationarity = False

        if in_variance:
            variance_lines.append(line)
        elif in_stationarity:
            stationarity_lines.append(line)

    # 0. Variance Stabilization Results
    variance_html = html.Div(
        [
            html.Pre(
                (
                    "\n".join(variance_lines[:20])
                    if variance_lines
                    else "Checking variance stability..."
                ),
                style={"fontFamily": "monospace", "fontSize": "12px"},
            ),
            dbc.Alert(
                [
                    html.Strong("📌 Purpose: "),
                    "The Breusch-Pagan test checks if variance is constant over time. ",
                    "If heteroscedasticity is detected (p < 0.05), we apply log transformation to stabilize variance.",
                ],
                color="info",
                className="mt-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Y (S&P 500) Transformation"),
                                    dbc.CardBody(
                                        [
                                            html.P(
                                                (
                                                    f"✓ Log transform applied"
                                                    if adl_results.get(
                                                        "y_logged", False
                                                    )
                                                    else "✗ No log transform (variance is stable)"
                                                ),
                                                className="mb-0",
                                            ),
                                        ]
                                    ),
                                ],
                                color=(
                                    "success"
                                    if adl_results.get("y_logged", False)
                                    else "light"
                                ),
                                outline=True,
                            )
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("X (Sunspots) Transformation"),
                                    dbc.CardBody(
                                        [
                                            html.P(
                                                (
                                                    f"✓ Log transform applied"
                                                    if adl_results.get(
                                                        "x_logged", False
                                                    )
                                                    else "✗ No log transform (variance is stable)"
                                                ),
                                                className="mb-0",
                                            ),
                                        ]
                                    ),
                                ],
                                color=(
                                    "success"
                                    if adl_results.get("x_logged", False)
                                    else "light"
                                ),
                                outline=True,
                            )
                        ],
                        width=6,
                    ),
                ],
                className="mt-2",
            ),
        ]
    )

    # 1. Stationarity Results
    stationarity_html = html.Div(
        [
            html.Pre(
                (
                    "\n".join(stationarity_lines[:20])
                    if stationarity_lines
                    else "Running stationarity tests..."
                ),
                style={"fontFamily": "monospace", "fontSize": "12px"},
            ),
            dbc.Alert(
                [
                    html.Strong("📌 Purpose: "),
                    "The ADF test checks if a time series is stationary (constant mean/variance). ",
                    "If p-value > 0.05, the series has a unit root (non-stationary) and needs differencing.",
                ],
                color="info",
                className="mt-2",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Transformation Applied"),
                    dbc.CardBody(
                        [
                            html.P(
                                (
                                    f"✓ First differencing applied (Δ)"
                                    if adl_results.get("differenced", False)
                                    else "✗ No differencing needed (already stationary)"
                                ),
                                className="mb-0",
                            ),
                        ]
                    ),
                ],
                color="warning" if adl_results.get("differenced", False) else "light",
                outline=True,
                className="mt-2",
            ),
        ]
    )

    # 2. Model Comparison Table
    results_table = adl_results.get("results_table", pd.DataFrame())
    if not results_table.empty:
        model_table = dash_table.DataTable(
            data=results_table.round(4).to_dict("records"),
            columns=[{"name": i, "id": i} for i in results_table.columns],
            style_cell={"textAlign": "center", "padding": "8px", "fontSize": "12px"},
            style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
            style_data_conditional=[
                {
                    "if": {"row_index": 0},
                    "backgroundColor": "#d4edda",
                    "fontWeight": "bold",
                }
            ],
            page_size=10,
        )
        best_model = adl_results.get("best_model", {})
        model_comparison_html = html.Div(
            [
                model_table,
                dbc.Alert(
                    [
                        html.Strong(
                            f"🏆 Best Model: ADL({best_model.get('y_lags', '?')},{best_model.get('x_lags', '?')}) "
                        ),
                        f"selected by lowest RMSFE = {adl_results.get('rmsfe', 0):.4f}",
                    ],
                    color="success",
                    className="mt-2",
                ),
            ]
        )
    else:
        model_comparison_html = html.P("No model results available.")

    # 3. OLS Summary
    ols_model = adl_results.get("best_model", {}).get("model")
    ols_summary_text = (
        ols_model.summary().as_text() if ols_model else "No model available"
    )

    # 4. Granger Causality
    granger_table_data = granger_results.get("results", [])
    if granger_table_data:
        granger_df = pd.DataFrame(granger_table_data)
        granger_table = dash_table.DataTable(
            data=granger_df.round(4).to_dict("records"),
            columns=[{"name": i, "id": i} for i in granger_df.columns],
            style_cell={"textAlign": "center", "padding": "8px", "fontSize": "12px"},
            style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
            style_data_conditional=[
                {
                    "if": {"filter_query": "{significant} = true"},
                    "backgroundColor": "#d4edda",
                }
            ],
        )
        granger_causes = granger_results.get("granger_causes", False)
        granger_html = html.Div(
            [
                granger_table,
                dbc.Alert(
                    [
                        html.Strong("Verdict: "),
                        (
                            "✓ Sunspots Granger-cause S&P 500"
                            if granger_causes
                            else "✗ Sunspots do NOT Granger-cause S&P 500"
                        ),
                        f" (min p-value: {granger_results.get('min_pvalue', 1):.4f})",
                    ],
                    color="success" if granger_causes else "warning",
                    className="mt-2",
                ),
            ]
        )
    else:
        granger_html = html.P("Granger test results not available.")

    # 5. Seasonality Analysis
    # Create seasonality chart
    seas_chart = create_seasonality_chart(monthly_effects)

    # Famous effects summary
    effects_text = []
    for effect_name, effect_data in famous_effects.items():
        if isinstance(effect_data, dict):
            sig = "✓" if effect_data.get("significant") else "✗"
            p_val = effect_data.get("p_value", None)
            p_val_str = f"{p_val:.4f}" if isinstance(p_val, float) else "N/A"
            effects_text.append(
                f"{sig} {effect_name.replace('_', ' ').title()}: p={p_val_str}"
            )

    seasonality_html = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col([dcc.Graph(figure=seas_chart)], width=8),
                    dbc.Col(
                        [
                            html.H6("Famous Calendar Effects:"),
                            (
                                html.Ul([html.Li(e) for e in effects_text])
                                if effects_text
                                else html.P("No famous effects analyzed")
                            ),
                            dbc.Alert(
                                [
                                    html.Strong("Joint F-test: "),
                                    f"p-value = {seasonal_results.get('p_value', 1):.4f}. ",
                                    (
                                        "Seasonality detected!"
                                        if seasonal_results.get("has_seasonality")
                                        else "No significant seasonality."
                                    ),
                                ],
                                color=(
                                    "success"
                                    if seasonal_results.get("has_seasonality")
                                    else "secondary"
                                ),
                                className="mt-2",
                            ),
                        ],
                        width=4,
                    ),
                ]
            )
        ]
    )

    # 6. Diagnostic Plots
    residuals = (
        adl_results.get("best_model", {}).get("model").resid
        if adl_results.get("best_model", {}).get("model")
        else np.array([0])
    )
    fig_diagnostics = create_diagnostic_plots(residuals)

    # 7. Summary metrics
    forecast_val = adl_results.get("forecast", 0)
    last_actual = adl_results.get("last_actual", 1)
    pct_change = ((forecast_val - last_actual) / last_actual) * 100
    forecast_display = f"{forecast_val:,.2f} ({pct_change:+.2f}%)"

    bp_pvalue = adl_results.get("best_model", {}).get("bp_pvalue", 1)
    jb_pvalue = adl_results.get("best_model", {}).get("jb_pvalue", 1)

    bp_res = f"{('Present' if bp_pvalue < 0.05 else 'Absent')} (p={bp_pvalue:.4f})"
    jb_res = f"{('Non-Normal' if jb_pvalue < 0.05 else 'Normal')} (p={jb_pvalue:.4f})"

    granger_p = granger_results.get("min_pvalue", 1)
    granger_res_text = f"{('ACCEPTED' if granger_results.get('granger_causes') else 'REJECTED')} (p={granger_p:.4f})"

    return (
        variance_html,
        stationarity_html,
        model_comparison_html,
        ols_summary_text,
        granger_html,
        seasonality_html,
        fig_diagnostics,
        forecast_display,
        granger_res_text,
        bp_res,
        jb_res,
        summary_text,
    )


# 5. RUN SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)