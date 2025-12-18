# wise-test

Econometric toolbox and dashboard exploring links between S&P 500 prices and solar activity (sunspots). It automates data collection, time-series modeling, hypothesis testing, and communicates results via a Dash web UI.

## What the app does
- **Data ingestion**: Fetches S&P 500 prices from Yahoo Finance and monthly sunspot numbers from SILSO; merges daily and monthly views.
- **Diagnostics & transformations**: Breusch-Pagan variance checks, ADF stationarity tests, optional log/differencing.
- **Modeling**: ADL model search with **rolling window cross-validation** (proper out-of-sample selection), robust SE when heteroscedasticity is detected, Jarque-Bera normality check, RMSFE and direction accuracy metrics.
- **Causality tests**: Granger causality to evaluate predictive power of sunspots.
- **Seasonality tests**: Monthly dummy joint F-tests, individual month effects, and famous calendar anomalies (e.g., Santa Claus rally).
- **Daily robustness**: Correlation analysis and nested model comparison at daily frequency.
- **Visualization/UI**: Dash app with data exploration charts, model comparison table, diagnostics, and executive summary.

## Key Methodological Innovation

### ✅ Proper Out-of-Sample Model Selection

**Traditional approach (WRONG for forecasting):**
- Select model by in-sample AIC/BIC
- Problem: Rewards overfitting to historical noise
- Result: Models with great AIC often forecast terribly

**This project's approach (CORRECT):**
- **Rolling Window Cross-Validation**
- Simulates real-world forecasting scenario
- Expanding training window, 1-step ahead forecasts
- Selects model with lowest **RMSFE** (Root Mean Squared Forecast Error)
- Ensures selected model actually predicts well, not just fits well

```
Traditional (In-Sample):  [========= Fit All Data =========] → Pick lowest AIC
                          Problem: AIC = how well you explain the PAST

Proper (Out-of-Sample):   [=== Train ===][Forecast t+1]
                          [==== Train ====][Forecast t+2]
                          [===== Train =====][Forecast t+3]
                          ...
                          → Pick lowest forecast error
                          Goal: How well you predict the FUTURE
```

## Project layout (key files)
- `src/data/api/` … data fetchers (Yahoo Finance, SILSO)
- `src/data/processing/` … merge & plotting utilities
- `src/model/fcast_function.py` … **ADL pipeline with rolling CV** (the core innovation)
- `src/model/daily_analysis.py` … daily frequency robustness checks
- `src/model/granger_test.py` … Granger causality testing
- `src/model/seas_test.py` … seasonality and calendar anomaly tests
- `src/model/summary.py` … executive summary generator
- `src/main.py` … Core analysis pipeline and CLI entry point
- `web/app.py` … Dash dashboard (Flask server wrapper)
- `web/plots.py` … Plotting functions for the dashboard (Plotly)
- `notebooks/analysis.ipynb` … Jupyter notebook with narrative analysis

## Prerequisites
- Python 3.13 (see `.python-version`)
- uv package manager

## Setup
```bash
# Install dependencies
uv sync

# Install project in editable mode
uv pip install -e .
```

## Running the dashboard
```bash
python web/app.py
```
The Dash server will start (by default at http://127.0.0.1:8050). Open in a browser to interact with the analysis steps, diagnostics, and executive summary.

## Running the CLI pipeline
```bash
python -m src.main
```
Runs the ADL pipeline with rolling CV, Granger test, daily robustness check, seasonality analysis, and prints the executive summary to stdout.

## Running the Jupyter Notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```
Interactive narrative analysis with visualizations and step-by-step econometric testing.

## Data notes
- **S&P 500**: Pulled via Yahoo Finance (`^GSPC`), 20 years of history
- **Sunspots**: Monthly totals from SILSO (scraped fixed-width text)
- Missing values are handled during merges
- Transformations guard against zeros when logging

## Results overview

### Key Findings

Based on rigorous econometric analysis with proper out-of-sample validation:

1. **Market Behavior**
   - S&P 500 follows a **Random Walk with Drift** pattern
   - Drift ≈ 0.7% monthly average growth
   - Past returns do NOT predict future returns (EMH confirmed)
   - Fat tails detected → extreme events more likely than normal distribution predicts

2. **Sunspot Hypothesis: REJECTED**
   - Rolling CV selected best forecasting model
   - No sunspot lags are statistically significant
   - Granger causality: Sunspots do NOT Granger-cause S&P 500
   - Daily frequency analysis confirms: no predictive relationship

3. **Seasonality: NOT SIGNIFICANT**
   - Joint F-test: no monthly seasonality
   - Famous effects tested:
     - ✗ Santa Claus Rally (December)
     - ✗ Sell in May
     - ✗ September Effect

4. **Investment Recommendation**
   > ❌ **DO NOT** use sunspot activity as a trading signal. The statistical evidence (evaluated with proper out-of-sample testing) shows no predictive relationship. Any observed correlation is spurious.

### Methodological Strengths

- ✅ **Look-ahead bias prevention**: Sunspot data from month M only predicts returns from M+1 onwards (enforced via lagged variables)
- ✅ **Out-of-sample validation**: Models selected by actual forecasting performance (RMSFE), not in-sample fit (AIC/BIC)
- ✅ **Robustness checks**: Multiple frequencies (daily/monthly), multiple tests (ADL, Granger, seasonality)
- ✅ **Proper inference**: Robust standard errors (HC3) when heteroscedasticity detected
- ✅ **Comprehensive diagnostics**: Stationarity (ADF), normality (JB), variance stability (BP), residual autocorrelation

## Model Selection Comparison

**In-Sample (Traditional) vs Out-of-Sample (This Project)**

| Criterion | In-Sample (AIC/BIC) | Out-of-Sample (Rolling CV) |
|-----------|---------------------|----------------------------|
| **Question Asked** | "Which model explains the past best?" | "Which model forecasts the future best?" |
| **Selection Metric** | AIC, BIC, R² | RMSFE (Root Mean Squared Forecast Error) |
| **Risk** | Overfitting to noise | Model complexity naturally penalized |
| **Trading Relevance** | Low (historical fit ≠ future performance) | High (simulates real forecasting) |
| **Implementation** | Fit once on all data | Recursive: train → forecast → expand → repeat |

**Example from this project:**
- Model ADL(3,3) had lowest AIC (best in-sample fit)
- Model ADL(1,1) had lowest RMSFE (best actual forecasting)
- **Selected: ADL(1,1)** ← Simpler model that actually predicts better

## Tests
- `tests/test_silso.py` - Stubs SILSO scraper to verify column parsing and data URL extraction
- Guards against silent breakage if SILSO changes page format

## Citation & Attribution
- **S&P 500 Data**: Yahoo Finance (via `yfinance` library)
- **Sunspot Data**: SILSO (Royal Observatory of Belgium) - [http://www.sidc.be/SILSO/](http://www.sidc.be/SILSO/)

## License
MIT License - See LICENSE file for details

---

**Academic Note**: This project demonstrates the importance of proper model selection in time series forecasting. In-sample metrics (AIC/BIC) are fundamentally flawed for predictive modeling because they conflate explanatory power with forecasting accuracy. Rolling window cross-validation is the econometric standard for selecting models intended for forecasting, not curve-fitting.