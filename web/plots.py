import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import numpy as np
from scipy import stats

# Color palette (Plotly-compatible)
COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "cyan": "#17becf",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "steelblue": "#4682b4",
}

def create_level_chart(freq_name, df, col_y, col_x):
    """Creates a dual-axis chart for level data."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[col_y],
            name="S&P 500 Close Price",
            line=dict(color=COLORS["blue"], width=2),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[col_x],
            name="Monthly Total Sunspot Number",
            mode="markers",
            marker=dict(color=COLORS["orange"], size=4),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title_text=f"{freq_name} S&P 500 and Sunspots Over Time",
        xaxis=dict(title="Date", tickformat="%Y", dtick="M24", tickangle=45),
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.update_yaxes(
        title=dict(text="S&P 500 Close Price", font=dict(color=COLORS["blue"])),
        tickfont=dict(color=COLORS["blue"]),
        secondary_y=False,
    )
    fig.update_yaxes(
        title=dict(text="Sunspot Number", font=dict(color=COLORS["orange"])),
        tickfont=dict(color=COLORS["orange"]),
        secondary_y=True,
    )

    return fig


def create_log_scale_chart(freq_name, df, col_y, col_x):
    """Creates a dual-axis chart with log scale."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[col_y],
            name="S&P 500 Close Price",
            line=dict(color=COLORS["blue"], width=2),
        ),
        secondary_y=False,
    )

    sunspot_data = df[col_x].replace(0, np.nan)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=sunspot_data,
            name="Sunspot Number",
            mode="markers",
            marker=dict(color=COLORS["orange"], size=4),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title_text=f"{freq_name} Data (Log Scale)",
        xaxis=dict(title="Date", tickformat="%Y", dtick="M24", tickangle=45),
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.update_yaxes(
        title=dict(text="S&P 500 (Log)", font=dict(color=COLORS["blue"])),
        tickfont=dict(color=COLORS["blue"]),
        type="log",
        gridcolor="rgba(0,0,0,0.1)",
        secondary_y=False,
    )
    fig.update_yaxes(
        title=dict(text="Sunspots (Log)", font=dict(color=COLORS["orange"])),
        tickfont=dict(color=COLORS["orange"]),
        type="log",
        secondary_y=True,
    )

    return fig


def create_monthly_chart_with_avg(df, col_y_eom, col_y_avg, col_x):
    """Creates monthly chart showing EOM and Average close prices."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[col_y_eom],
            name="S&P 500 EOM Close",
            line=dict(color=COLORS["blue"], width=2),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[col_y_avg],
            name="S&P 500 Avg Close",
            line=dict(color=COLORS["cyan"], width=1, dash="dash"),
            opacity=0.7,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[col_x],
            name="Sunspot Number",
            line=dict(color=COLORS["orange"], width=1.5),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="Monthly S&P 500 and Sunspots",
        xaxis=dict(title="Date", tickformat="%Y", dtick="M24", tickangle=45),
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.update_yaxes(
        title=dict(text="S&P 500 Close Price", font=dict(color=COLORS["blue"])),
        tickfont=dict(color=COLORS["blue"]),
        secondary_y=False,
    )
    fig.update_yaxes(
        title=dict(text="Sunspot Number", font=dict(color=COLORS["orange"])),
        tickfont=dict(color=COLORS["orange"]),
        secondary_y=True,
    )

    return fig


def create_diagnostic_plots(residuals):
    """Generates a 4-panel diagnostic plot."""

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Residuals over Time",
            "Distribution of Residuals",
            "Q-Q Plot",
            "Autocorrelation (ACF)",
        ),
    )

    # 1. Residuals over Time
    fig.add_trace(
        go.Scatter(
            y=residuals,
            mode="lines",
            name="Residuals",
            line=dict(color=COLORS["blue"], width=1),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=0, line=dict(color=COLORS["red"], dash="dash", width=1.5), row=1, col=1
    )

    # 2. Histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            name="Histogram",
            histnorm="probability density",
            marker_color=COLORS["green"],
            opacity=0.6,
        ),
        row=1,
        col=2,
    )
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    normal_pdf = stats.norm.pdf(x_range, residuals.mean(), residuals.std())
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=normal_pdf,
            mode="lines",
            name="Normal",
            line=dict(color="black", width=2, dash="dash"),
        ),
        row=1,
        col=2,
    )

    # 3. Q-Q Plot
    qq_data = sm.ProbPlot(residuals)
    theoretical_quantiles = qq_data.theoretical_quantiles
    sample_quantiles = qq_data.sample_quantiles
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode="markers",
            name="Q-Q",
            marker=dict(color=COLORS["blue"], size=5),
        ),
        row=2,
        col=1,
    )
    min_val = min(min(theoretical_quantiles), min(sample_quantiles))
    max_val = max(max(theoretical_quantiles), max(sample_quantiles))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Normal Line",
            line=dict(color=COLORS["red"], dash="dash"),
        ),
        row=2,
        col=1,
    )

    # 4. ACF
    acf_values = sm.tsa.acf(residuals, nlags=20)
    n = len(residuals)
    conf_interval = 1.96 / np.sqrt(n)
    fig.add_trace(
        go.Bar(
            x=list(range(len(acf_values))),
            y=acf_values,
            name="ACF",
            marker_color=COLORS["steelblue"],
        ),
        row=2,
        col=2,
    )
    fig.add_hline(
        y=conf_interval,
        line=dict(color=COLORS["red"], dash="dash", width=1),
        row=2,
        col=2,
    )
    fig.add_hline(
        y=-conf_interval,
        line=dict(color=COLORS["red"], dash="dash", width=1),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=600, title_text="Model Residuals Diagnostics", showlegend=False
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")

    return fig


def create_seasonality_chart(monthly_results):
    """Create bar chart of monthly coefficients."""
    fig = go.Figure()

    colors = [
        COLORS["green"] if p < 0.05 else COLORS["steelblue"]
        for p in monthly_results["p-value"]
    ]

    fig.add_trace(
        go.Bar(
            x=monthly_results["Month"],
            y=monthly_results["Coefficient"],
            marker_color=colors,
            text=[f"p={p:.3f}" for p in monthly_results["p-value"]],
            textposition="outside",
        )
    )

    fig.add_hline(y=0, line=dict(color="black", width=1))

    fig.update_layout(
        title="Monthly Seasonal Effects (vs January baseline)",
        xaxis_title="Month",
        yaxis_title="Coefficient (deviation from January)",
        height=350,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig
