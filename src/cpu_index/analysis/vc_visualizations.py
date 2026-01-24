"""
VC-CPU Correlation Visualizations

Creates publication-quality charts for CPU-VC correlation analysis.
Follows project styling conventions from output/visualizations.py.

Functions:
- plot_cpu_vc_timeseries: Dual-axis time series plot
- plot_cross_correlation: CCF bar plot with confidence bands
- plot_rolling_correlation: Rolling correlation with uncertainty
- plot_stage_distribution: Stacked area chart of deal stages
- save_all_visualizations: Generate all plots to output directory
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless operation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Optional seaborn import for enhanced styling
try:
    import seaborn as sns
    _SEABORN_AVAILABLE = True
except ImportError:
    _SEABORN_AVAILABLE = False


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "lines.linewidth": 1.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# Color palette for consistency
COLORS = {
    "cpu": "#1f77b4",       # Blue for CPU
    "vc": "#ff7f0e",        # Orange for VC
    "seed": "#2ca02c",      # Green for seed stage
    "early": "#9467bd",     # Purple for early stage
    "late": "#d62728",      # Red for late stage
    "significant": "#2ca02c",  # Green for significant
    "not_significant": "#7f7f7f",  # Gray for not significant
    "confidence": "#d62728",  # Red for confidence bounds
}


def _apply_style():
    """Apply consistent matplotlib styling (with seaborn if available)."""
    plt.rcParams.update(STYLE_CONFIG)
    if _SEABORN_AVAILABLE:
        sns.set_style("whitegrid", {"axes.edgecolor": ".3"})


def _ensure_datetime_index(series: pd.Series) -> pd.Series:
    """Ensure series has a datetime index."""
    if not isinstance(series.index, pd.DatetimeIndex):
        if hasattr(series.index, 'to_timestamp'):
            # PeriodIndex
            series = series.copy()
            series.index = series.index.to_timestamp()
        else:
            # Try to convert string index
            series = series.copy()
            series.index = pd.to_datetime(series.index)
    return series


def _save_or_return(
    fig: plt.Figure,
    output_path: Optional[Union[str, Path]],
) -> Union[plt.Figure, Path]:
    """Save figure to path or return figure object."""
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return fig


# =============================================================================
# MAIN VISUALIZATION FUNCTIONS
# =============================================================================

def plot_cpu_vc_timeseries(
    cpu_series: pd.Series,
    vc_series: pd.Series,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "CPU Index vs VC Deal Activity",
    show_trend: bool = True,
) -> Union[plt.Figure, Path]:
    """
    Create dual-axis time series plot of CPU index and VC deal count.

    Args:
        cpu_series: CPU index values with datetime index
        vc_series: VC deal count values with datetime index
        output_path: Optional path to save figure. If None, returns Figure.
        title: Chart title
        show_trend: Whether to show trend lines (default: True)

    Returns:
        matplotlib Figure if output_path is None, else Path to saved file

    Example:
        >>> cpu = pd.Series([100, 110, 105], index=pd.date_range('2020', periods=3, freq='MS'))
        >>> vc = pd.Series([50, 45, 55], index=pd.date_range('2020', periods=3, freq='MS'))
        >>> fig = plot_cpu_vc_timeseries(cpu, vc)
    """
    _apply_style()

    # Ensure datetime indices
    cpu_series = _ensure_datetime_index(cpu_series)
    vc_series = _ensure_datetime_index(vc_series)

    # Align series
    combined = pd.DataFrame({
        'cpu': cpu_series,
        'vc': vc_series,
    }).dropna()

    if len(combined) == 0:
        raise ValueError("No overlapping data between CPU and VC series")

    dates = combined.index
    cpu_values = combined['cpu'].values
    vc_values = combined['vc'].values

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # CPU on left axis
    ax1.plot(dates, cpu_values, color=COLORS["cpu"], linewidth=2, label="CPU Index")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("CPU Index", color=COLORS["cpu"])
    ax1.tick_params(axis='y', labelcolor=COLORS["cpu"])

    # Add trend line for CPU
    if show_trend and len(dates) > 2:
        x_numeric = np.arange(len(dates))
        z_cpu = np.polyfit(x_numeric, cpu_values, 1)
        p_cpu = np.poly1d(z_cpu)
        ax1.plot(dates, p_cpu(x_numeric), color=COLORS["cpu"],
                 linestyle='--', linewidth=1, alpha=0.7, label="CPU Trend")

    # VC on right axis
    ax2 = ax1.twinx()
    ax2.plot(dates, vc_values, color=COLORS["vc"], linewidth=2, label="VC Deal Count")
    ax2.set_ylabel("VC Deal Count", color=COLORS["vc"])
    ax2.tick_params(axis='y', labelcolor=COLORS["vc"])

    # Add trend line for VC
    if show_trend and len(dates) > 2:
        z_vc = np.polyfit(x_numeric, vc_values, 1)
        p_vc = np.poly1d(z_vc)
        ax2.plot(dates, p_vc(x_numeric), color=COLORS["vc"],
                 linestyle='--', linewidth=1, alpha=0.7, label="VC Trend")

    # Title and formatting
    ax1.set_title(title)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Remove top spine
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()

    return _save_or_return(fig, output_path)


def plot_cross_correlation(
    ccf_results: Union[pd.DataFrame, dict],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Cross-Correlation: CPU leads VC",
    alpha: float = 0.05,
) -> Union[plt.Figure, Path]:
    """
    Create bar plot of cross-correlations at each lag with confidence bands.

    Args:
        ccf_results: DataFrame from cross_correlation() or dict with 'results' key
        output_path: Optional path to save figure. If None, returns Figure.
        title: Chart title
        alpha: Significance level for confidence bands (default: 0.05)

    Returns:
        matplotlib Figure if output_path is None, else Path to saved file

    Notes:
        - Positive lags mean CPU leads VC
        - 95% confidence bands shown as dashed horizontal lines
        - Significant correlations highlighted in green
    """
    _apply_style()

    # Handle dict input from analyze_cpu_vc_correlation
    if isinstance(ccf_results, dict):
        if 'results' in ccf_results:
            ccf_df = pd.DataFrame(ccf_results['results'])
        else:
            raise ValueError("Dict must contain 'results' key with CCF data")
    else:
        ccf_df = ccf_results.copy()

    if len(ccf_df) == 0:
        raise ValueError("Empty cross-correlation results")

    lags = ccf_df['lag'].values
    correlations = ccf_df['correlation'].values
    n_obs = ccf_df['n_observations'].iloc[0] if 'n_observations' in ccf_df.columns else 30

    # Calculate 95% confidence bounds (Bartlett approximation)
    # Under H0: rho=0, se(r) ~ 1/sqrt(n)
    from scipy import stats
    z_critical = stats.norm.ppf(1 - alpha / 2)
    conf_bound = z_critical / np.sqrt(n_obs)

    # Determine significance
    significant = np.abs(correlations) > conf_bound

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar colors based on significance
    colors = [COLORS["significant"] if sig else COLORS["not_significant"]
              for sig in significant]

    # Bar plot
    bars = ax.bar(lags, correlations, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Confidence bands
    ax.axhline(y=conf_bound, color=COLORS["confidence"], linestyle='--',
               linewidth=1.5, label=f'{int((1-alpha)*100)}% Confidence Bound')
    ax.axhline(y=-conf_bound, color=COLORS["confidence"], linestyle='--', linewidth=1.5)

    # Zero line
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Zero lag marker
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    # Annotations for significant lags
    for i, (lag, corr, sig) in enumerate(zip(lags, correlations, significant)):
        if sig:
            ax.annotate(f'{corr:.2f}',
                        xy=(lag, corr),
                        xytext=(0, 5 if corr > 0 else -12),
                        textcoords='offset points',
                        ha='center', fontsize=8, fontweight='bold')

    # Labels and title
    ax.set_xlabel("Lag (months, positive = CPU leads)")
    ax.set_ylabel("Correlation")
    ax.set_title(title)
    ax.legend(loc='upper right')

    # Set x-ticks to integers
    ax.set_xticks(lags)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return _save_or_return(fig, output_path)


def plot_rolling_correlation(
    cpu_series: pd.Series,
    vc_series: pd.Series,
    window: int = 12,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Rolling Correlation: CPU vs VC",
) -> Union[plt.Figure, Path]:
    """
    Create rolling correlation plot with confidence interval.

    Args:
        cpu_series: CPU index values with datetime index
        vc_series: VC deal count values with datetime index
        window: Rolling window size in months (default: 12)
        output_path: Optional path to save figure. If None, returns Figure.
        title: Chart title

    Returns:
        matplotlib Figure if output_path is None, else Path to saved file

    Notes:
        - Rolling correlation computed over specified window
        - Shaded area shows approximate 95% confidence interval
        - Confidence interval approximated using Fisher z-transformation
    """
    _apply_style()

    # Ensure datetime indices
    cpu_series = _ensure_datetime_index(cpu_series)
    vc_series = _ensure_datetime_index(vc_series)

    # Align series
    combined = pd.DataFrame({
        'cpu': cpu_series,
        'vc': vc_series,
    }).dropna()

    if len(combined) < window:
        raise ValueError(f"Insufficient data for window={window}. Have {len(combined)} observations.")

    # Compute rolling correlation
    rolling_corr = combined['cpu'].rolling(window=window).corr(combined['vc'])

    # Approximate confidence interval using Fisher z-transformation
    # SE of Fisher z ~ 1/sqrt(n-3), then transform back
    n = window
    if n > 3:
        se_z = 1 / np.sqrt(n - 3)
        z_vals = np.arctanh(rolling_corr.clip(-0.999, 0.999))  # Fisher z
        z_lower = z_vals - 1.96 * se_z
        z_upper = z_vals + 1.96 * se_z
        ci_lower = np.tanh(z_lower)
        ci_upper = np.tanh(z_upper)
    else:
        ci_lower = ci_upper = rolling_corr

    dates = combined.index

    fig, ax = plt.subplots(figsize=(12, 6))

    # Rolling correlation line
    ax.plot(dates, rolling_corr, color=COLORS["cpu"], linewidth=2, label=f'{window}-month Rolling Correlation')

    # Confidence interval
    ax.fill_between(dates, ci_lower, ci_upper, alpha=0.2, color=COLORS["cpu"],
                    label='95% Confidence Interval')

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Significance thresholds
    ax.axhline(y=0.3, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=-0.3, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.set_title(title)
    ax.set_ylim(-1, 1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.legend(loc='upper right')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return _save_or_return(fig, output_path)


def plot_stage_distribution(
    monthly_data: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "VC Deal Stage Distribution Over Time",
) -> Union[plt.Figure, Path]:
    """
    Create stacked area chart of seed/early/late stage deals over time.

    Args:
        monthly_data: DataFrame with columns 'seed_count', 'early_count', 'late_count'
                      and datetime index (from vc_aggregator.aggregate_monthly)
        output_path: Optional path to save figure. If None, returns Figure.
        title: Chart title

    Returns:
        matplotlib Figure if output_path is None, else Path to saved file

    Notes:
        - Shows composition of VC deals by funding stage
        - Y-axis shows deal counts
    """
    _apply_style()

    # Validate required columns
    required_cols = ['seed_count', 'early_count', 'late_count']
    missing = [c for c in required_cols if c not in monthly_data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure datetime index
    df = monthly_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if hasattr(df.index, 'to_timestamp'):
            df.index = df.index.to_timestamp()
        else:
            df.index = pd.to_datetime(df.index)

    dates = df.index
    seed = df['seed_count'].values
    early = df['early_count'].values
    late = df['late_count'].values

    fig, ax = plt.subplots(figsize=(12, 6))

    # Stacked area chart
    ax.fill_between(dates, 0, seed, alpha=0.8, color=COLORS["seed"], label='Seed')
    ax.fill_between(dates, seed, seed + early, alpha=0.8, color=COLORS["early"], label='Early')
    ax.fill_between(dates, seed + early, seed + early + late, alpha=0.8, color=COLORS["late"], label='Late')

    # Labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Deals")
    ax.set_title(title)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.legend(loc='upper left')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    return _save_or_return(fig, output_path)


# =============================================================================
# BATCH EXPORT
# =============================================================================

def save_all_visualizations(
    cpu_series: pd.Series,
    vc_monthly: pd.DataFrame,
    ccf_results: Union[pd.DataFrame, dict],
    output_dir: Union[str, Path],
    window: int = 12,
) -> dict[str, Path]:
    """
    Generate and save all CPU-VC correlation visualizations.

    Args:
        cpu_series: CPU index values with datetime index
        vc_monthly: Monthly VC metrics from vc_aggregator.aggregate_monthly()
        ccf_results: Cross-correlation results from correlation.cross_correlation()
                     or dict from analyze_cpu_vc_correlation()
        output_dir: Directory to save figures
        window: Rolling correlation window size (default: 12)

    Returns:
        Dict mapping figure names to output paths

    Example:
        >>> paths = save_all_visualizations(cpu, vc_monthly, ccf, './figures')
        >>> print(paths['timeseries'])  # ./figures/cpu_vc_timeseries.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Extract VC deal count from monthly data
    vc_col = 'deal_count' if 'deal_count' in vc_monthly.columns else vc_monthly.columns[0]
    vc_series = vc_monthly[vc_col]

    # 1. Time series plot
    try:
        paths['timeseries'] = plot_cpu_vc_timeseries(
            cpu_series, vc_series,
            output_path=output_dir / "cpu_vc_timeseries.png"
        )
    except Exception as e:
        paths['timeseries'] = f"Error: {e}"

    # 2. Cross-correlation plot
    try:
        paths['cross_correlation'] = plot_cross_correlation(
            ccf_results,
            output_path=output_dir / "cpu_vc_ccf.png"
        )
    except Exception as e:
        paths['cross_correlation'] = f"Error: {e}"

    # 3. Rolling correlation plot
    try:
        paths['rolling_correlation'] = plot_rolling_correlation(
            cpu_series, vc_series,
            window=window,
            output_path=output_dir / "cpu_vc_rolling_corr.png"
        )
    except Exception as e:
        paths['rolling_correlation'] = f"Error: {e}"

    # 4. Stage distribution plot
    try:
        paths['stage_distribution'] = plot_stage_distribution(
            vc_monthly,
            output_path=output_dir / "vc_stage_distribution.png"
        )
    except Exception as e:
        paths['stage_distribution'] = f"Error: {e}"

    return paths


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=== VC Visualizations Module Test ===")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2020-01", periods=48, freq="MS")

    # Sample CPU index
    cpu = pd.Series(
        100 + np.cumsum(np.random.randn(48) * 5),
        index=dates,
        name="cpu_index"
    )

    # Sample VC data
    vc_monthly = pd.DataFrame({
        'deal_count': np.random.poisson(50, 48),
        'seed_count': np.random.poisson(20, 48),
        'early_count': np.random.poisson(20, 48),
        'late_count': np.random.poisson(10, 48),
    }, index=dates)

    # Sample CCF results
    lags = list(range(-12, 13))
    correlations = [np.random.uniform(-0.5, 0.5) for _ in lags]
    ccf_results = pd.DataFrame({
        'lag': lags,
        'correlation': correlations,
        'n_observations': [40] * len(lags),
        'interpretation': ['test'] * len(lags),
    })

    output_dir = Path("outputs/test_vc_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving test figures to {output_dir}")

    # Test individual functions
    print("  - Testing plot_cpu_vc_timeseries...")
    fig = plot_cpu_vc_timeseries(cpu, vc_monthly['deal_count'],
                                  output_path=output_dir / "test_timeseries.png")
    print(f"    Saved: {fig}")

    print("  - Testing plot_cross_correlation...")
    fig = plot_cross_correlation(ccf_results,
                                  output_path=output_dir / "test_ccf.png")
    print(f"    Saved: {fig}")

    print("  - Testing plot_rolling_correlation...")
    fig = plot_rolling_correlation(cpu, vc_monthly['deal_count'],
                                    window=12,
                                    output_path=output_dir / "test_rolling.png")
    print(f"    Saved: {fig}")

    print("  - Testing plot_stage_distribution...")
    fig = plot_stage_distribution(vc_monthly,
                                   output_path=output_dir / "test_stages.png")
    print(f"    Saved: {fig}")

    print("  - Testing save_all_visualizations...")
    paths = save_all_visualizations(cpu, vc_monthly, ccf_results, output_dir / "all")
    for name, path in paths.items():
        print(f"    {name}: {path}")

    print("\nAll tests completed successfully!")
