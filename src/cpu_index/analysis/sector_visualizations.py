"""
Sector-Specific CPU-VC Correlation Visualizations

Creates publication-quality charts for sector-level CPU-VC correlation analysis.
Follows project styling conventions from vc_visualizations.py.

Functions:
- plot_sector_correlation_heatmap: Correlation heatmap across sectors and lags
- plot_sector_timeseries: Dual-axis time series for multiple sectors
- plot_sensitivity_ranking: Ranked bar chart of sector CPU sensitivity
- save_sector_visualizations: Generate all sector plots to output directory
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Optional seaborn for heatmaps
try:
    import seaborn as sns
    _SEABORN_AVAILABLE = True
except ImportError:
    _SEABORN_AVAILABLE = False


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

STYLE_CONFIG = {
    "figure.figsize": (12, 8),
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

# Color palette for sectors
SECTOR_COLORS = {
    "Energy": "#1f77b4",
    "Industrial": "#ff7f0e",
    "Built_Environment": "#2ca02c",
    "Food_Land_Use": "#d62728",
    "Transportation": "#9467bd",
    "Climate_Mgmt": "#8c564b",
    "Carbon": "#e377c2",
    "Others": "#7f7f7f",
}

# Correlation colormap
CORR_CMAP = "RdBu_r"  # Red (negative) to Blue (positive)


def _apply_style():
    """Apply consistent matplotlib styling."""
    plt.rcParams.update(STYLE_CONFIG)
    if _SEABORN_AVAILABLE:
        sns.set_style("whitegrid", {"axes.edgecolor": ".3"})


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

def plot_sector_correlation_heatmap(
    sector_results: dict,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "CPU-VC Cross-Correlation by Sector",
) -> Union[plt.Figure, Path]:
    """
    Create correlation heatmap showing sectors × lags.

    Args:
        sector_results: Dict with sector names as keys, each containing
                       'ccf' DataFrame from analyze_sector_cpu_correlation()
        output_path: Path to save figure (or None to return Figure)
        title: Plot title

    Returns:
        Figure object or Path to saved file
    """
    _apply_style()

    # Extract correlation data at each lag for each sector
    lags = None
    data = {}
    for sector, result in sector_results.items():
        if result.get('ccf') is not None:
            ccf = result['ccf']
            if lags is None:
                lags = ccf['lag'].values
            data[sector] = ccf.set_index('lag')['correlation'].values

    if not data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No valid correlation data", ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        return _save_or_return(fig, output_path)

    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(data, index=lags).T
    heatmap_df = heatmap_df.sort_index()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    if _SEABORN_AVAILABLE:
        sns.heatmap(
            heatmap_df,
            cmap=CORR_CMAP,
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            ax=ax,
            cbar_kws={"label": "Correlation"},
        )
    else:
        im = ax.imshow(heatmap_df.values, cmap=CORR_CMAP, vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(heatmap_df.columns)))
        ax.set_xticklabels(heatmap_df.columns)
        ax.set_yticks(range(len(heatmap_df.index)))
        ax.set_yticklabels(heatmap_df.index)
        plt.colorbar(im, ax=ax, label="Correlation")

    ax.set_xlabel("Lag (months, positive = CPU leads)")
    ax.set_ylabel("Sector")
    ax.set_title(title)

    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_sector_timeseries(
    cpu_df: pd.DataFrame,
    sector_vc_monthly: dict,
    top_n: int = 4,
    cpu_column: str = 'cpu_index',
    output_path: Optional[Union[str, Path]] = None,
    title: str = "CPU Index vs Sector VC Activity",
) -> Union[plt.Figure, Path]:
    """
    Create dual-axis time series plots for top N most sensitive sectors.

    Args:
        cpu_df: CPU index DataFrame
        sector_vc_monthly: Dict with sector names as keys, monthly VC Series as values
        top_n: Number of sectors to show (default: 4 for 2x2 grid)
        cpu_column: Column name for CPU values
        output_path: Path to save figure
        title: Main title

    Returns:
        Figure object or Path to saved file
    """
    _apply_style()

    # Prepare CPU series
    cpu = cpu_df.copy()
    if 'month' in cpu.columns:
        cpu['month'] = pd.to_datetime(cpu['month'])
        cpu = cpu.set_index('month')
    cpu_series = cpu[cpu_column]

    # Get top N sectors by data availability
    sectors = list(sector_vc_monthly.keys())[:top_n]

    # Create subplots
    n_rows = (len(sectors) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten() if n_rows > 1 else [axes] if len(sectors) == 1 else axes.flatten()

    for idx, sector in enumerate(sectors):
        ax1 = axes[idx]
        vc_series = sector_vc_monthly[sector]

        # Ensure datetime index
        if not isinstance(vc_series.index, pd.DatetimeIndex):
            vc_series.index = pd.to_datetime(vc_series.index)

        # Plot CPU on left axis
        color1 = "#1f77b4"
        ax1.plot(cpu_series.index, cpu_series.values, color=color1, linewidth=1.5, label='CPU')
        ax1.set_ylabel('CPU Index', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        # Plot VC on right axis
        ax2 = ax1.twinx()
        color2 = SECTOR_COLORS.get(sector, "#ff7f0e")
        ax2.plot(vc_series.index, vc_series.values, color=color2, linewidth=1.5, alpha=0.8, label='VC Deals')
        ax2.set_ylabel('Deal Count', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.fill_between(vc_series.index, 0, vc_series.values, alpha=0.2, color=color2)

        ax1.set_title(sector)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_major_locator(mdates.YearLocator())

    # Hide unused subplots
    for idx in range(len(sectors), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_sensitivity_ranking(
    ranking_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Sector CPU Sensitivity Ranking",
    top_n: Optional[int] = None,
) -> Union[plt.Figure, Path]:
    """
    Create ranked bar chart of sector CPU sensitivity.

    Args:
        ranking_df: DataFrame from analyze_all_sectors() with columns:
                   sector, correlation, abs_correlation, direction
        output_path: Path to save figure
        title: Plot title
        top_n: Limit to top N sectors (default: all)

    Returns:
        Figure object or Path to saved file
    """
    _apply_style()

    df = ranking_df.copy()
    if top_n is not None:
        df = df.head(top_n)

    # Sort by absolute correlation
    df = df.sort_values('abs_correlation', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.5)))

    # Create horizontal bar chart
    colors = []
    for _, row in df.iterrows():
        if pd.isna(row['direction']):
            colors.append('#7f7f7f')  # Gray for NA
        elif row['direction'] == 'positive':
            colors.append('#2ca02c')  # Green for positive
        else:
            colors.append('#d62728')  # Red for negative

    bars = ax.barh(df['sector'], df['abs_correlation'].fillna(0), color=colors)

    # Add correlation values as labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        if pd.notna(row['correlation']):
            label = f"{row['correlation']:.3f}"
            x_pos = bar.get_width() + 0.01
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, label,
                   va='center', fontsize=10)

    ax.set_xlabel('Absolute Correlation at Optimal Lag')
    ax.set_ylabel('Sector')
    ax.set_title(title)
    ax.set_xlim(0, min(1.0, df['abs_correlation'].max() * 1.2) if df['abs_correlation'].notna().any() else 1.0)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Positive correlation'),
        Patch(facecolor='#d62728', label='Negative correlation'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    return _save_or_return(fig, output_path)


def plot_decomposition_comparison(
    decomposition_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "CPU Decomposition: Implementation vs Reversal Sensitivity",
) -> Union[plt.Figure, Path]:
    """
    Create comparison plot showing impl vs reversal correlation by sector.

    Args:
        decomposition_df: DataFrame from analyze_cpu_decomposition() with columns:
                         sector, impl_correlation, reversal_correlation, asymmetry_ratio
        output_path: Path to save figure
        title: Plot title

    Returns:
        Figure object or Path to saved file
    """
    _apply_style()

    df = decomposition_df.dropna(subset=['impl_correlation', 'reversal_correlation']).copy()
    df = df.sort_values('asymmetry_ratio')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Scatter plot of impl vs reversal
    ax1 = axes[0]
    for _, row in df.iterrows():
        color = SECTOR_COLORS.get(row['sector'], '#7f7f7f')
        ax1.scatter(row['impl_correlation'], row['reversal_correlation'],
                   s=100, c=color, label=row['sector'], alpha=0.8)

    # Add diagonal line
    lims = [-1, 1]
    ax1.plot(lims, lims, 'k--', alpha=0.3, label='Equal sensitivity')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel('Implementation Uncertainty Correlation')
    ax1.set_ylabel('Reversal Uncertainty Correlation')
    ax1.set_title('Implementation vs Reversal Sensitivity')
    ax1.legend(loc='best', fontsize=8)
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(0, color='gray', linestyle='-', alpha=0.3)

    # Right panel: Asymmetry ratio bar chart
    ax2 = axes[1]
    colors = ['#2ca02c' if r > 0 else '#d62728' for r in df['asymmetry_ratio']]
    ax2.barh(df['sector'], df['asymmetry_ratio'], color=colors)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.set_xlabel('Asymmetry Ratio\n(+ = more impl-sensitive, - = more reversal-sensitive)')
    ax2.set_ylabel('Sector')
    ax2.set_title('CPU Type Asymmetry')
    ax2.set_xlim(-1, 1)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return _save_or_return(fig, output_path)


def save_sector_visualizations(
    sector_results: dict,
    ranking_df: pd.DataFrame,
    decomposition_df: pd.DataFrame,
    cpu_df: pd.DataFrame,
    sector_vc_monthly: dict,
    output_dir: Union[str, Path],
    cpu_column: str = 'cpu_index',
) -> dict:
    """
    Generate all sector visualization plots and save to output directory.

    Args:
        sector_results: Dict from running analyze_sector_cpu_correlation on each sector
        ranking_df: DataFrame from analyze_all_sectors()
        decomposition_df: DataFrame from analyze_cpu_decomposition()
        cpu_df: CPU index DataFrame
        sector_vc_monthly: Dict of sector monthly VC Series
        output_dir: Directory to save figures
        cpu_column: Column name for CPU values

    Returns:
        Dict mapping figure names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # 1. Correlation heatmap
    path = output_dir / "fig_sector_heatmap.png"
    plot_sector_correlation_heatmap(sector_results, output_path=path)
    saved_files['heatmap'] = path

    # 2. Time series
    path = output_dir / "fig_sector_timeseries.png"
    plot_sector_timeseries(cpu_df, sector_vc_monthly, cpu_column=cpu_column, output_path=path)
    saved_files['timeseries'] = path

    # 3. Sensitivity ranking
    path = output_dir / "fig_sensitivity_ranking.png"
    plot_sensitivity_ranking(ranking_df, output_path=path)
    saved_files['ranking'] = path

    # 4. Decomposition comparison
    if not decomposition_df.empty:
        path = output_dir / "fig_decomposition_comparison.png"
        plot_decomposition_comparison(decomposition_df, output_path=path)
        saved_files['decomposition'] = path

    return saved_files
