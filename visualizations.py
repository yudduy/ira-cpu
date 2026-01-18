"""
Visualization Functions for CPU Index Deliverables

Creates publication-quality charts following BBD (2016) standards.
All functions export to PNG at 300 DPI with consistent academic styling.

Main Text Figures (REQ-2):
- Figure 1: CPU Time Series with Event Annotations
- Figure 2: CPU Decomposition (4-panel)
- Figure 3: Direction Balance Metric

Appendix Figures (REQ-3):
- Figure A1: Outlet Correlation Heatmap
- Figure A2: Keyword Sensitivity Forest Plot
- Figure A3: Placebo Comparison
- Figure A4: LLM vs Keyword Scatter
- Figure A5: Article Volume Baseline
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

# Consistent styling for all figures
STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "lines.linewidth": 1.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Headline events for annotations
HEADLINE_EVENTS = [
    {"date": "2022-08", "label": "IRA Signed", "color": "green"},
    {"date": "2024-11", "label": "Trump Election", "color": "red"},
    {"date": "2025-02", "label": "OBBBA Introduced", "color": "orange"},
]


def _apply_style():
    """Apply consistent styling to matplotlib."""
    plt.rcParams.update(STYLE_CONFIG)


def _month_to_date(month_str: str) -> datetime:
    """Convert YYYY-MM to datetime."""
    return datetime.strptime(month_str, "%Y-%m")


def _prepare_timeseries(data: dict, key: str = "cpu") -> tuple[list, list]:
    """Extract sorted dates and values from index data."""
    sorted_months = sorted(data.keys())
    dates = [_month_to_date(m) for m in sorted_months]
    values = [data[m].get(key, 0) if isinstance(data[m], dict) else data[m] for m in sorted_months]
    return dates, values


# =============================================================================
# MAIN TEXT FIGURES
# =============================================================================

def plot_cpu_timeseries(
    index_data: dict,
    events: list[dict],
    output_path: Union[str, Path],
    title: str = "Climate Policy Uncertainty Index",
) -> Path:
    """
    Figure 1: CPU Time Series with Event Annotations.

    Args:
        index_data: Dict mapping months to {cpu: value}
        events: List of event dicts with date, label, color
        output_path: Path for output PNG
        title: Chart title

    Returns:
        Path to created file
    """
    _apply_style()
    output_path = Path(output_path)

    dates, values = _prepare_timeseries(index_data, "cpu")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(dates, values, color="#1f77b4", linewidth=2, label="CPU Index")
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    for event in events:
        event_date = _month_to_date(event["date"])
        ax.axvline(
            x=event_date,
            color=event.get("color", "gray"),
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )
        ax.annotate(
            event["label"],
            xy=(event_date, ax.get_ylim()[1] * 0.95),
            ha="center",
            fontsize=9,
            color=event.get("color", "gray"),
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("CPU Index (Base Period = 100)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_cpu_decomposition(
    index_data: dict,
    output_path: Union[str, Path],
    title: str = "CPU Index Decomposition",
) -> Path:
    """
    Figure 2: CPU Decomposition (4-panel faceted).

    Panels:
    1. Total CPU
    2. CPU_impl (implementation uncertainty)
    3. CPU_reversal (rollback risk)
    4. Percentage contribution

    Args:
        index_data: Dict mapping months to {cpu, cpu_impl, cpu_reversal}
        output_path: Path for output PNG
        title: Overall title

    Returns:
        Path to created file
    """
    _apply_style()
    output_path = Path(output_path)

    dates, cpu_values = _prepare_timeseries(index_data, "cpu")
    _, impl_values = _prepare_timeseries(index_data, "cpu_impl")
    _, reversal_values = _prepare_timeseries(index_data, "cpu_reversal")

    # Handle missing data
    impl_values = [v if v else 0 for v in impl_values]
    reversal_values = [v if v else 0 for v in reversal_values]

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Panel 1: Total CPU
    axes[0].plot(dates, cpu_values, color="#1f77b4", linewidth=2)
    axes[0].set_ylabel("CPU")
    axes[0].set_title("Total CPU Index")
    axes[0].axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Panel 2: CPU_impl
    axes[1].plot(dates, impl_values, color="#2ca02c", linewidth=2)
    axes[1].set_ylabel("CPU_impl")
    axes[1].set_title("Implementation Uncertainty")
    axes[1].axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Panel 3: CPU_reversal
    axes[2].plot(dates, reversal_values, color="#d62728", linewidth=2)
    axes[2].set_ylabel("CPU_reversal")
    axes[2].set_title("Reversal Risk")
    axes[2].axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Panel 4: Percentage contribution
    total = np.array(impl_values) + np.array(reversal_values)
    # Avoid division by zero
    total = np.where(total == 0, 1, total)
    impl_pct = np.array(impl_values) / total * 100
    reversal_pct = np.array(reversal_values) / total * 100

    axes[3].fill_between(dates, 0, impl_pct, alpha=0.7, color="#2ca02c", label="Implementation")
    axes[3].fill_between(dates, impl_pct, 100, alpha=0.7, color="#d62728", label="Reversal")
    axes[3].set_ylabel("% of Total")
    axes[3].set_title("Composition")
    axes[3].set_ylim(0, 100)
    axes[3].legend(loc="upper right")

    # Shared x-axis formatting
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_direction_balance(
    index_data: dict,
    output_path: Union[str, Path],
    title: str = "Direction Balance: Implementation vs Reversal",
) -> Path:
    """
    Figure 3: Direction Balance Metric.

    Balance = (CPU_impl - CPU_reversal) / (CPU_impl + CPU_reversal)
    Range: -1 (pure reversal) to +1 (pure implementation)

    Args:
        index_data: Dict mapping months to {cpu_impl, cpu_reversal}
        output_path: Path for output PNG
        title: Chart title

    Returns:
        Path to created file
    """
    _apply_style()
    output_path = Path(output_path)

    dates, impl_values = _prepare_timeseries(index_data, "cpu_impl")
    _, reversal_values = _prepare_timeseries(index_data, "cpu_reversal")

    # Calculate balance metric
    balance = []
    for impl, rev in zip(impl_values, reversal_values):
        impl = impl or 0
        rev = rev or 0
        total = impl + rev
        if total > 0:
            balance.append((impl - rev) / total)
        else:
            balance.append(0)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot balance line
    ax.plot(dates, balance, color="#1f77b4", linewidth=2)

    # Zero line (equal balance)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

    # Shaded regions
    ax.fill_between(dates, 0, balance, where=[b > 0 for b in balance],
                    alpha=0.3, color="#2ca02c", label="Implementation-dominated")
    ax.fill_between(dates, 0, balance, where=[b < 0 for b in balance],
                    alpha=0.3, color="#d62728", label="Reversal-dominated")

    # Formatting
    ax.set_xlabel("Date")
    ax.set_ylabel("Direction Balance")
    ax.set_title(title)
    ax.set_ylim(-1, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    # Add labels for regions
    ax.text(0.98, 0.95, "+1 = Pure Implementation", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#2ca02c")
    ax.text(0.98, 0.05, "-1 = Pure Reversal", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#d62728")

    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


# =============================================================================
# APPENDIX FIGURES
# =============================================================================

def plot_outlet_correlation_heatmap(
    outlet_indices: dict,
    output_path: Union[str, Path],
    title: str = "Outlet-Level CPU Correlation",
) -> Path:
    """
    Figure A1: Outlet Correlation Heatmap.

    Args:
        outlet_indices: Dict mapping outlet names to {month: cpu_value}
        output_path: Path for output PNG
        title: Chart title

    Returns:
        Path to created file
    """
    _apply_style()
    output_path = Path(output_path)

    outlets = list(outlet_indices.keys())
    n_outlets = len(outlets)

    fig, ax = plt.subplots(figsize=(10, 8))

    if n_outlets < 2:
        # Not enough outlets for correlation
        ax.text(0.5, 0.5, "Insufficient outlets for correlation\n(need at least 2)",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_title(title)
    else:
        # Align series to common months
        all_months = set()
        for series in outlet_indices.values():
            all_months.update(series.keys())
        sorted_months = sorted(all_months)

        # Build matrix
        data_matrix = []
        for outlet in outlets:
            series = outlet_indices[outlet]
            row = [series.get(m, np.nan) for m in sorted_months]
            data_matrix.append(row)
        data_matrix = np.array(data_matrix)

        # Compute correlation matrix
        corr_matrix = np.zeros((n_outlets, n_outlets))
        for i in range(n_outlets):
            for j in range(n_outlets):
                mask = ~np.isnan(data_matrix[i]) & ~np.isnan(data_matrix[j])
                if mask.sum() >= 2:
                    corr_matrix[i, j] = np.corrcoef(
                        data_matrix[i][mask], data_matrix[j][mask]
                    )[0, 1]
                else:
                    corr_matrix[i, j] = np.nan

        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap="RdYlBu", vmin=-1, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation")

        # Labels
        ax.set_xticks(range(n_outlets))
        ax.set_yticks(range(n_outlets))
        ax.set_xticklabels(outlets, rotation=45, ha="right")
        ax.set_yticklabels(outlets)

        # Add correlation values as text
        for i in range(n_outlets):
            for j in range(n_outlets):
                val = corr_matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if abs(val) > 0.5 else "black", fontsize=9)

        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_keyword_sensitivity(
    sensitivity_results: dict,
    output_path: Union[str, Path],
    title: str = "Keyword Sensitivity Analysis",
    threshold: float = 0.95,
) -> Path:
    """
    Figure A2: Keyword Sensitivity Forest Plot.

    Args:
        sensitivity_results: Dict mapping variant name to correlation with baseline
        output_path: Path for output PNG
        title: Chart title
        threshold: Threshold line (default 0.95)

    Returns:
        Path to created file
    """
    _apply_style()
    output_path = Path(output_path)

    # Sort by correlation
    sorted_items = sorted(sensitivity_results.items(), key=lambda x: x[1])
    names = [item[0] for item in sorted_items]
    correlations = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.3)))

    # Horizontal bar chart
    y_pos = range(len(names))
    colors = ["#d62728" if c < threshold else "#2ca02c" for c in correlations]
    ax.barh(y_pos, correlations, color=colors, alpha=0.8)

    # Threshold line
    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold ({threshold})")

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Correlation with Baseline")
    ax.set_title(title)
    ax.set_xlim(0.8, 1.0)  # Focus on high correlations
    ax.legend(loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_placebo_comparison(
    cpu: dict,
    tpu: dict,
    mpu: dict,
    output_path: Union[str, Path],
    title: str = "CPU vs Placebo Indices",
) -> Path:
    """
    Figure A3: Placebo Comparison.

    Args:
        cpu: Climate Policy Uncertainty {month: value}
        tpu: Trade Policy Uncertainty {month: value}
        mpu: Monetary Policy Uncertainty {month: value}
        output_path: Path for output PNG
        title: Chart title

    Returns:
        Path to created file
    """
    _apply_style()
    output_path = Path(output_path)

    # Prepare series
    cpu_dates = [_month_to_date(m) for m in sorted(cpu.keys())]
    cpu_values = [cpu[m] for m in sorted(cpu.keys())]

    tpu_dates = [_month_to_date(m) for m in sorted(tpu.keys())]
    tpu_values = [tpu[m] for m in sorted(tpu.keys())]

    mpu_dates = [_month_to_date(m) for m in sorted(mpu.keys())]
    mpu_values = [mpu[m] for m in sorted(mpu.keys())]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(cpu_dates, cpu_values, color="#1f77b4", linewidth=2, label="CPU (Climate)")
    ax.plot(tpu_dates, tpu_values, color="#ff7f0e", linewidth=2, label="TPU (Trade)")
    ax.plot(mpu_dates, mpu_values, color="#2ca02c", linewidth=2, label="MPU (Monetary)")

    # Reference line
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Formatting
    ax.set_xlabel("Date")
    ax.set_ylabel("Index Value (Base = 100)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.legend(loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_llm_validation_scatter(
    keyword_cpu: dict,
    llm_cpu: dict,
    output_path: Union[str, Path],
    title: str = "Keyword vs LLM-Validated CPU",
) -> Path:
    """
    Figure A4: LLM vs Keyword Scatter.

    Args:
        keyword_cpu: Keyword-only CPU {month: value}
        llm_cpu: LLM-validated CPU {month: value}
        output_path: Path for output PNG
        title: Chart title

    Returns:
        Path to created file
    """
    _apply_style()
    output_path = Path(output_path)

    # Find common months
    common_months = sorted(set(keyword_cpu.keys()) & set(llm_cpu.keys()))
    x = [keyword_cpu[m] for m in common_months]
    y = [llm_cpu[m] for m in common_months]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(x, y, color="#1f77b4", alpha=0.7, s=50)

    # 45-degree reference line
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    ax.plot([min_val, max_val], [min_val, max_val], color="gray",
            linestyle="--", linewidth=1, label="Perfect agreement")

    # Compute correlation
    if len(x) >= 2:
        corr = np.corrcoef(x, y)[0, 1]
        ax.annotate(f"r = {corr:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                    fontsize=12, ha="left", va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Formatting
    ax.set_xlabel("Keyword-Only CPU")
    ax.set_ylabel("LLM-Validated CPU")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_aspect("equal")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_article_volume(
    monthly_counts: dict,
    output_path: Union[str, Path],
    title: str = "Monthly Article Volume (Denominator)",
) -> Path:
    """
    Figure A5: Article Volume Baseline.

    Args:
        monthly_counts: Dict mapping months to article counts
        output_path: Path for output PNG
        title: Chart title

    Returns:
        Path to created file
    """
    _apply_style()
    output_path = Path(output_path)

    dates = [_month_to_date(m) for m in sorted(monthly_counts.keys())]
    values = [monthly_counts[m] for m in sorted(monthly_counts.keys())]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Bar chart
    ax.bar(dates, values, width=20, color="#1f77b4", alpha=0.8)

    # Mean line
    mean_val = np.mean(values)
    ax.axhline(y=mean_val, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean ({mean_val:.0f})")

    # Formatting
    ax.set_xlabel("Date")
    ax.set_ylabel("Article Count")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.legend(loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


# =============================================================================
# UTILITIES
# =============================================================================

def generate_all_figures(
    index_data: dict,
    outlet_indices: dict,
    sensitivity_results: dict,
    placebo_data: Optional[dict] = None,
    llm_data: Optional[dict] = None,
    output_dir: Union[str, Path] = "outputs/figures",
) -> list[Path]:
    """
    Generate all figures in one call.

    Args:
        index_data: Main index data with all components
        outlet_indices: Outlet-level CPU indices
        sensitivity_results: Keyword sensitivity results
        placebo_data: Optional dict with "tpu" and "mpu" series
        llm_data: Optional dict with "keyword_cpu" and "llm_cpu" series
        output_dir: Output directory

    Returns:
        List of paths to created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []

    # Main text figures
    paths.append(plot_cpu_timeseries(
        index_data, HEADLINE_EVENTS, output_dir / "fig1_cpu_timeseries.png"
    ))
    paths.append(plot_cpu_decomposition(
        index_data, output_dir / "fig2_cpu_decomposition.png"
    ))
    paths.append(plot_direction_balance(
        index_data, output_dir / "fig3_direction_balance.png"
    ))

    # Appendix figures
    paths.append(plot_outlet_correlation_heatmap(
        outlet_indices, output_dir / "figA1_outlet_heatmap.png"
    ))
    paths.append(plot_keyword_sensitivity(
        sensitivity_results, output_dir / "figA2_keyword_sensitivity.png"
    ))

    # Optional figures
    if placebo_data:
        cpu_series = {m: d.get("cpu", 0) for m, d in index_data.items()}
        paths.append(plot_placebo_comparison(
            cpu_series,
            placebo_data.get("tpu", {}),
            placebo_data.get("mpu", {}),
            output_dir / "figA3_placebo_comparison.png"
        ))

    if llm_data:
        paths.append(plot_llm_validation_scatter(
            llm_data.get("keyword_cpu", {}),
            llm_data.get("llm_cpu", {}),
            output_dir / "figA4_llm_scatter.png"
        ))

    # Article volume
    volume_data = {m: d.get("denominator", 0) for m, d in index_data.items()}
    paths.append(plot_article_volume(
        volume_data, output_dir / "figA5_article_volume.png"
    ))

    return paths


if __name__ == "__main__":
    # Example usage with sample data
    sample_index = {
        "2021-01": {"cpu": 100.0, "cpu_impl": 45.0, "cpu_reversal": 55.0, "denominator": 1250},
        "2021-06": {"cpu": 105.0, "cpu_impl": 50.0, "cpu_reversal": 55.0, "denominator": 1300},
        "2022-01": {"cpu": 110.0, "cpu_impl": 55.0, "cpu_reversal": 55.0, "denominator": 1400},
        "2022-08": {"cpu": 95.0, "cpu_impl": 60.0, "cpu_reversal": 35.0, "denominator": 1500},
        "2023-01": {"cpu": 115.0, "cpu_impl": 65.0, "cpu_reversal": 50.0, "denominator": 1450},
        "2024-01": {"cpu": 125.0, "cpu_impl": 55.0, "cpu_reversal": 70.0, "denominator": 1600},
        "2024-11": {"cpu": 150.0, "cpu_impl": 50.0, "cpu_reversal": 100.0, "denominator": 1800},
        "2025-01": {"cpu": 160.0, "cpu_impl": 55.0, "cpu_reversal": 105.0, "denominator": 1900},
    }

    output_dir = Path("outputs/sample_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating sample figures...")
    plot_cpu_timeseries(sample_index, HEADLINE_EVENTS, output_dir / "cpu_timeseries.png")
    plot_cpu_decomposition(sample_index, output_dir / "cpu_decomposition.png")
    plot_direction_balance(sample_index, output_dir / "direction_balance.png")
    print(f"Sample figures saved to {output_dir}")
