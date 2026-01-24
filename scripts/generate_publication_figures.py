#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for CPU-VC Analysis

Creates enhanced visualizations with policy event annotations for the paper.
"""

import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =============================================================================
# KEY POLICY EVENTS
# =============================================================================

POLICY_EVENTS = [
    # Format: (date, label, event_type)
    # event_type: 'positive' (green), 'negative' (red), 'neutral' (gray)

    # Elections (gray vertical lines)
    ("2008-11-04", "Obama\nElected", "election"),
    ("2012-11-06", "Obama\nRe-elected", "election"),
    ("2016-11-08", "Trump\nElected", "election"),
    ("2020-11-03", "Biden\nElected", "election"),
    ("2024-11-05", "Trump\nRe-elected", "election"),

    # Major Climate Policies - SUPPORTIVE
    ("2009-02-17", "ARRA\n($90B)", "positive"),
    ("2015-12-12", "Paris\nAgreement", "positive"),
    ("2022-08-16", "IRA\n($369B)", "positive"),

    # Major Climate Policies - REVERSAL RISK
    ("2017-06-01", "Paris\nWithdrawal", "negative"),
    ("2025-01-20", "OBBBA\n(IRA Repeal)", "negative"),  # Trump 2.0 + OBBBA reconciliation

    # Political Transitions (shaded regions)
    ("2017-01-20", "", "transition_negative"),  # Trump 1.0 inauguration
    ("2021-01-20", "", "transition_positive"),  # Biden inauguration
    ("2025-01-20", "", "transition_negative"),  # Trump 2.0 inauguration
]

# Color scheme
COLORS = {
    "cpu_index": "#1f77b4",      # Blue
    "cpu_impl": "#2ca02c",       # Green
    "cpu_reversal": "#d62728",   # Red
    "vc_deals": "#ff7f0e",       # Orange
    "vc_amount": "#9467bd",      # Purple
    "positive": "#2ca02c",       # Green for supportive policies
    "negative": "#d62728",       # Red for restrictive policies
    "election": "#7f7f7f",       # Gray for elections
    "transition_positive": "#d4edda",  # Light green shade
    "transition_negative": "#f8d7da",  # Light red shade
    "significant": "#2ca02c",
    "not_significant": "#7f7f7f",
}


def load_data():
    """Load CPU index and VC data."""
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "outputs" / "cpu_vc_analysis"

    # Load CPU index
    cpu_df = pd.read_csv(data_dir / "cpu_index.csv")
    cpu_df['month'] = pd.to_datetime(cpu_df['month'])
    cpu_df = cpu_df.set_index('month')

    # Load merged data
    merged_df = pd.read_csv(output_dir / "merged_cpu_vc_monthly.csv")
    merged_df['month'] = pd.to_datetime(merged_df.iloc[:, 0])  # First column is the index
    merged_df = merged_df.set_index('month')

    return cpu_df, merged_df


def add_policy_events(ax, y_min, y_max, show_labels=True, label_position='top'):
    """Add policy event annotations with prominent labels directly on the plot."""

    # Calculate label positions
    y_range = y_max - y_min
    if label_position == 'top':
        label_y = y_max - y_range * 0.05
    else:
        label_y = y_min + y_range * 0.05

    for date_str, label, event_type in POLICY_EVENTS:
        date = pd.to_datetime(date_str)

        # Skip if outside axis range
        xlim = ax.get_xlim()
        date_num = mdates.date2num(date)
        if date_num < xlim[0] or date_num > xlim[1]:
            continue

        if event_type == "election":
            ax.axvline(date, color=COLORS["election"], linestyle=':', alpha=0.4, linewidth=1)
            # Skip election labels to reduce clutter

        elif event_type == "positive":
            ax.axvline(date, color=COLORS["positive"], linestyle='-', alpha=0.8, linewidth=2)
            if show_labels and label:
                # Add background box for readability
                ax.annotate(label.replace('\n', ' '),
                           xy=(date, label_y),
                           fontsize=11, ha='center', va='top',
                           color='white', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS["positive"],
                                    edgecolor='none', alpha=0.9),
                           zorder=10)

        elif event_type == "negative":
            ax.axvline(date, color=COLORS["negative"], linestyle='-', alpha=0.8, linewidth=2)
            if show_labels and label:
                # Add background box for readability
                ax.annotate(label.replace('\n', ' '),
                           xy=(date, label_y),
                           fontsize=11, ha='center', va='top',
                           color='white', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS["negative"],
                                    edgecolor='none', alpha=0.9),
                           zorder=10)

        elif event_type.startswith("transition_"):
            # Shade periods after political transitions
            color = COLORS[event_type]
            end_date = date + pd.DateOffset(months=6)
            ax.axvspan(date, end_date, alpha=0.15, color=color, zorder=0)


def figure1_cpu_components_timeseries(cpu_df, merged_df, output_path):
    """
    Figure 1: CPU Index Components with Policy Events

    Upper panel: CPU_impl and CPU_reversal over time with policy annotations
    Lower panel: VC deal count
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08})

    dates = cpu_df.index

    # Upper panel: CPU components
    ax1.plot(dates, cpu_df['cpu_impl'], color=COLORS["cpu_impl"],
             linewidth=2.5, label='Implementation Uncertainty (CPU_impl)', alpha=0.9)
    ax1.plot(dates, cpu_df['cpu_reversal'], color=COLORS["cpu_reversal"],
             linewidth=2.5, label='Reversal Uncertainty (CPU_reversal)', alpha=0.9)
    ax1.plot(dates, cpu_df['cpu_index'], color=COLORS["cpu_index"],
             linewidth=1.5, label='Aggregate CPU', linestyle='--', alpha=0.6)

    ax1.axhline(100, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax1.set_ylabel('CPU Index (mean = 100)', fontsize=13)
    ax1.set_ylim(65, 140)

    # Add policy events with prominent labels
    add_policy_events(ax1, 65, 140, show_labels=True, label_position='top')

    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax1.set_title('Climate Policy Uncertainty: IRA Implementation vs OBBBA Reversal Risk',
                  fontsize=15, fontweight='bold', pad=10)

    # Lower panel: VC deals
    vc_dates = merged_df.index
    ax2.fill_between(vc_dates, merged_df['deal_count'], alpha=0.3, color=COLORS["vc_deals"])
    ax2.plot(vc_dates, merged_df['deal_count'], color=COLORS["vc_deals"], linewidth=2)

    # Add 12-month moving average
    ma = merged_df['deal_count'].rolling(12).mean()
    ax2.plot(vc_dates, ma, color=COLORS["vc_deals"], linewidth=2, linestyle='--',
             alpha=0.8, label='12-month MA')

    ax2.set_ylabel('VC Deals/Month', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylim(0, 160)

    # Add policy events to lower panel (no labels, just lines)
    add_policy_events(ax2, 0, 160, show_labels=False)

    ax2.legend(loc='upper left', fontsize=10)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    # Grid
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def figure2_cross_correlation(merged_df, output_path):
    """
    Figure 2: Cross-Correlation Function with annotations
    """
    from cpu_index.analysis.correlation import cross_correlation, find_optimal_lag
    from scipy import stats

    # Calculate CCF for reversal (the key finding)
    ccf_rev = cross_correlation(merged_df['cpu_reversal'], merged_df['deal_count'], max_lag=12)
    ccf_impl = cross_correlation(merged_df['cpu_impl'], merged_df['deal_count'], max_lag=12)

    fig, ax = plt.subplots(figsize=(12, 6))

    lags = ccf_rev['lag'].values
    n_obs = len(merged_df)

    # Calculate 95% confidence bounds
    z_critical = stats.norm.ppf(0.975)
    conf_bound = z_critical / np.sqrt(n_obs)

    # Bar width for side-by-side bars
    width = 0.35

    # Plot reversal CCF
    colors_rev = [COLORS["negative"] if abs(c) > conf_bound else COLORS["not_significant"]
                  for c in ccf_rev['correlation'].values]
    bars_rev = ax.bar(lags - width/2, ccf_rev['correlation'].values, width,
                      color=colors_rev, alpha=0.8, label='Reversal Uncertainty')

    # Plot implementation CCF
    colors_impl = [COLORS["positive"] if abs(c) > conf_bound else COLORS["not_significant"]
                   for c in ccf_impl['correlation'].values]
    bars_impl = ax.bar(lags + width/2, ccf_impl['correlation'].values, width,
                       color=colors_impl, alpha=0.8, label='Implementation Uncertainty')

    # Confidence bounds
    ax.axhline(conf_bound, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(-conf_bound, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(0, color='black', linewidth=0.5)

    # Annotate peak correlations
    opt_rev = find_optimal_lag(ccf_rev)
    opt_impl = find_optimal_lag(ccf_impl)

    # Highlight optimal lags
    ax.annotate(f'Peak: r={opt_rev["max_correlation"]:.2f}\nat lag {opt_rev["optimal_lag"]}',
                xy=(opt_rev["optimal_lag"], opt_rev["max_correlation"]),
                xytext=(opt_rev["optimal_lag"] - 3, opt_rev["max_correlation"] - 0.15),
                fontsize=10, color=COLORS["negative"],
                arrowprops=dict(arrowstyle='->', color=COLORS["negative"], alpha=0.7))

    ax.annotate(f'Peak: r={opt_impl["max_correlation"]:.2f}\nat lag {opt_impl["optimal_lag"]}',
                xy=(opt_impl["optimal_lag"], opt_impl["max_correlation"]),
                xytext=(opt_impl["optimal_lag"] + 2, opt_impl["max_correlation"] + 0.1),
                fontsize=10, color=COLORS["positive"],
                arrowprops=dict(arrowstyle='->', color=COLORS["positive"], alpha=0.7))

    ax.set_xlabel('Lag (months)\n← CPU leads VC | CPU lags VC →', fontsize=12)
    ax.set_ylabel('Cross-Correlation', fontsize=12)
    ax.set_title('Cross-Correlation: CPU Components vs VC Deal Count', fontsize=14, fontweight='bold')
    ax.set_xticks(lags)
    ax.set_ylim(-0.8, 0.5)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add interpretation note
    ax.text(0.98, 0.98, '95% confidence bounds shown',
            transform=ax.transAxes, fontsize=9, ha='right', va='top',
            style='italic', color='gray')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def figure3_rolling_correlation(merged_df, output_path):
    """
    Figure 3: Rolling Correlation with Political Regime Shading
    Highlights IRA implementation period vs OBBBA reversal risk period
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    window = 24  # 24-month rolling window

    # Calculate rolling correlations
    roll_rev = merged_df['cpu_reversal'].rolling(window).corr(merged_df['deal_count'])
    roll_impl = merged_df['cpu_impl'].rolling(window).corr(merged_df['deal_count'])

    dates = merged_df.index

    # Add political regime shading
    # Obama: 2009-01 to 2017-01
    ax.axvspan(pd.Timestamp('2009-01-20'), pd.Timestamp('2017-01-20'),
               alpha=0.1, color='blue', label='_nolegend_')
    # Trump 1: 2017-01 to 2021-01
    ax.axvspan(pd.Timestamp('2017-01-20'), pd.Timestamp('2021-01-20'),
               alpha=0.1, color='red', label='_nolegend_')
    # Biden/IRA era: 2021-01 to 2025-01
    ax.axvspan(pd.Timestamp('2021-01-20'), pd.Timestamp('2025-01-20'),
               alpha=0.1, color='blue', label='_nolegend_')
    # Trump 2/OBBBA era: 2025-01+
    ax.axvspan(pd.Timestamp('2025-01-20'), dates.max(),
               alpha=0.15, color='red', label='_nolegend_')

    # Plot rolling correlations
    ax.plot(dates, roll_rev, color=COLORS["cpu_reversal"], linewidth=2.5,
            label='Reversal Uncertainty', alpha=0.9)
    ax.plot(dates, roll_impl, color=COLORS["cpu_impl"], linewidth=2.5,
            label='Implementation Uncertainty', alpha=0.9)

    ax.axhline(0, color='black', linewidth=0.5)

    # Add regime labels with boxes
    ax.annotate('Obama', xy=(pd.Timestamp('2013-01-01'), 0.88),
                fontsize=12, ha='center', va='center', color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7))
    ax.annotate('Trump 1.0', xy=(pd.Timestamp('2019-01-01'), 0.88),
                fontsize=12, ha='center', va='center', color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
    ax.annotate('Biden', xy=(pd.Timestamp('2023-01-01'), 0.88),
                fontsize=12, ha='center', va='center', color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7))
    ax.annotate('Trump 2.0', xy=(pd.Timestamp('2025-03-15'), 0.88),
                fontsize=12, ha='center', va='center', color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))

    # Add IRA annotation with prominent box
    ira_date = pd.Timestamp('2022-08-16')
    ax.axvline(ira_date, color=COLORS["positive"], linestyle='-', linewidth=3, alpha=0.8)
    ax.annotate('IRA PASSED\n($369B Climate)', xy=(ira_date, 0.5),
                fontsize=11, ha='center', va='center', color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS["positive"], edgecolor='darkgreen', linewidth=2),
                zorder=10)

    # Add OBBBA annotation with prominent box
    obbba_date = pd.Timestamp('2025-01-20')
    ax.axvline(obbba_date, color=COLORS["negative"], linestyle='-', linewidth=3, alpha=0.8)
    ax.annotate('OBBBA\n(IRA Repeal Risk)', xy=(obbba_date, -0.5),
                fontsize=11, ha='center', va='center', color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS["negative"], edgecolor='darkred', linewidth=2),
                zorder=10)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{window}-Month Rolling Correlation with VC Deals', fontsize=12)
    ax.set_title('Rolling Correlation: Policy Uncertainty Components vs VC Activity',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def figure4_vc_stage_distribution(merged_df, output_path):
    """
    Figure 4: VC Stage Distribution with Policy Event Markers
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = merged_df.index

    # Create stacked area chart
    seed = merged_df['seed_count'].values
    early = merged_df['early_count'].values
    late = merged_df['late_count'].values

    ax.stackplot(dates, seed, early, late,
                 labels=['Seed/Angel', 'Early Stage', 'Late Stage'],
                 colors=[COLORS["cpu_impl"], COLORS["cpu_index"], COLORS["cpu_reversal"]],
                 alpha=0.7)

    # Add policy events
    for date_str, label, event_type in POLICY_EVENTS:
        date = pd.to_datetime(date_str)
        if date < dates.min() or date > dates.max():
            continue

        if event_type in ["positive", "negative"]:
            color = COLORS[event_type]
            ax.axvline(date, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
            if label:
                ax.text(date, ax.get_ylim()[1] * 0.95, label.replace('\n', ' '),
                       fontsize=8, ha='center', va='top', color=color,
                       fontweight='bold', rotation=90)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of VC Deals', fontsize=12)
    ax.set_title('Climate Tech VC Deals by Stage', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def figure5_decomposition_scatter(merged_df, output_path):
    """
    Figure 5: Scatter plots showing CPU component vs VC deals relationship
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Reversal vs Deals
    ax1 = axes[0]
    ax1.scatter(merged_df['cpu_reversal'], merged_df['deal_count'],
                alpha=0.5, color=COLORS["cpu_reversal"], s=30)
    z = np.polyfit(merged_df['cpu_reversal'], merged_df['deal_count'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_df['cpu_reversal'].min(), merged_df['cpu_reversal'].max(), 100)
    ax1.plot(x_line, p(x_line), color=COLORS["cpu_reversal"], linewidth=2, linestyle='--')
    r = merged_df['cpu_reversal'].corr(merged_df['deal_count'])
    ax1.text(0.05, 0.95, f'r = {r:.2f}', transform=ax1.transAxes, fontsize=12,
             fontweight='bold', color=COLORS["cpu_reversal"], va='top')
    ax1.set_xlabel('Reversal Uncertainty Index', fontsize=11)
    ax1.set_ylabel('VC Deals/Month', fontsize=11)
    ax1.set_title('Reversal Uncertainty\n(Strong Negative)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Implementation vs Deals
    ax2 = axes[1]
    ax2.scatter(merged_df['cpu_impl'], merged_df['deal_count'],
                alpha=0.5, color=COLORS["cpu_impl"], s=30)
    z = np.polyfit(merged_df['cpu_impl'], merged_df['deal_count'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_df['cpu_impl'].min(), merged_df['cpu_impl'].max(), 100)
    ax2.plot(x_line, p(x_line), color=COLORS["cpu_impl"], linewidth=2, linestyle='--')
    r = merged_df['cpu_impl'].corr(merged_df['deal_count'])
    ax2.text(0.05, 0.95, f'r = {r:.2f}', transform=ax2.transAxes, fontsize=12,
             fontweight='bold', color=COLORS["cpu_impl"], va='top')
    ax2.set_xlabel('Implementation Uncertainty Index', fontsize=11)
    ax2.set_ylabel('VC Deals/Month', fontsize=11)
    ax2.set_title('Implementation Uncertainty\n(Moderate Positive)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Aggregate vs Deals
    ax3 = axes[2]
    ax3.scatter(merged_df['cpu_index'], merged_df['deal_count'],
                alpha=0.5, color=COLORS["cpu_index"], s=30)
    z = np.polyfit(merged_df['cpu_index'], merged_df['deal_count'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_df['cpu_index'].min(), merged_df['cpu_index'].max(), 100)
    ax3.plot(x_line, p(x_line), color=COLORS["cpu_index"], linewidth=2, linestyle='--')
    r = merged_df['cpu_index'].corr(merged_df['deal_count'])
    ax3.text(0.05, 0.95, f'r = {r:.2f}', transform=ax3.transAxes, fontsize=12,
             fontweight='bold', color=COLORS["cpu_index"], va='top')
    ax3.set_xlabel('Aggregate CPU Index', fontsize=11)
    ax3.set_ylabel('VC Deals/Month', fontsize=11)
    ax3.set_title('Aggregate Index\n(Negligible)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.suptitle('CPU Components vs VC Deal Activity: Opposing Effects Cancel in Aggregate',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    """Generate all publication figures."""
    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "exports_for_pi"
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("\nLoading data...")
    cpu_df, merged_df = load_data()
    print(f"  CPU data: {len(cpu_df)} months")
    print(f"  Merged data: {len(merged_df)} months")

    # Generate figures
    print("\nGenerating figures...")

    figure1_cpu_components_timeseries(
        cpu_df, merged_df,
        output_dir / "fig1_cpu_vc_timeseries.png"
    )

    figure2_cross_correlation(
        merged_df,
        output_dir / "fig2_cross_correlation.png"
    )

    figure3_rolling_correlation(
        merged_df,
        output_dir / "fig3_rolling_correlation.png"
    )

    figure4_vc_stage_distribution(
        merged_df,
        output_dir / "fig4_vc_stage_distribution.png"
    )

    figure5_decomposition_scatter(
        merged_df,
        output_dir / "fig5_decomposition_scatter.png"
    )

    print("\n" + "=" * 60)
    print("FIGURES GENERATED")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nFigures for LaTeX:")
    print("  fig1_cpu_vc_timeseries.png   - Main time series with policy events")
    print("  fig2_cross_correlation.png    - CCF with peak annotations")
    print("  fig3_rolling_correlation.png  - Rolling correlation with regimes")
    print("  fig4_vc_stage_distribution.png - Stage breakdown")
    print("  fig5_decomposition_scatter.png - Scatter plots showing opposing effects")


if __name__ == "__main__":
    main()
