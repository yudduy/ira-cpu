#!/usr/bin/env python3
"""
Enhanced Sector Analysis Visualizations

Creates publication-quality figures for sector-specific CPU-VC correlation analysis
with intuitive interpretations and event annotations.

Figures:
1. Summary Dashboard - Key findings at a glance
2. Sensitivity Ranking with Lead-Lag - Shows direction AND timing
3. Decomposition Story - Implementation vs Reversal narrative
4. Timeline with Events - CPU and sector VC with IRA/OBBBA markers
5. Mechanism Diagram - Visual explanation of findings
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "lines.linewidth": 1.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colors
COLORS = {
    "positive": "#2E7D32",  # Green - VC leads or positive correlation
    "negative": "#C62828",  # Red - CPU leads or negative correlation
    "neutral": "#757575",   # Gray
    "impl": "#1976D2",      # Blue - Implementation
    "reversal": "#D32F2F",  # Red - Reversal
    "balanced": "#7B1FA2",  # Purple - Balanced
    "cpu": "#1565C0",       # Dark blue - CPU line
    "ira": "#2E7D32",       # Green - IRA event
    "obbba": "#FF6F00",     # Orange - OBBBA event
    "election": "#C62828",  # Red - Election event
}

SECTOR_COLORS = {
    "Industrial": "#FF9800",
    "Energy": "#2196F3",
    "Carbon": "#9C27B0",
    "Built_Environment": "#4CAF50",
    "Food_Land_Use": "#795548",
    "Climate_Mgmt": "#607D8B",
    "Transportation": "#E91E63",
}

# Key events
EVENTS = [
    {"date": "2021-01", "label": "Biden\nInauguration", "color": COLORS["ira"]},
    {"date": "2022-08", "label": "IRA\nSigned", "color": COLORS["ira"]},
    {"date": "2024-11", "label": "Trump\nElection", "color": COLORS["election"]},
    {"date": "2025-01", "label": "Trump\nInauguration", "color": COLORS["election"]},
]


# =============================================================================
# FIGURE 1: SUMMARY DASHBOARD
# =============================================================================

def plot_summary_dashboard(ranking_df, decomp_df, ira_df, output_path):
    """
    Create a single-page summary of key findings.
    """
    fig = plt.figure(figsize=(16, 12))

    # Title
    fig.suptitle("Climate Tech Sector Sensitivity to Policy Uncertainty\n(IRA Era: 2021-2025)",
                 fontsize=18, fontweight="bold", y=0.98)

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3,
                          left=0.08, right=0.92, top=0.90, bottom=0.08)

    # --- Panel A: Key Finding Box ---
    ax_key = fig.add_subplot(gs[0, :])
    ax_key.axis("off")

    # Key findings text box
    key_text = """KEY FINDINGS

• Industrial is the ONLY sector dominated by implementation uncertainty (not reversal fear)
• Energy, Built Environment, Food & Land Use, Climate Management all show CPU → VC suppression
• High-IRA exposure companies (score ≥6) are MORE sensitive to policy uncertainty than low-IRA companies
• Most sectors fear policy REVERSAL more than implementation delays"""

    ax_key.text(0.5, 0.5, key_text, transform=ax_key.transAxes,
                fontsize=13, va="center", ha="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", edgecolor="#1976D2", linewidth=2),
                family="monospace")

    # --- Panel B: Sensitivity Ranking ---
    ax_rank = fig.add_subplot(gs[1, 0])

    df = ranking_df.sort_values("abs_correlation", ascending=True)
    colors = [COLORS["positive"] if d == "positive" else COLORS["negative"]
              for d in df["direction"]]

    bars = ax_rank.barh(df["sector"], df["abs_correlation"], color=colors, alpha=0.8)
    ax_rank.set_xlabel("|Correlation| with CPU")
    ax_rank.set_title("A. Sector Sensitivity Ranking")
    ax_rank.set_xlim(0, 0.6)

    # Add correlation values
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax_rank.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{row['correlation']:+.2f}", va="center", fontsize=9)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["positive"], label="Positive (VC leads CPU)"),
        mpatches.Patch(facecolor=COLORS["negative"], label="Negative (CPU suppresses VC)"),
    ]
    ax_rank.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # --- Panel C: Lead-Lag Pattern ---
    ax_lag = fig.add_subplot(gs[1, 1])

    # Plot optimal lags
    df_sorted = ranking_df.sort_values("optimal_lag")
    y_pos = range(len(df_sorted))
    colors = [SECTOR_COLORS.get(s, "#757575") for s in df_sorted["sector"]]

    ax_lag.barh(y_pos, df_sorted["optimal_lag"], color=colors, alpha=0.8)
    ax_lag.set_yticks(y_pos)
    ax_lag.set_yticklabels(df_sorted["sector"])
    ax_lag.axvline(0, color="black", linewidth=1)
    ax_lag.set_xlabel("Optimal Lag (months)")
    ax_lag.set_title("B. Lead-Lag Structure")

    # Add annotations
    ax_lag.text(-11, -0.8, "← VC LEADS CPU", fontsize=9, ha="left", color=COLORS["positive"])
    ax_lag.text(11, -0.8, "CPU LEADS VC →", fontsize=9, ha="right", color=COLORS["negative"])

    # --- Panel D: Decomposition ---
    ax_decomp = fig.add_subplot(gs[1, 2])

    df_decomp = decomp_df.sort_values("asymmetry_ratio")
    colors = []
    for dt in df_decomp["dominant_type"]:
        if dt == "implementation":
            colors.append(COLORS["impl"])
        elif dt == "reversal":
            colors.append(COLORS["reversal"])
        else:
            colors.append(COLORS["balanced"])

    ax_decomp.barh(df_decomp["sector"], df_decomp["asymmetry_ratio"], color=colors, alpha=0.8)
    ax_decomp.axvline(0, color="black", linewidth=1)
    ax_decomp.set_xlabel("Asymmetry Ratio")
    ax_decomp.set_title("C. Implementation vs Reversal")
    ax_decomp.set_xlim(-0.5, 0.5)

    # Add labels
    ax_decomp.text(-0.45, -0.8, "← REVERSAL-dominated", fontsize=9, ha="left", color=COLORS["reversal"])
    ax_decomp.text(0.45, -0.8, "IMPL-dominated →", fontsize=9, ha="right", color=COLORS["impl"])

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["impl"], label="Implementation"),
        mpatches.Patch(facecolor=COLORS["reversal"], label="Reversal"),
        mpatches.Patch(facecolor=COLORS["balanced"], label="Balanced"),
    ]
    ax_decomp.legend(handles=legend_elements, loc="upper left", fontsize=8)

    # --- Panel E: IRA Exposure ---
    ax_ira = fig.add_subplot(gs[2, 0])

    groups = ["High IRA\n(score ≥6)", "Low IRA\n(score ≤3)"]
    correlations = [ira_df.loc[ira_df["group"] == "high_ira", "correlation"].values[0],
                   ira_df.loc[ira_df["group"] == "low_ira", "correlation"].values[0]]
    n_companies = [ira_df.loc[ira_df["group"] == "high_ira", "n_companies"].values[0],
                  ira_df.loc[ira_df["group"] == "low_ira", "n_companies"].values[0]]

    colors = [COLORS["positive"] if c > 0 else COLORS["negative"] for c in correlations]
    bars = ax_ira.bar(groups, correlations, color=colors, alpha=0.8)
    ax_ira.axhline(0, color="black", linewidth=0.5)
    ax_ira.set_ylabel("Correlation with CPU")
    ax_ira.set_title("D. IRA Exposure Effect")
    ax_ira.set_ylim(-0.5, 0.5)

    # Add n labels
    for bar, n in zip(bars, n_companies):
        height = bar.get_height()
        ax_ira.text(bar.get_x() + bar.get_width()/2, height + 0.03 if height > 0 else height - 0.08,
                   f"n={n:,}", ha="center", fontsize=9)

    # --- Panel F: Interpretation Box ---
    ax_interp = fig.add_subplot(gs[2, 1:])
    ax_interp.axis("off")

    interp_text = """INTERPRETATION

PATTERN 1: "Dark Spots" - Sectors Where Uncertainty Hurts Investment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Energy, Built Environment, Food & Land Use, Climate Management show NEGATIVE correlations
with CPU leading VC by 3-4 months. This is the classic "uncertainty suppresses investment" channel.

PATTERN 2: Industrial is Different
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Industrial is the ONLY sector dominated by implementation uncertainty (not reversal fear).
Manufacturing companies need operational certainty for capex decisions - they're less worried
about policy repeal and more worried about "when will the rules take effect?"

PATTERN 3: High-IRA Companies More Sensitive
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Companies with high IRA exposure (score ≥6) show STRONGER CPU correlation (+0.42) than
low-IRA companies (-0.32). The more policy-dependent the business model, the more
sensitive to policy uncertainty."""

    ax_interp.text(0.02, 0.95, interp_text, transform=ax_interp.transAxes,
                   fontsize=10, va="top", ha="left", family="monospace",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor="#FF9800"))

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# =============================================================================
# FIGURE 2: SENSITIVITY WITH LEAD-LAG EXPLANATION
# =============================================================================

def plot_sensitivity_with_leadlag(ranking_df, output_path):
    """
    Create a clear visualization of sensitivity AND lead-lag structure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    fig.suptitle("Sector CPU Sensitivity: Magnitude and Timing", fontsize=16, fontweight="bold")

    # --- Left: Sensitivity magnitude ---
    df = ranking_df.sort_values("abs_correlation", ascending=True)

    # Create color based on direction
    colors = [COLORS["positive"] if d == "positive" else COLORS["negative"]
              for d in df["direction"]]

    bars = ax1.barh(range(len(df)), df["abs_correlation"], color=colors, alpha=0.85, height=0.7)
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels(df["sector"], fontsize=11)
    ax1.set_xlabel("Absolute Correlation at Optimal Lag", fontsize=12)
    ax1.set_title("A. Sensitivity Magnitude", fontsize=13)
    ax1.set_xlim(0, 0.6)

    # Add value labels with actual correlation (signed)
    for i, (bar, (_, row)) in enumerate(zip(bars, df.iterrows())):
        color = "white" if bar.get_width() > 0.25 else "black"
        ax1.text(bar.get_width() - 0.02 if bar.get_width() > 0.25 else bar.get_width() + 0.02,
                bar.get_y() + bar.get_height()/2,
                f"r = {row['correlation']:+.3f}",
                va="center", ha="right" if bar.get_width() > 0.25 else "left",
                fontsize=10, color=color, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["positive"], label="Positive: VC activity LEADS uncertainty"),
        mpatches.Patch(facecolor=COLORS["negative"], label="Negative: Uncertainty SUPPRESSES VC"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.9)

    # --- Right: Lead-lag structure ---
    df_sorted = ranking_df.sort_values("optimal_lag")

    # Create horizontal bar showing lead-lag
    y_pos = range(len(df_sorted))
    lags = df_sorted["optimal_lag"].values

    # Color by whether VC leads (negative lag) or CPU leads (positive lag)
    colors = [COLORS["positive"] if lag < 0 else COLORS["negative"] for lag in lags]

    bars = ax2.barh(y_pos, lags, color=colors, alpha=0.85, height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_sorted["sector"], fontsize=11)
    ax2.axvline(0, color="black", linewidth=1.5)
    ax2.set_xlabel("Optimal Lag (months)", fontsize=12)
    ax2.set_title("B. Lead-Lag Structure", fontsize=13)
    ax2.set_xlim(-10, 6)

    # Add lag value labels
    for bar, lag in zip(bars, lags):
        x_pos = lag - 0.5 if lag < 0 else lag + 0.5
        ha = "right" if lag < 0 else "left"
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                f"{int(lag)} mo", va="center", ha=ha, fontsize=10, fontweight="bold")

    # Add interpretation arrows
    ax2.annotate("", xy=(-9, -1), xytext=(-2, -1),
                arrowprops=dict(arrowstyle="->", color=COLORS["positive"], lw=2))
    ax2.text(-5.5, -1.4, "VC LEADS CPU\n(anticipatory)", ha="center", fontsize=10,
            color=COLORS["positive"], fontweight="bold")

    ax2.annotate("", xy=(5, -1), xytext=(1, -1),
                arrowprops=dict(arrowstyle="->", color=COLORS["negative"], lw=2))
    ax2.text(3, -1.4, "CPU LEADS VC\n(reactive)", ha="center", fontsize=10,
            color=COLORS["negative"], fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# =============================================================================
# FIGURE 3: DECOMPOSITION NARRATIVE
# =============================================================================

def plot_decomposition_narrative(decomp_df, output_path):
    """
    Create a clear narrative visualization of implementation vs reversal sensitivity.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    fig.suptitle("CPU Decomposition: What Type of Uncertainty Matters?",
                 fontsize=16, fontweight="bold")

    # --- Panel A: Scatter plot with quadrants ---
    ax1 = axes[0]

    for _, row in decomp_df.iterrows():
        color = SECTOR_COLORS.get(row["sector"], "#757575")
        ax1.scatter(row["impl_abs"], row["reversal_abs"],
                   s=200, c=color, alpha=0.8, edgecolors="white", linewidth=2)

        # Add labels
        offset = 0.02
        ax1.annotate(row["sector"].replace("_", "\n"),
                    (row["impl_abs"] + offset, row["reversal_abs"] + offset),
                    fontsize=9, fontweight="bold")

    # Add diagonal
    ax1.plot([0, 0.6], [0, 0.6], "k--", alpha=0.5, label="Equal sensitivity")

    # Add quadrant labels
    ax1.text(0.5, 0.15, "IMPLEMENTATION\nDOMINATED", ha="center", fontsize=10,
            color=COLORS["impl"], alpha=0.7, fontweight="bold")
    ax1.text(0.15, 0.5, "REVERSAL\nDOMINATED", ha="center", fontsize=10,
            color=COLORS["reversal"], alpha=0.7, fontweight="bold")

    ax1.set_xlabel("|Implementation Uncertainty Correlation|", fontsize=11)
    ax1.set_ylabel("|Reversal Uncertainty Correlation|", fontsize=11)
    ax1.set_title("A. Sensitivity by Uncertainty Type", fontsize=12)
    ax1.set_xlim(0, 0.6)
    ax1.set_ylim(0, 0.6)
    ax1.set_aspect("equal")
    ax1.legend(loc="lower right", fontsize=9)

    # --- Panel B: Asymmetry bar chart ---
    ax2 = axes[1]

    df_sorted = decomp_df.sort_values("asymmetry_ratio")
    colors = []
    for dt in df_sorted["dominant_type"]:
        if dt == "implementation":
            colors.append(COLORS["impl"])
        elif dt == "reversal":
            colors.append(COLORS["reversal"])
        else:
            colors.append(COLORS["balanced"])

    bars = ax2.barh(range(len(df_sorted)), df_sorted["asymmetry_ratio"],
                   color=colors, alpha=0.85, height=0.7)
    ax2.set_yticks(range(len(df_sorted)))
    ax2.set_yticklabels(df_sorted["sector"], fontsize=11)
    ax2.axvline(0, color="black", linewidth=1.5)
    ax2.set_xlabel("Asymmetry Ratio", fontsize=11)
    ax2.set_title("B. Dominant Uncertainty Type", fontsize=12)
    ax2.set_xlim(-0.5, 0.35)

    # Add threshold zones
    ax2.axvspan(-0.5, -0.1, alpha=0.1, color=COLORS["reversal"])
    ax2.axvspan(0.1, 0.35, alpha=0.1, color=COLORS["impl"])

    # Add labels
    ax2.text(-0.4, -0.8, "← More REVERSAL-sensitive", fontsize=10,
            color=COLORS["reversal"], fontweight="bold")
    ax2.text(0.25, -0.8, "More IMPL-sensitive →", fontsize=10,
            color=COLORS["impl"], fontweight="bold")

    # --- Panel C: Interpretation ---
    ax3 = axes[2]
    ax3.axis("off")

    # Create explanation boxes
    interp_text = """
WHAT THIS MEANS:

┌─────────────────────────────────────┐
│  REVERSAL-DOMINATED SECTORS         │
│  (Climate Mgmt, Transport, Food,    │
│   Built Environment)                │
│                                     │
│  These sectors fear policy REPEAL   │
│  more than implementation delays.   │
│  The OBBBA threat matters more      │
│  than Treasury guidance delays.     │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  IMPLEMENTATION-DOMINATED SECTOR    │
│  (Industrial - ONLY ONE!)           │
│                                     │
│  Manufacturing needs operational    │
│  certainty for capex decisions.     │
│  "When will rules take effect?"     │
│  matters more than "will they       │
│  be repealed?"                      │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  BALANCED SECTORS                   │
│  (Energy, Carbon)                   │
│                                     │
│  Both types of uncertainty affect   │
│  investment equally.                │
└─────────────────────────────────────┘
"""
    ax3.text(0.05, 0.95, interp_text, transform=ax3.transAxes,
            fontsize=10, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="#9E9E9E"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# =============================================================================
# FIGURE 4: TIMELINE WITH EVENTS
# =============================================================================

def plot_timeline_with_events(cpu_df, sector_monthly, ranking_df, output_path):
    """
    Create time series showing CPU and sector VC with policy events marked.
    """
    # Get top 4 most sensitive sectors
    top_sectors = ranking_df.head(4)["sector"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    fig.suptitle("CPU Index vs Sector VC Activity (IRA Era 2021-2025)",
                 fontsize=16, fontweight="bold")

    # Prepare CPU data
    cpu = cpu_df.copy()
    cpu["month"] = pd.to_datetime(cpu["month"])
    cpu = cpu[cpu["month"] >= "2021-01-01"]
    cpu = cpu.set_index("month")

    for idx, sector in enumerate(top_sectors):
        ax1 = axes[idx]

        # Get sector data
        if sector not in sector_monthly:
            continue
        vc_series = sector_monthly[sector]
        if not isinstance(vc_series.index, pd.DatetimeIndex):
            vc_series.index = pd.to_datetime(vc_series.index)
        vc_series = vc_series[vc_series.index >= "2021-01-01"]

        # Plot CPU
        ax1.plot(cpu.index, cpu["cpu_index"], color=COLORS["cpu"],
                linewidth=2, label="CPU Index")
        ax1.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax1.set_ylabel("CPU Index", color=COLORS["cpu"])
        ax1.tick_params(axis="y", labelcolor=COLORS["cpu"])

        # Plot VC on secondary axis
        ax2 = ax1.twinx()
        sector_color = SECTOR_COLORS.get(sector, "#FF9800")
        ax2.fill_between(vc_series.index, 0, vc_series.values,
                        alpha=0.3, color=sector_color)
        ax2.plot(vc_series.index, vc_series.values,
                color=sector_color, linewidth=2, label=f"{sector} Deals")
        ax2.set_ylabel("Deal Count", color=sector_color)
        ax2.tick_params(axis="y", labelcolor=sector_color)

        # Add events
        for event in EVENTS:
            event_date = pd.to_datetime(event["date"])
            if event_date >= cpu.index.min():
                ax1.axvline(event_date, color=event["color"],
                           linestyle="--", linewidth=1.5, alpha=0.7)
                ax1.text(event_date, ax1.get_ylim()[1] * 0.98, event["label"],
                        ha="center", va="top", fontsize=8, color=event["color"],
                        fontweight="bold")

        # Get ranking info for title
        row = ranking_df[ranking_df["sector"] == sector].iloc[0]
        title = f"{sector}: r={row['correlation']:+.3f}, lag={int(row['optimal_lag'])} mo"
        ax1.set_title(title, fontsize=12, fontweight="bold")

        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# =============================================================================
# FIGURE 5: MECHANISM DIAGRAM
# =============================================================================

def plot_mechanism_diagram(output_path):
    """
    Create a conceptual diagram explaining the findings.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    fig.suptitle("Mechanism: How Policy Uncertainty Affects Climate Tech VC",
                 fontsize=16, fontweight="bold")

    # --- CPU Box (center top) ---
    cpu_box = FancyBboxPatch((5.5, 7.5), 3, 1.5, boxstyle="round,pad=0.1",
                              facecolor="#E3F2FD", edgecolor=COLORS["cpu"], linewidth=2)
    ax.add_patch(cpu_box)
    ax.text(7, 8.25, "Climate Policy\nUncertainty (CPU)", ha="center", va="center",
           fontsize=12, fontweight="bold")

    # --- Decomposition boxes ---
    # Implementation
    impl_box = FancyBboxPatch((2, 5), 3, 1.2, boxstyle="round,pad=0.1",
                               facecolor="#E8F5E9", edgecolor=COLORS["impl"], linewidth=2)
    ax.add_patch(impl_box)
    ax.text(3.5, 5.6, "Implementation\nUncertainty", ha="center", va="center",
           fontsize=11, fontweight="bold", color=COLORS["impl"])

    # Reversal
    rev_box = FancyBboxPatch((9, 5), 3, 1.2, boxstyle="round,pad=0.1",
                              facecolor="#FFEBEE", edgecolor=COLORS["reversal"], linewidth=2)
    ax.add_patch(rev_box)
    ax.text(10.5, 5.6, "Reversal\nUncertainty", ha="center", va="center",
           fontsize=11, fontweight="bold", color=COLORS["reversal"])

    # Arrows from CPU to decomposition
    ax.annotate("", xy=(3.5, 6.2), xytext=(6, 7.5),
               arrowprops=dict(arrowstyle="->", color="gray", lw=2))
    ax.annotate("", xy=(10.5, 6.2), xytext=(8, 7.5),
               arrowprops=dict(arrowstyle="->", color="gray", lw=2))

    # --- Sector boxes ---
    # Industrial (implementation-dominated)
    ind_box = FancyBboxPatch((1, 2), 3.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor="#FFF3E0", edgecolor=SECTOR_COLORS["Industrial"], linewidth=2)
    ax.add_patch(ind_box)
    ax.text(2.75, 2.75, "Industrial\n(Implementation-dominated)", ha="center", va="center",
           fontsize=10, fontweight="bold")
    ax.text(2.75, 2.1, '"When will rules take effect?"', ha="center", va="center",
           fontsize=9, style="italic")

    # Arrow from impl to industrial
    ax.annotate("", xy=(2.75, 3.5), xytext=(3.5, 5),
               arrowprops=dict(arrowstyle="->", color=COLORS["impl"], lw=2))

    # Energy/Carbon (balanced)
    bal_box = FancyBboxPatch((5.25, 2), 3.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor="#F3E5F5", edgecolor=COLORS["balanced"], linewidth=2)
    ax.add_patch(bal_box)
    ax.text(7, 2.75, "Energy, Carbon\n(Balanced)", ha="center", va="center",
           fontsize=10, fontweight="bold")
    ax.text(7, 2.1, "Both uncertainties matter", ha="center", va="center",
           fontsize=9, style="italic")

    # Arrows from both to balanced
    ax.annotate("", xy=(6, 3.5), xytext=(3.5, 5),
               arrowprops=dict(arrowstyle="->", color=COLORS["impl"], lw=1.5, alpha=0.5))
    ax.annotate("", xy=(8, 3.5), xytext=(10.5, 5),
               arrowprops=dict(arrowstyle="->", color=COLORS["reversal"], lw=1.5, alpha=0.5))

    # Others (reversal-dominated)
    rev_box2 = FancyBboxPatch((9.5, 2), 3.5, 1.5, boxstyle="round,pad=0.1",
                               facecolor="#FFEBEE", edgecolor=COLORS["reversal"], linewidth=2)
    ax.add_patch(rev_box2)
    ax.text(11.25, 2.75, "Climate Mgmt, Transport,\nFood, Built Env", ha="center", va="center",
           fontsize=10, fontweight="bold")
    ax.text(11.25, 2.1, '"Will policy be repealed?"', ha="center", va="center",
           fontsize=9, style="italic")

    # Arrow from reversal to others
    ax.annotate("", xy=(11.25, 3.5), xytext=(10.5, 5),
               arrowprops=dict(arrowstyle="->", color=COLORS["reversal"], lw=2))

    # --- VC Investment outcome ---
    vc_box = FancyBboxPatch((5.5, 0.2), 3, 1),
    ax.add_patch(FancyBboxPatch((5.5, 0.2), 3, 1, boxstyle="round,pad=0.1",
                                 facecolor="#E0E0E0", edgecolor="gray", linewidth=2))
    ax.text(7, 0.7, "VC Investment\nDecisions", ha="center", va="center",
           fontsize=11, fontweight="bold")

    # Arrows to VC
    ax.annotate("", xy=(5.5, 0.7), xytext=(2.75, 2),
               arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(7, 1.2), xytext=(7, 2),
               arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("", xy=(8.5, 0.7), xytext=(11.25, 2),
               arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    # --- Legend/Key ---
    ax.text(0.5, 9.5, "KEY INSIGHT: Industrial is unique - it responds to implementation delays,\n"
                      "while most sectors fear policy reversal/repeal (the 'dark side' of uncertainty).",
           fontsize=11, fontweight="bold", va="top",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFDE7", edgecolor="#FFC107"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate all enhanced sector visualizations."""

    # Paths
    output_dir = Path("outputs/sector_analysis")
    data_dir = Path("data")

    print("=" * 60)
    print("GENERATING ENHANCED SECTOR VISUALIZATIONS")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    ranking_df = pd.read_csv(output_dir / "sector_rankings.csv")
    decomp_df = pd.read_csv(output_dir / "decomposition.csv")
    ira_df = pd.read_csv(output_dir / "ira_stratification.csv")
    cpu_df = pd.read_csv(data_dir / "cpu_index.csv")

    # Load VC data for time series
    vc_df = pd.read_csv(data_dir / "judged_results.csv", low_memory=False)
    vc_df["Last Financing Date"] = pd.to_datetime(vc_df["Last Financing Date"], errors="coerce")

    # Prepare sector monthly data
    print("2. Preparing sector monthly data...")
    vc = vc_df.copy()
    vc = vc[(vc["Last Financing Date"] >= "2021-01-01") & (vc["Last Financing Date"] <= "2025-12-31")]
    vc["YearMonth"] = vc["Last Financing Date"].dt.to_period("M")

    sector_monthly = {}
    for sector in vc["judge_category"].dropna().unique():
        if sector == "Others":
            continue
        sector_vc = vc[vc["judge_category"] == sector]
        monthly = sector_vc.groupby("YearMonth").size()
        monthly.index = monthly.index.to_timestamp()

        # Fill gaps
        full_range = pd.date_range(start="2021-01-01", end="2025-05-01", freq="MS")
        monthly = monthly.reindex(full_range, fill_value=0)
        sector_monthly[sector] = monthly

    # Generate figures
    print("\n3. Generating figures...")

    # Figure 1: Summary Dashboard
    plot_summary_dashboard(
        ranking_df, decomp_df, ira_df,
        output_dir / "fig_summary_dashboard.png"
    )

    # Figure 2: Sensitivity with Lead-Lag
    plot_sensitivity_with_leadlag(
        ranking_df,
        output_dir / "fig_sensitivity_leadlag.png"
    )

    # Figure 3: Decomposition Narrative
    plot_decomposition_narrative(
        decomp_df,
        output_dir / "fig_decomposition_narrative.png"
    )

    # Figure 4: Timeline with Events
    plot_timeline_with_events(
        cpu_df, sector_monthly, ranking_df,
        output_dir / "fig_timeline_events.png"
    )

    # Figure 5: Mechanism Diagram
    plot_mechanism_diagram(
        output_dir / "fig_mechanism_diagram.png"
    )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nFigures saved to: {output_dir}")
    print("\nGenerated figures:")
    print("  - fig_summary_dashboard.png    (Key findings at a glance)")
    print("  - fig_sensitivity_leadlag.png  (Magnitude + timing)")
    print("  - fig_decomposition_narrative.png (Impl vs Reversal story)")
    print("  - fig_timeline_events.png      (Time series with IRA/OBBBA)")
    print("  - fig_mechanism_diagram.png    (Conceptual framework)")


if __name__ == "__main__":
    main()
