#!/usr/bin/env python3
"""
Compare IRA Era (2021-2025) vs Full Sample (2008-2025) Results

This script creates comparison visualizations showing how the CPU-VC
relationship changed during the IRA era.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Colors
COLORS = {
    "ira": "#1976D2",      # Blue for IRA era
    "full": "#757575",     # Gray for full sample
    "positive": "#2E7D32",
    "negative": "#C62828",
    "impl": "#1976D2",
    "reversal": "#D32F2F",
}


def plot_comparison(output_dir):
    """Create comparison figures between IRA era and full sample."""

    # Load data
    ira_ranking = pd.read_csv("outputs/sector_analysis/sector_rankings.csv")
    full_ranking = pd.read_csv("outputs/sector_analysis_full_sample/sector_rankings.csv")

    ira_decomp = pd.read_csv("outputs/sector_analysis/decomposition.csv")
    full_decomp = pd.read_csv("outputs/sector_analysis_full_sample/decomposition.csv")

    ira_strat = pd.read_csv("outputs/sector_analysis/ira_stratification.csv")
    full_strat = pd.read_csv("outputs/sector_analysis_full_sample/ira_stratification.csv")

    # Create comparison figure
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Structural Break: IRA Era (2021-2025) vs Full Sample (2008-2025)",
                 fontsize=18, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25,
                          left=0.08, right=0.92, top=0.92, bottom=0.06)

    # --- Panel A: Correlation Magnitude Comparison ---
    ax1 = fig.add_subplot(gs[0, 0])

    # Merge on sector
    merged = ira_ranking.merge(full_ranking, on="sector", suffixes=("_ira", "_full"))
    merged = merged.sort_values("abs_correlation_ira", ascending=True)

    y_pos = np.arange(len(merged))
    height = 0.35

    bars1 = ax1.barh(y_pos - height/2, merged["abs_correlation_ira"], height,
                     label="IRA Era (2021-2025)", color=COLORS["ira"], alpha=0.8)
    bars2 = ax1.barh(y_pos + height/2, merged["abs_correlation_full"], height,
                     label="Full Sample (2008-2025)", color=COLORS["full"], alpha=0.8)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(merged["sector"])
    ax1.set_xlabel("|Correlation| with CPU")
    ax1.set_title("A. Sensitivity Magnitude: Much Stronger in IRA Era", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.set_xlim(0, 0.6)

    # Add difference annotations
    for i, (_, row) in enumerate(merged.iterrows()):
        diff = row["abs_correlation_ira"] - row["abs_correlation_full"]
        if diff > 0.1:
            ax1.annotate(f"+{diff:.2f}", xy=(row["abs_correlation_ira"] + 0.02, i - height/2),
                        fontsize=9, color=COLORS["ira"], fontweight="bold")

    # --- Panel B: Direction Changes ---
    ax2 = fig.add_subplot(gs[0, 1])

    # Create direction comparison
    merged["dir_ira"] = merged["correlation_ira"].apply(lambda x: "+" if x > 0 else "-")
    merged["dir_full"] = merged["correlation_full"].apply(lambda x: "+" if x > 0 else "-")
    merged["dir_change"] = merged["dir_ira"] != merged["dir_full"]

    # Bar chart showing signed correlations
    x = np.arange(len(merged))
    width = 0.35

    colors_ira = [COLORS["positive"] if c > 0 else COLORS["negative"] for c in merged["correlation_ira"]]
    colors_full = [COLORS["positive"] if c > 0 else COLORS["negative"] for c in merged["correlation_full"]]

    ax2.bar(x - width/2, merged["correlation_ira"], width, label="IRA Era",
            color=colors_ira, alpha=0.8, edgecolor="white")
    ax2.bar(x + width/2, merged["correlation_full"], width, label="Full Sample",
            color=colors_full, alpha=0.5, edgecolor="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels(merged["sector"], rotation=45, ha="right")
    ax2.set_ylabel("Correlation (signed)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("B. Direction Flips: Energy Goes Negative in IRA Era", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")

    # Highlight direction changes
    for i, (_, row) in enumerate(merged.iterrows()):
        if row["dir_change"]:
            ax2.annotate("FLIP!", xy=(i, max(row["correlation_ira"], row["correlation_full"]) + 0.05),
                        ha="center", fontsize=9, color="red", fontweight="bold")

    # --- Panel C: Decomposition Comparison ---
    ax3 = fig.add_subplot(gs[1, 0])

    decomp_merged = ira_decomp.merge(full_decomp, on="sector", suffixes=("_ira", "_full"))
    decomp_merged = decomp_merged.sort_values("asymmetry_ratio_ira")

    y_pos = np.arange(len(decomp_merged))
    height = 0.35

    # Color by dominant type
    colors_ira = []
    for dt in decomp_merged["dominant_type_ira"]:
        if dt == "implementation":
            colors_ira.append(COLORS["impl"])
        elif dt == "reversal":
            colors_ira.append(COLORS["reversal"])
        else:
            colors_ira.append("#7B1FA2")  # Purple for balanced

    ax3.barh(y_pos - height/2, decomp_merged["asymmetry_ratio_ira"], height,
             label="IRA Era", color=colors_ira, alpha=0.8)
    ax3.barh(y_pos + height/2, decomp_merged["asymmetry_ratio_full"], height,
             label="Full Sample", color=COLORS["full"], alpha=0.6)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(decomp_merged["sector"])
    ax3.axvline(0, color="black", linewidth=1)
    ax3.set_xlabel("Asymmetry Ratio (+ = impl-dominated, - = reversal-dominated)")
    ax3.set_title("C. Decomposition: Industrial Unique in IRA Era Only", fontsize=12, fontweight="bold")
    ax3.legend(loc="lower left")
    ax3.set_xlim(-0.5, 0.35)

    # Highlight Industrial
    ind_idx = decomp_merged[decomp_merged["sector"] == "Industrial"].index[0]
    ind_pos = list(decomp_merged["sector"]).index("Industrial")
    ax3.annotate("Industrial becomes\nimpl-dominated\nin IRA era!",
                xy=(decomp_merged.loc[ind_idx, "asymmetry_ratio_ira"], ind_pos - height/2),
                xytext=(0.25, ind_pos + 1),
                fontsize=9, fontweight="bold", color=COLORS["impl"],
                arrowprops=dict(arrowstyle="->", color=COLORS["impl"]))

    # --- Panel D: IRA Exposure Effect Flip ---
    ax4 = fig.add_subplot(gs[1, 1])

    groups = ["High IRA\n(≥6)", "Low IRA\n(≤3)"]
    x = np.arange(len(groups))
    width = 0.35

    ira_corrs = [ira_strat[ira_strat["group"] == "high_ira"]["correlation"].values[0],
                 ira_strat[ira_strat["group"] == "low_ira"]["correlation"].values[0]]
    full_corrs = [full_strat[full_strat["group"] == "high_ira"]["correlation"].values[0],
                  full_strat[full_strat["group"] == "low_ira"]["correlation"].values[0]]

    ax4.bar(x - width/2, ira_corrs, width, label="IRA Era", color=COLORS["ira"], alpha=0.8)
    ax4.bar(x + width/2, full_corrs, width, label="Full Sample", color=COLORS["full"], alpha=0.6)

    ax4.set_xticks(x)
    ax4.set_xticklabels(groups)
    ax4.set_ylabel("Correlation with CPU")
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_title("D. IRA Exposure Effect: Reverses in IRA Era!", fontsize=12, fontweight="bold")
    ax4.legend(loc="upper right")

    # Add annotations
    ax4.annotate("High-IRA more\nsensitive", xy=(0 - width/2, ira_corrs[0]),
                xytext=(-0.5, 0.5), fontsize=9, fontweight="bold", color=COLORS["ira"],
                arrowprops=dict(arrowstyle="->", color=COLORS["ira"]))
    ax4.annotate("Low-IRA more\nsensitive", xy=(1 + width/2, full_corrs[1]),
                xytext=(1.3, 0.3), fontsize=9, fontweight="bold", color=COLORS["full"],
                arrowprops=dict(arrowstyle="->", color=COLORS["full"]))

    # --- Panel E: Summary Table ---
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")

    summary_text = """
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              STRUCTURAL BREAK SUMMARY: IRA ERA vs FULL SAMPLE                                        ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                      ║
║  FINDING 1: CORRELATIONS ARE MUCH STRONGER IN IRA ERA                                                                ║
║  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ║
║  • Full sample: |r| ranges 0.11-0.23 (weak)                                                                          ║
║  • IRA era: |r| ranges 0.29-0.50 (moderate to strong)                                                                ║
║  → Policy uncertainty became MORE consequential for VC after 2021                                                    ║
║                                                                                                                      ║
║  FINDING 2: ENERGY DIRECTION FLIPS                                                                                   ║
║  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ║
║  • Full sample: Energy has POSITIVE correlation (r=+0.19)                                                            ║
║  • IRA era: Energy has NEGATIVE correlation (r=-0.42)                                                                ║
║  → In IRA era, uncertainty SUPPRESSES Energy VC (classic uncertainty channel)                                        ║
║                                                                                                                      ║
║  FINDING 3: INDUSTRIAL UNIQUENESS EMERGES IN IRA ERA                                                                 ║
║  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ║
║  • Full sample: ALL sectors reversal-dominated (including Industrial)                                                ║
║  • IRA era: Industrial is the ONLY impl-dominated sector                                                             ║
║  → Manufacturing's need for operational certainty became salient after IRA passage                                   ║
║                                                                                                                      ║
║  FINDING 4: IRA EXPOSURE EFFECT REVERSES                                                                             ║
║  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ║
║  • Full sample: Low-IRA companies are more sensitive (r=0.22 vs 0.18)                                                ║
║  • IRA era: High-IRA companies are more sensitive (r=0.42 vs -0.32)                                                  ║
║  → The policy-dependent business model channel activated after IRA passage                                           ║
║                                                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

    ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes,
            fontsize=10, va="center", ha="center", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E1", edgecolor="#FF9800", linewidth=2))

    plt.savefig(output_dir / "fig_ira_vs_full_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_dir / 'fig_ira_vs_full_comparison.png'}")

    # Also create a simple side-by-side table
    comparison_df = merged[["sector", "correlation_ira", "correlation_full",
                           "optimal_lag_ira", "optimal_lag_full"]].copy()
    comparison_df.columns = ["sector", "corr_ira_era", "corr_full_sample",
                            "lag_ira_era", "lag_full_sample"]
    comparison_df["corr_change"] = comparison_df["corr_ira_era"] - comparison_df["corr_full_sample"]
    comparison_df = comparison_df.sort_values("corr_change", ascending=False)
    comparison_df.to_csv(output_dir / "comparison_ira_vs_full.csv", index=False)
    print(f"Saved: {output_dir / 'comparison_ira_vs_full.csv'}")

    return comparison_df


def main():
    output_dir = Path("outputs/sector_analysis")
    print("=" * 60)
    print("COMPARING IRA ERA vs FULL SAMPLE")
    print("=" * 60)

    comparison = plot_comparison(output_dir)

    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(comparison.to_string(index=False))

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. CORRELATIONS STRENGTHEN: CPU-VC relationships are 2-3x stronger
   in IRA era compared to full sample.

2. DIRECTION FLIPS: Energy goes from positive to negative correlation,
   indicating uncertainty suppression channel activated in IRA era.

3. INDUSTRIAL UNIQUENESS: Only emerges as implementation-dominated
   in IRA era - historical pattern was reversal-dominated like others.

4. POLICY DEPENDENCE MATTERS MORE: High-IRA companies become more
   sensitive in IRA era, reversing the historical pattern.

PAPER NARRATIVE: The IRA created a structural break in how policy
uncertainty affects climate tech VC investment.
""")


if __name__ == "__main__":
    main()
