#!/usr/bin/env python3
"""
Sector-Specific CPU-VC Correlation Analysis

Identifies which climate tech sectors are most sensitive to Climate Policy Uncertainty (CPU),
measuring the "dark spots" where VC investment is most affected by policy uncertainty.

Outputs:
- sector_correlations.csv: Full correlation results by sector
- sector_rankings.csv: Sectors ranked by CPU sensitivity
- ira_stratification.csv: High-IRA vs Low-IRA comparison
- decomposition.csv: CPU_impl vs CPU_reversal sensitivity
- PNG visualizations: heatmap, timeseries, ranking, decomposition
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cpu_index.analysis.sector_analysis import (
    analyze_sector_cpu_correlation,
    analyze_all_sectors,
    analyze_cpu_decomposition,
    stratify_by_ira_exposure,
    run_classifier_robustness,
)
from cpu_index.analysis.sector_visualizations import (
    plot_sector_correlation_heatmap,
    plot_sector_timeseries,
    plot_sensitivity_ranking,
    plot_decomposition_comparison,
    save_sector_visualizations,
)


def load_cpu_index(csv_path: str | Path) -> pd.DataFrame:
    """Load CPU index data from CSV."""
    df = pd.read_csv(csv_path)
    df['month'] = pd.to_datetime(df['month'])
    return df


def load_judged_results(csv_path: str | Path) -> pd.DataFrame:
    """Load judged VC results with classifications."""
    df = pd.read_csv(csv_path, low_memory=False)
    # Clean numeric columns
    df['Total Raised'] = pd.to_numeric(df['Total Raised'], errors='coerce')
    df['IRA_Index'] = pd.to_numeric(df['IRA_Index'], errors='coerce')
    return df


def prepare_sector_monthly_data(
    vc_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    category_column: str = 'judge_category',
) -> dict:
    """Prepare monthly VC data by sector for visualization."""
    vc = vc_df.copy()
    vc['Last Financing Date'] = pd.to_datetime(vc['Last Financing Date'], errors='coerce')
    vc = vc[(vc['Last Financing Date'] >= start_date) & (vc['Last Financing Date'] <= end_date)]
    vc['YearMonth'] = vc['Last Financing Date'].dt.to_period('M')

    sectors = vc[category_column].dropna().unique()
    sector_monthly = {}

    for sector in sectors:
        if sector == 'Others':
            continue
        sector_vc = vc[vc[category_column] == sector]
        monthly = sector_vc.groupby('YearMonth').agg(
            deal_count=('Company ID', 'count')
        )
        monthly.index = monthly.index.to_timestamp()

        # Fill gaps
        full_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        monthly = monthly.reindex(full_range, fill_value=0)
        sector_monthly[sector] = monthly['deal_count']

    return sector_monthly


def run_full_sector_analysis(
    cpu_path: str,
    vc_path: str,
    output_dir: str,
    start_date: str = '2021-01-01',
    vc_metric: str = 'deal_count',
):
    """Run complete sector-specific CPU-VC correlation analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SECTOR-SPECIFIC CPU-VC CORRELATION ANALYSIS")
    print("=" * 70)
    print(f"Start date: {start_date}")
    print(f"VC metric: {vc_metric}")
    print(f"Output directory: {output_dir}")

    # Load data
    print("\n1. Loading data...")
    cpu_df = load_cpu_index(cpu_path)
    vc_df = load_judged_results(vc_path)

    print(f"   CPU: {len(cpu_df)} months ({cpu_df['month'].min()} to {cpu_df['month'].max()})")
    print(f"   VC: {len(vc_df):,} companies")

    # Category distribution
    print("\n   Sector distribution:")
    for cat, count in vc_df['judge_category'].value_counts().items():
        print(f"     {cat}: {count:,}")

    # Analyze all sectors
    print("\n2. Running sector correlation analysis...")
    ranking_df = analyze_all_sectors(
        cpu_df, vc_df,
        cpu_column='cpu_index',
        vc_metric=vc_metric,
        start_date=start_date,
    )
    print("\n   Sector rankings by CPU sensitivity:")
    print(ranking_df[['sector', 'correlation', 'optimal_lag', 'direction']].to_string(index=False))

    # Save rankings
    ranking_path = output_dir / "sector_rankings.csv"
    ranking_df.to_csv(ranking_path, index=False)
    print(f"\n   Saved: {ranking_path}")

    # CPU decomposition analysis
    print("\n3. Analyzing CPU decomposition (impl vs reversal)...")
    decomp_df = analyze_cpu_decomposition(
        cpu_df, vc_df,
        vc_metric=vc_metric,
        start_date=start_date,
    )
    print("\n   Decomposition results:")
    print(decomp_df[['sector', 'impl_correlation', 'reversal_correlation', 'asymmetry_ratio', 'dominant_type']].to_string(index=False))

    decomp_path = output_dir / "decomposition.csv"
    decomp_df.to_csv(decomp_path, index=False)
    print(f"\n   Saved: {decomp_path}")

    # IRA exposure stratification
    print("\n4. Stratifying by IRA exposure...")
    ira_results = stratify_by_ira_exposure(
        cpu_df, vc_df,
        cpu_column='cpu_index',
        vc_metric=vc_metric,
        start_date=start_date,
    )
    high_corr = ira_results['comparison']['high_correlation']
    low_corr = ira_results['comparison']['low_correlation']
    high_corr_str = f"{high_corr:.3f}" if high_corr is not None else 'N/A'
    low_corr_str = f"{low_corr:.3f}" if low_corr is not None else 'N/A'
    print(f"   High IRA (>=6): {ira_results['comparison']['high_n_companies']:,} companies, r={high_corr_str}")
    print(f"   Low IRA (<=3): {ira_results['comparison']['low_n_companies']:,} companies, r={low_corr_str}")

    ira_df = pd.DataFrame([{
        'group': 'high_ira',
        'threshold': '>=6',
        'n_companies': ira_results['comparison']['high_n_companies'],
        'correlation': ira_results['comparison']['high_correlation'],
        'optimal_lag': ira_results['high_ira']['optimal_lag'],
    }, {
        'group': 'low_ira',
        'threshold': '<=3',
        'n_companies': ira_results['comparison']['low_n_companies'],
        'correlation': ira_results['comparison']['low_correlation'],
        'optimal_lag': ira_results['low_ira']['optimal_lag'],
    }])
    ira_path = output_dir / "ira_stratification.csv"
    ira_df.to_csv(ira_path, index=False)
    print(f"\n   Saved: {ira_path}")

    # Prepare data for visualizations
    print("\n5. Generating visualizations...")

    # Get sector-level correlation details for heatmap
    sector_results = {}
    cpu = cpu_df.copy()
    cpu['month'] = pd.to_datetime(cpu['month'])
    cpu = cpu.set_index('month')
    cpu = cpu[cpu.index >= start_date]

    sector_monthly = prepare_sector_monthly_data(
        vc_df, start_date, cpu.index.max().strftime('%Y-%m-%d')
    )

    for sector, vc_series in sector_monthly.items():
        result = analyze_sector_cpu_correlation(
            cpu['cpu_index'], vc_series, sector
        )
        sector_results[sector] = result

    # Save all visualizations
    saved_figs = save_sector_visualizations(
        sector_results=sector_results,
        ranking_df=ranking_df,
        decomposition_df=decomp_df,
        cpu_df=cpu_df,
        sector_vc_monthly=sector_monthly,
        output_dir=output_dir,
    )

    print(f"   Saved figures:")
    for name, path in saved_figs.items():
        print(f"     - {path.name}")

    # Robustness check (classifier comparison)
    print("\n6. Running classifier robustness check...")
    robustness = run_classifier_robustness(
        cpu_df, vc_df,
        cpu_column='cpu_index',
        vc_metric=vc_metric,
        start_date=start_date,
    )
    agreement = robustness['agreement_score']
    agreement_str = f"{agreement:.2f}" if agreement is not None else 'N/A'
    print(f"   Classifier agreement score: {agreement_str}")

    # Save robustness results
    robustness_data = []
    for classifier, result in robustness['by_classifier'].items():
        if isinstance(result, pd.DataFrame):
            for _, row in result.iterrows():
                robustness_data.append({
                    'classifier': classifier,
                    'sector': row['sector'],
                    'correlation': row['correlation'],
                    'optimal_lag': row['optimal_lag'],
                })
    if robustness_data:
        robustness_df = pd.DataFrame(robustness_data)
        robustness_path = output_dir / "classifier_robustness.csv"
        robustness_df.to_csv(robustness_path, index=False)
        print(f"   Saved: {robustness_path}")

    # Summary report
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nKEY FINDINGS:")
    print("-" * 40)

    # Most sensitive sectors
    if not ranking_df.empty and ranking_df['correlation'].notna().any():
        top = ranking_df.iloc[0]
        print(f"1. Most CPU-sensitive sector: {top['sector']}")
        print(f"   Correlation: {top['correlation']:.3f} ({top['direction']})")
        print(f"   Optimal lag: {top['optimal_lag']} months")

    # Decomposition insight
    impl_dom = decomp_df[decomp_df['dominant_type'] == 'implementation']['sector'].tolist()
    rev_dom = decomp_df[decomp_df['dominant_type'] == 'reversal']['sector'].tolist()
    print(f"\n2. Implementation-dominated sectors: {', '.join(impl_dom) if impl_dom else 'None'}")
    print(f"   Reversal-dominated sectors: {', '.join(rev_dom) if rev_dom else 'None'}")

    # IRA exposure
    if ira_results['comparison']['high_more_sensitive'] is not None:
        if ira_results['comparison']['high_more_sensitive']:
            print("\n3. High-IRA sectors are MORE sensitive to CPU")
        else:
            print("\n3. Low-IRA sectors are MORE sensitive to CPU")

    print(f"\nOutput files saved to: {output_dir}")

    return {
        'rankings': ranking_df,
        'decomposition': decomp_df,
        'ira_stratification': ira_results,
        'robustness': robustness,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sector-specific CPU-VC correlation analysis')
    parser.add_argument('--cpu', default='data/cpu_index.csv', help='Path to CPU index CSV')
    parser.add_argument('--vc', default='data/judged_results.csv', help='Path to judged results CSV')
    parser.add_argument('--output', default='outputs/sector_analysis', help='Output directory')
    parser.add_argument('--start-date', default='2021-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--metric', choices=['deal_count', 'total_amount'], default='deal_count',
                       help='VC metric to analyze')

    args = parser.parse_args()

    run_full_sector_analysis(
        cpu_path=args.cpu,
        vc_path=args.vc,
        output_dir=args.output,
        start_date=args.start_date,
        vc_metric=args.metric,
    )
