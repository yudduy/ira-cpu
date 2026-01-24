#!/usr/bin/env python3
"""
CPU-VC Correlation Analysis

Maps Climate Policy Uncertainty (CPU) index decomposition with
Venture Capital financing trends in CleanTech/ClimateTech.

Outputs:
- Correlation analysis results (JSON)
- Visualizations (PNG)
- Summary report (Markdown)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cpu_index.analysis.vc_loader import load_vc_deals, get_deal_summary
from cpu_index.analysis.vc_aggregator import aggregate_monthly, create_analysis_dataset
from cpu_index.analysis.correlation import (
    analyze_cpu_vc_correlation,
    cross_correlation,
    find_optimal_lag,
)
from cpu_index.analysis.vc_visualizations import save_all_visualizations


def load_cpu_index(csv_path: str | Path) -> pd.DataFrame:
    """Load CPU index data from CSV."""
    df = pd.read_csv(csv_path)
    df['month'] = pd.to_datetime(df['month'])
    df = df.set_index('month')
    return df


def run_full_analysis(cpu_path: str, vc_path: str, output_dir: str):
    """Run complete CPU-VC correlation analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CPU-VC CORRELATION ANALYSIS")
    print("=" * 60)

    # Load CPU index
    print("\n1. Loading CPU Index data...")
    cpu_df = load_cpu_index(cpu_path)
    print(f"   CPU data: {len(cpu_df)} months ({cpu_df.index.min().strftime('%Y-%m')} to {cpu_df.index.max().strftime('%Y-%m')})")
    print(f"   Columns: {', '.join(cpu_df.columns)}")

    # Load VC data
    print("\n2. Loading VC deal data...")
    vc_deals = load_vc_deals(vc_path)
    vc_summary = get_deal_summary(vc_deals)
    print(f"   VC deals: {vc_summary['total_deals']:,}")
    print(f"   Date range: {vc_summary['date_range']['start']} to {vc_summary['date_range']['end']}")

    # Aggregate VC monthly
    print("\n3. Aggregating VC metrics by month...")
    vc_monthly = aggregate_monthly(vc_deals)
    print(f"   Monthly observations: {len(vc_monthly)}")

    # Create merged dataset
    print("\n4. Creating analysis dataset...")

    # Align indices
    cpu_df.index = pd.to_datetime(cpu_df.index)
    vc_monthly.index = pd.to_datetime(vc_monthly.index)

    # Find overlapping period
    start_date = max(cpu_df.index.min(), vc_monthly.index.min())
    end_date = min(cpu_df.index.max(), vc_monthly.index.max())

    cpu_aligned = cpu_df[(cpu_df.index >= start_date) & (cpu_df.index <= end_date)]
    vc_aligned = vc_monthly[(vc_monthly.index >= start_date) & (vc_monthly.index <= end_date)]

    # Merge
    merged = vc_aligned.join(cpu_aligned[['cpu_index', 'cpu_impl', 'cpu_reversal']], how='inner')
    print(f"   Merged observations: {len(merged)}")
    print(f"   Period: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")

    # Save merged dataset
    merged.to_csv(output_dir / "merged_cpu_vc_monthly.csv")

    # Run correlation analysis
    print("\n5. Running correlation analysis...")
    results = {}

    # CPU vs Deal Count
    print("\n   5a. CPU Index vs Deal Count")
    ccf_cpu_count = cross_correlation(merged['cpu_index'], merged['deal_count'])
    optimal_cpu_count = find_optimal_lag(ccf_cpu_count)
    results['cpu_vs_deal_count'] = {
        'contemporaneous_corr': ccf_cpu_count[ccf_cpu_count['lag'] == 0]['correlation'].values[0],
        'optimal_lag': optimal_cpu_count,
    }
    print(f"      Contemporaneous correlation: {results['cpu_vs_deal_count']['contemporaneous_corr']:.4f}")
    print(f"      Optimal lag: {optimal_cpu_count['optimal_lag']} months (r={optimal_cpu_count['max_correlation']:.4f})")

    # CPU_impl vs Deal Count
    print("\n   5b. CPU Implementation vs Deal Count")
    ccf_impl_count = cross_correlation(merged['cpu_impl'], merged['deal_count'])
    optimal_impl_count = find_optimal_lag(ccf_impl_count)
    results['cpu_impl_vs_deal_count'] = {
        'contemporaneous_corr': ccf_impl_count[ccf_impl_count['lag'] == 0]['correlation'].values[0],
        'optimal_lag': optimal_impl_count,
    }
    print(f"      Contemporaneous correlation: {results['cpu_impl_vs_deal_count']['contemporaneous_corr']:.4f}")
    print(f"      Optimal lag: {optimal_impl_count['optimal_lag']} months (r={optimal_impl_count['max_correlation']:.4f})")

    # CPU_reversal vs Deal Count
    print("\n   5c. CPU Reversal vs Deal Count")
    ccf_rev_count = cross_correlation(merged['cpu_reversal'], merged['deal_count'])
    optimal_rev_count = find_optimal_lag(ccf_rev_count)
    results['cpu_reversal_vs_deal_count'] = {
        'contemporaneous_corr': ccf_rev_count[ccf_rev_count['lag'] == 0]['correlation'].values[0],
        'optimal_lag': optimal_rev_count,
    }
    print(f"      Contemporaneous correlation: {results['cpu_reversal_vs_deal_count']['contemporaneous_corr']:.4f}")
    print(f"      Optimal lag: {optimal_rev_count['optimal_lag']} months (r={optimal_rev_count['max_correlation']:.4f})")

    # CPU vs Total Amount
    print("\n   5d. CPU Index vs Total Investment Amount")
    ccf_cpu_amount = cross_correlation(merged['cpu_index'], merged['total_amount'])
    optimal_cpu_amount = find_optimal_lag(ccf_cpu_amount)
    results['cpu_vs_total_amount'] = {
        'contemporaneous_corr': ccf_cpu_amount[ccf_cpu_amount['lag'] == 0]['correlation'].values[0],
        'optimal_lag': optimal_cpu_amount,
    }
    print(f"      Contemporaneous correlation: {results['cpu_vs_total_amount']['contemporaneous_corr']:.4f}")
    print(f"      Optimal lag: {optimal_cpu_amount['optimal_lag']} months (r={optimal_cpu_amount['max_correlation']:.4f})")

    # Save correlation results
    results['metadata'] = {
        'analysis_date': datetime.now().isoformat(),
        'n_observations': len(merged),
        'period_start': start_date.strftime('%Y-%m'),
        'period_end': end_date.strftime('%Y-%m'),
    }

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj

    results = convert_numpy(results)

    with open(output_dir / "correlation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   Results saved to: {output_dir / 'correlation_results.json'}")

    # Generate visualizations
    print("\n6. Generating visualizations...")

    # Prepare CCF results for visualization
    ccf_results = ccf_cpu_count.to_dict('records')

    try:
        viz_paths = save_all_visualizations(
            cpu_series=merged['cpu_index'],
            vc_monthly=merged,
            ccf_results={'results': ccf_results},
            output_dir=output_dir,
        )
        for name, path in viz_paths.items():
            print(f"   {name}: {path}")
    except Exception as e:
        print(f"   Warning: Some visualizations failed: {e}")

    # Generate summary report
    print("\n7. Generating summary report...")
    generate_report(results, merged, output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")

    return results


def generate_report(results: dict, merged: pd.DataFrame, output_dir: Path):
    """Generate markdown summary report."""
    report = f"""# CPU-VC Correlation Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Overview

Analysis of the relationship between Climate Policy Uncertainty (CPU) index
and Venture Capital financing in CleanTech/ClimateTech sectors.

- **Period:** {results['metadata']['period_start']} to {results['metadata']['period_end']}
- **Observations:** {results['metadata']['n_observations']} months

## Key Findings

### 1. CPU Index vs VC Deal Count

| Metric | Value |
|--------|-------|
| Contemporaneous Correlation | {results['cpu_vs_deal_count']['contemporaneous_corr']:.4f} |
| Optimal Lag | {results['cpu_vs_deal_count']['optimal_lag']['optimal_lag']} months |
| Correlation at Optimal Lag | {results['cpu_vs_deal_count']['optimal_lag']['max_correlation']:.4f} |
| Direction | {results['cpu_vs_deal_count']['optimal_lag']['direction']} |

### 2. CPU Implementation Uncertainty vs VC Deal Count

| Metric | Value |
|--------|-------|
| Contemporaneous Correlation | {results['cpu_impl_vs_deal_count']['contemporaneous_corr']:.4f} |
| Optimal Lag | {results['cpu_impl_vs_deal_count']['optimal_lag']['optimal_lag']} months |
| Correlation at Optimal Lag | {results['cpu_impl_vs_deal_count']['optimal_lag']['max_correlation']:.4f} |

### 3. CPU Reversal Uncertainty vs VC Deal Count

| Metric | Value |
|--------|-------|
| Contemporaneous Correlation | {results['cpu_reversal_vs_deal_count']['contemporaneous_corr']:.4f} |
| Optimal Lag | {results['cpu_reversal_vs_deal_count']['optimal_lag']['optimal_lag']} months |
| Correlation at Optimal Lag | {results['cpu_reversal_vs_deal_count']['optimal_lag']['max_correlation']:.4f} |

### 4. CPU Index vs Total Investment Amount

| Metric | Value |
|--------|-------|
| Contemporaneous Correlation | {results['cpu_vs_total_amount']['contemporaneous_corr']:.4f} |
| Optimal Lag | {results['cpu_vs_total_amount']['optimal_lag']['optimal_lag']} months |
| Correlation at Optimal Lag | {results['cpu_vs_total_amount']['optimal_lag']['max_correlation']:.4f} |

## Interpretation

### Correlation Direction
- **Negative correlation**: Higher policy uncertainty is associated with *lower* VC activity
- **Positive correlation**: Higher policy uncertainty is associated with *higher* VC activity

### Lag Interpretation
- **Negative lag** (e.g., -3): CPU *leads* VC by 3 months (CPU changes predict future VC)
- **Positive lag** (e.g., +3): CPU *lags* VC by 3 months (VC changes precede CPU)
- **Zero lag**: Contemporaneous relationship

## Data Summary

### CPU Index Statistics
| Metric | cpu_index | cpu_impl | cpu_reversal |
|--------|-----------|----------|--------------|
| Mean | {merged['cpu_index'].mean():.2f} | {merged['cpu_impl'].mean():.2f} | {merged['cpu_reversal'].mean():.2f} |
| Std | {merged['cpu_index'].std():.2f} | {merged['cpu_impl'].std():.2f} | {merged['cpu_reversal'].std():.2f} |
| Min | {merged['cpu_index'].min():.2f} | {merged['cpu_impl'].min():.2f} | {merged['cpu_reversal'].min():.2f} |
| Max | {merged['cpu_index'].max():.2f} | {merged['cpu_impl'].max():.2f} | {merged['cpu_reversal'].max():.2f} |

### VC Activity Statistics
| Metric | deal_count | total_amount ($M) |
|--------|------------|-------------------|
| Mean | {merged['deal_count'].mean():.1f} | {merged['total_amount'].mean():.1f} |
| Std | {merged['deal_count'].std():.1f} | {merged['total_amount'].std():.1f} |
| Min | {merged['deal_count'].min():.0f} | {merged['total_amount'].min():.1f} |
| Max | {merged['deal_count'].max():.0f} | {merged['total_amount'].max():.1f} |

## Files Generated

- `merged_cpu_vc_monthly.csv` - Combined monthly dataset
- `correlation_results.json` - Detailed correlation analysis
- `cpu_vc_timeseries.png` - Time series visualization
- `cross_correlation.png` - Cross-correlation plot
- `rolling_correlation.png` - Rolling correlation over time
- `stage_distribution.png` - VC stage distribution

---
*Generated by CPU-VC Correlation Analysis Tool*
"""

    with open(output_dir / "analysis_report.md", "w") as f:
        f.write(report)
    print(f"   Report saved to: {output_dir / 'analysis_report.md'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CPU-VC correlation analysis")
    parser.add_argument("--cpu-data", default="data/cpu_index.csv", help="Path to CPU index CSV")
    parser.add_argument("--vc-data", default="ClimateTech_Deals.csv", help="Path to VC deals CSV")
    parser.add_argument("--output-dir", default="outputs/cpu_vc_analysis", help="Output directory")
    args = parser.parse_args()

    run_full_analysis(args.cpu_data, args.vc_data, args.output_dir)
