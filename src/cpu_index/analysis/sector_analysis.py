"""
Sector-Specific CPU-VC Correlation Analysis

Analyzes which climate tech sectors are most sensitive to Climate Policy Uncertainty (CPU),
identifying "dark spots" where VC investment is most affected by policy uncertainty.

Reference:
- Noailly, Nowzohour & van den Heuvel (2022) for EnvPU-VC methodology
- Baker, Bloom & Davis (2016) for EPU index construction
"""

import warnings
from typing import Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd

from .correlation import cross_correlation, find_optimal_lag, MIN_OBSERVATIONS
from .vc_aggregator import aggregate_by_category_monthly, aggregate_by_subtopic_monthly


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_MAX_LAG = 12
DEFAULT_START_DATE = '2021-01-01'  # IRA era
FULL_SAMPLE_START = '2008-01-01'

# Sectors to exclude (too small for meaningful analysis)
EXCLUDED_SECTORS = {'Others'}

# IRA exposure thresholds
HIGH_IRA_THRESHOLD = 6
LOW_IRA_THRESHOLD = 3


# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================

def analyze_sector_cpu_correlation(
    cpu_series: pd.Series,
    sector_vc_series: pd.Series,
    sector_name: str,
    max_lag: int = DEFAULT_MAX_LAG,
) -> dict:
    """
    Compute cross-correlation between CPU and a single sector's VC activity.

    Args:
        cpu_series: CPU index series (datetime indexed)
        sector_vc_series: Sector VC metrics series (datetime indexed)
        sector_name: Name of the sector
        max_lag: Maximum lag for cross-correlation

    Returns:
        Dict with:
        - sector: Sector name
        - n_observations: Number of overlapping observations
        - ccf: DataFrame of cross-correlation at all lags
        - optimal_lag: Lag with strongest correlation
        - max_correlation: Correlation at optimal lag
        - direction: 'positive' or 'negative'
        - interpretation: Human-readable interpretation
        - warnings: List of any warnings
    """
    result = {
        'sector': sector_name,
        'n_observations': 0,
        'ccf': None,
        'optimal_lag': None,
        'max_correlation': None,
        'abs_correlation': None,
        'direction': None,
        'interpretation': None,
        'warnings': [],
    }

    # Align series
    aligned = pd.DataFrame({
        'cpu': cpu_series,
        'vc': sector_vc_series
    }).dropna()

    result['n_observations'] = len(aligned)

    if len(aligned) < MIN_OBSERVATIONS:
        result['warnings'].append(
            f"Insufficient observations ({len(aligned)} < {MIN_OBSERVATIONS})"
        )
        return result

    # Compute cross-correlation
    try:
        ccf = cross_correlation(aligned['cpu'], aligned['vc'], max_lag=max_lag)
        result['ccf'] = ccf

        # Find optimal lag
        optimal = find_optimal_lag(ccf)
        result['optimal_lag'] = optimal['optimal_lag']
        result['max_correlation'] = optimal['max_correlation']
        result['abs_correlation'] = abs(optimal['max_correlation']) if optimal['max_correlation'] else None
        result['direction'] = optimal['direction']

        # Generate interpretation
        if optimal['optimal_lag'] is not None:
            lag = optimal['optimal_lag']
            corr = optimal['max_correlation']
            if lag > 0:
                lead = "CPU leads VC"
            elif lag < 0:
                lead = "VC leads CPU"
            else:
                lead = "Contemporaneous"

            result['interpretation'] = (
                f"{sector_name}: {lead} by {abs(lag)} months, "
                f"r={corr:.3f} ({optimal['direction']})"
            )
    except Exception as e:
        result['warnings'].append(f"Correlation failed: {str(e)}")

    return result


def analyze_all_sectors(
    cpu_df: pd.DataFrame,
    vc_df: pd.DataFrame,
    cpu_column: str = 'cpu_index',
    vc_metric: str = 'deal_count',
    category_column: str = 'judge_category',
    start_date: str = DEFAULT_START_DATE,
    max_lag: int = DEFAULT_MAX_LAG,
    exclude_sectors: Optional[set] = None,
) -> pd.DataFrame:
    """
    Run correlation analysis across all sectors and rank by sensitivity.

    Args:
        cpu_df: CPU index DataFrame (must have 'month' column or datetime index)
        vc_df: VC deals DataFrame with category classifications
        cpu_column: Column name for CPU values
        vc_metric: 'deal_count' or 'total_amount'
        category_column: Column containing sector classification
        start_date: Start date for analysis (default: 2021-01-01 for IRA era)
        max_lag: Maximum lag for cross-correlation
        exclude_sectors: Set of sector names to exclude

    Returns:
        DataFrame with columns: sector, optimal_lag, correlation, abs_correlation,
        direction, n_observations, interpretation
    """
    if exclude_sectors is None:
        exclude_sectors = EXCLUDED_SECTORS

    # Prepare CPU series
    cpu = cpu_df.copy()
    if 'month' in cpu.columns:
        cpu['month'] = pd.to_datetime(cpu['month'])
        cpu = cpu.set_index('month')
    cpu = cpu[cpu.index >= start_date]
    cpu_series = cpu[cpu_column]

    # Prepare VC data - filter to start_date
    vc = vc_df.copy()
    vc['Last Financing Date'] = pd.to_datetime(vc['Last Financing Date'], errors='coerce')
    vc = vc[vc['Last Financing Date'] >= start_date]

    # Create YearMonth column for aggregation
    vc['YearMonth'] = vc['Last Financing Date'].dt.to_period('M')

    # Get unique sectors
    sectors = vc[category_column].dropna().unique()
    sectors = [s for s in sectors if s not in exclude_sectors]

    results = []
    for sector in sectors:
        # Filter to sector
        sector_vc = vc[vc[category_column] == sector]

        # Aggregate monthly
        monthly = sector_vc.groupby('YearMonth').agg(
            deal_count=('Company ID', 'count'),
            total_amount=('Total Raised', lambda x: pd.to_numeric(x, errors='coerce').sum())
        )
        monthly.index = monthly.index.to_timestamp()

        # Fill gaps
        full_range = pd.date_range(
            start=start_date,
            end=cpu_series.index.max(),
            freq='MS'
        )
        monthly = monthly.reindex(full_range, fill_value=0)

        # Run correlation analysis
        sector_series = monthly[vc_metric]
        analysis = analyze_sector_cpu_correlation(
            cpu_series, sector_series, sector, max_lag=max_lag
        )

        results.append({
            'sector': sector,
            'optimal_lag': analysis['optimal_lag'],
            'correlation': analysis['max_correlation'],
            'abs_correlation': analysis['abs_correlation'],
            'direction': analysis['direction'],
            'n_observations': analysis['n_observations'],
            'interpretation': analysis['interpretation'],
        })

    # Create DataFrame and sort by absolute correlation
    df = pd.DataFrame(results)
    df = df.sort_values('abs_correlation', ascending=False, na_position='last')
    df = df.reset_index(drop=True)

    return df


def analyze_cpu_decomposition(
    cpu_df: pd.DataFrame,
    vc_df: pd.DataFrame,
    category_column: str = 'judge_category',
    vc_metric: str = 'deal_count',
    start_date: str = DEFAULT_START_DATE,
    max_lag: int = DEFAULT_MAX_LAG,
    exclude_sectors: Optional[set] = None,
) -> pd.DataFrame:
    """
    Analyze sectors' differential response to CPU_impl vs CPU_reversal.

    Args:
        cpu_df: CPU index DataFrame with cpu_impl and cpu_reversal columns
        vc_df: VC deals DataFrame
        category_column: Column containing sector classification
        vc_metric: 'deal_count' or 'total_amount'
        start_date: Start date for analysis
        max_lag: Maximum lag for cross-correlation
        exclude_sectors: Sectors to exclude

    Returns:
        DataFrame with columns: sector, impl_correlation, impl_lag,
        reversal_correlation, reversal_lag, asymmetry_ratio, dominant_type
    """
    if exclude_sectors is None:
        exclude_sectors = EXCLUDED_SECTORS

    # Run analysis for each CPU type
    impl_results = analyze_all_sectors(
        cpu_df, vc_df,
        cpu_column='cpu_impl',
        vc_metric=vc_metric,
        category_column=category_column,
        start_date=start_date,
        max_lag=max_lag,
        exclude_sectors=exclude_sectors,
    )
    impl_results = impl_results.rename(columns={
        'correlation': 'impl_correlation',
        'optimal_lag': 'impl_lag',
        'abs_correlation': 'impl_abs',
    })[['sector', 'impl_correlation', 'impl_lag', 'impl_abs']]

    reversal_results = analyze_all_sectors(
        cpu_df, vc_df,
        cpu_column='cpu_reversal',
        vc_metric=vc_metric,
        category_column=category_column,
        start_date=start_date,
        max_lag=max_lag,
        exclude_sectors=exclude_sectors,
    )
    reversal_results = reversal_results.rename(columns={
        'correlation': 'reversal_correlation',
        'optimal_lag': 'reversal_lag',
        'abs_correlation': 'reversal_abs',
    })[['sector', 'reversal_correlation', 'reversal_lag', 'reversal_abs']]

    # Merge results
    merged = impl_results.merge(reversal_results, on='sector', how='outer')

    # Calculate asymmetry ratio: (|impl| - |reversal|) / (|impl| + |reversal|)
    # Positive = more impl-sensitive, Negative = more reversal-sensitive
    merged['asymmetry_ratio'] = np.where(
        (merged['impl_abs'].notna()) & (merged['reversal_abs'].notna()) &
        ((merged['impl_abs'] + merged['reversal_abs']) > 0),
        (merged['impl_abs'] - merged['reversal_abs']) /
        (merged['impl_abs'] + merged['reversal_abs']),
        np.nan
    )

    # Determine dominant type
    merged['dominant_type'] = np.where(
        merged['asymmetry_ratio'] > 0.1, 'implementation',
        np.where(merged['asymmetry_ratio'] < -0.1, 'reversal', 'balanced')
    )

    return merged.sort_values('asymmetry_ratio', ascending=True, na_position='last')


def stratify_by_ira_exposure(
    cpu_df: pd.DataFrame,
    vc_df: pd.DataFrame,
    cpu_column: str = 'cpu_index',
    vc_metric: str = 'deal_count',
    start_date: str = DEFAULT_START_DATE,
    max_lag: int = DEFAULT_MAX_LAG,
    high_threshold: int = HIGH_IRA_THRESHOLD,
    low_threshold: int = LOW_IRA_THRESHOLD,
) -> dict:
    """
    Compare CPU sensitivity between high-IRA and low-IRA exposure groups.

    Args:
        cpu_df: CPU index DataFrame
        vc_df: VC deals DataFrame (must have 'IRA_Index' column)
        cpu_column: Column name for CPU values
        vc_metric: 'deal_count' or 'total_amount'
        start_date: Start date for analysis
        max_lag: Maximum lag
        high_threshold: IRA_Index >= this is "high exposure" (default: 6)
        low_threshold: IRA_Index <= this is "low exposure" (default: 3)

    Returns:
        Dict with:
        - high_ira: Analysis results for high-exposure companies
        - low_ira: Analysis results for low-exposure companies
        - comparison: Statistical comparison
    """
    if 'IRA_Index' not in vc_df.columns:
        raise ValueError("VC DataFrame must have 'IRA_Index' column")

    # Prepare CPU series
    cpu = cpu_df.copy()
    if 'month' in cpu.columns:
        cpu['month'] = pd.to_datetime(cpu['month'])
        cpu = cpu.set_index('month')
    cpu = cpu[cpu.index >= start_date]
    cpu_series = cpu[cpu_column]

    # Prepare VC data
    vc = vc_df.copy()
    vc['Last Financing Date'] = pd.to_datetime(vc['Last Financing Date'], errors='coerce')
    vc = vc[vc['Last Financing Date'] >= start_date]
    vc['YearMonth'] = vc['Last Financing Date'].dt.to_period('M')

    def analyze_group(group_vc, group_name):
        """Analyze a single IRA exposure group."""
        monthly = group_vc.groupby('YearMonth').agg(
            deal_count=('Company ID', 'count'),
            total_amount=('Total Raised', lambda x: pd.to_numeric(x, errors='coerce').sum())
        )
        monthly.index = monthly.index.to_timestamp()

        full_range = pd.date_range(start=start_date, end=cpu_series.index.max(), freq='MS')
        monthly = monthly.reindex(full_range, fill_value=0)

        analysis = analyze_sector_cpu_correlation(
            cpu_series, monthly[vc_metric], group_name, max_lag=max_lag
        )
        return analysis

    # Analyze high and low IRA groups
    high_vc = vc[vc['IRA_Index'] >= high_threshold]
    low_vc = vc[vc['IRA_Index'] <= low_threshold]

    high_result = analyze_group(high_vc, f'High IRA (>={high_threshold})')
    low_result = analyze_group(low_vc, f'Low IRA (<={low_threshold})')

    # Comparison
    comparison = {
        'high_n_companies': len(high_vc['Company ID'].unique()),
        'low_n_companies': len(low_vc['Company ID'].unique()),
        'high_correlation': high_result['max_correlation'],
        'low_correlation': low_result['max_correlation'],
        'difference': None,
        'high_more_sensitive': None,
    }

    if high_result['abs_correlation'] and low_result['abs_correlation']:
        comparison['difference'] = high_result['abs_correlation'] - low_result['abs_correlation']
        comparison['high_more_sensitive'] = comparison['difference'] > 0

    return {
        'high_ira': high_result,
        'low_ira': low_result,
        'comparison': comparison,
    }


def run_classifier_robustness(
    cpu_df: pd.DataFrame,
    vc_df: pd.DataFrame,
    cpu_column: str = 'cpu_index',
    vc_metric: str = 'deal_count',
    start_date: str = DEFAULT_START_DATE,
    max_lag: int = DEFAULT_MAX_LAG,
) -> dict:
    """
    Run sector analysis using different classifiers for robustness check.

    Args:
        cpu_df: CPU index DataFrame
        vc_df: VC deals DataFrame (must have ChatGPT_Category, Gemini_Category,
               DeepSeek_Category, judge_category columns)
        cpu_column: Column name for CPU values
        vc_metric: 'deal_count' or 'total_amount'
        start_date: Start date for analysis
        max_lag: Maximum lag

    Returns:
        Dict with results for each classifier and agreement metrics
    """
    classifiers = {
        'judge_category': 'Consensus (Judge)',
        'ChatGPT_Category': 'ChatGPT',
        'Gemini_Category': 'Gemini',
        'DeepSeek_Category': 'DeepSeek',
    }

    results = {}
    for col, name in classifiers.items():
        if col not in vc_df.columns:
            results[name] = {'error': f'Column {col} not found'}
            continue

        analysis = analyze_all_sectors(
            cpu_df, vc_df,
            cpu_column=cpu_column,
            vc_metric=vc_metric,
            category_column=col,
            start_date=start_date,
            max_lag=max_lag,
        )
        results[name] = analysis

    # Calculate agreement metrics
    # Check if top 3 most sensitive sectors agree across classifiers
    valid_results = [r for r in results.values() if isinstance(r, pd.DataFrame)]
    if len(valid_results) >= 2:
        top3_sets = [set(r.head(3)['sector'].tolist()) for r in valid_results]
        # Jaccard similarity
        if len(top3_sets) >= 2:
            intersection = set.intersection(*top3_sets)
            union = set.union(*top3_sets)
            agreement = len(intersection) / len(union) if union else 0
        else:
            agreement = 1.0
    else:
        agreement = None

    return {
        'by_classifier': results,
        'agreement_score': agreement,
    }
