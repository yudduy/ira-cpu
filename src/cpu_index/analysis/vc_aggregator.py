"""
VC Deal Aggregator

Aggregates VC deal data by month for time series analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path

from .vc_loader import load_vc_deals


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate VC deals by month.

    Args:
        df: DataFrame from load_vc_deals()

    Returns:
        DataFrame with monthly aggregates indexed by YearMonth
    """
    # Group by YearMonth
    monthly = df.groupby('YearMonth').agg(
        deal_count=('Deal ID', 'count'),
        total_amount=('Deal Size', 'sum'),
        median_amount=('Deal Size', 'median'),
        mean_amount=('Deal Size', 'mean'),
        deals_with_size=('Deal Size', lambda x: x.notna().sum()),
        seed_count=('Stage', lambda x: (x == 'Seed').sum()),
        early_count=('Stage', lambda x: (x == 'Early').sum()),
        late_count=('Stage', lambda x: (x == 'Late').sum())
    )

    # Calculate size coverage percentage
    monthly['size_coverage_pct'] = 100 * monthly['deals_with_size'] / monthly['deal_count']

    # Convert index to datetime for easier merging
    monthly.index = monthly.index.to_timestamp()

    # Fill any gaps in the time series with zeros
    full_range = pd.date_range(
        start=monthly.index.min(),
        end=monthly.index.max(),
        freq='MS'  # Month Start
    )
    monthly = monthly.reindex(full_range, fill_value=0)
    monthly.index.name = 'month'

    return monthly


def aggregate_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate VC deals by month and sector.

    Args:
        df: DataFrame from load_vc_deals()

    Returns:
        DataFrame with monthly sector aggregates
    """
    sector_monthly = df.groupby(['YearMonth', 'Primary Industry Sector']).agg(
        deal_count=('Deal ID', 'count'),
        total_amount=('Deal Size', 'sum')
    ).reset_index()

    sector_monthly['YearMonth'] = sector_monthly['YearMonth'].dt.to_timestamp()

    return sector_monthly


def get_monthly_metrics(
    csv_path: str | Path,
    min_date: str = '2008-01-01'
) -> pd.DataFrame:
    """
    Load VC data and return monthly aggregated metrics.

    Args:
        csv_path: Path to ClimateTech_Deals.csv
        min_date: Minimum deal date to include

    Returns:
        DataFrame with monthly VC metrics
    """
    df = load_vc_deals(csv_path, min_date=min_date)
    return aggregate_monthly(df)


def create_analysis_dataset(
    vc_monthly: pd.DataFrame,
    cpu_index: pd.DataFrame,
    cpu_column: str = 'normalized_index'
) -> pd.DataFrame:
    """
    Merge VC metrics with CPU index for correlation analysis.

    Args:
        vc_monthly: Monthly VC metrics from aggregate_monthly()
        cpu_index: CPU index data with 'month' column
        cpu_column: Column name for CPU index values

    Returns:
        Merged DataFrame ready for correlation analysis
    """
    # Ensure CPU index has datetime index
    if 'month' in cpu_index.columns:
        cpu_df = cpu_index.copy()
        cpu_df['month'] = pd.to_datetime(cpu_df['month'])
        cpu_df = cpu_df.set_index('month')
    else:
        cpu_df = cpu_index.copy()

    # Select CPU column
    if cpu_column in cpu_df.columns:
        cpu_series = cpu_df[[cpu_column]].rename(columns={cpu_column: 'cpu_index'})
    else:
        raise ValueError(f"Column '{cpu_column}' not found in CPU index data")

    # Merge on month index
    merged = vc_monthly.join(cpu_series, how='inner')

    return merged


def compute_rolling_stats(
    df: pd.DataFrame,
    window: int = 12,
    columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Compute rolling statistics for VC metrics.

    Args:
        df: Monthly metrics DataFrame
        window: Rolling window size in months
        columns: Columns to compute rolling stats for

    Returns:
        DataFrame with rolling mean and std columns added
    """
    if columns is None:
        columns = ['deal_count', 'total_amount']

    result = df.copy()

    for col in columns:
        if col in result.columns:
            result[f'{col}_rolling_mean'] = result[col].rolling(window=window).mean()
            result[f'{col}_rolling_std'] = result[col].rolling(window=window).std()

    return result


if __name__ == '__main__':
    from pathlib import Path

    csv_path = Path(__file__).parent.parent.parent.parent / 'ClimateTech_Deals.csv'

    if csv_path.exists():
        monthly = get_monthly_metrics(csv_path)

        print("=== Monthly VC Metrics ===")
        print(f"Months covered: {len(monthly)}")
        print(f"Date range: {monthly.index.min()} to {monthly.index.max()}")
        print("\nRecent months:")
        print(monthly.tail(12)[['deal_count', 'total_amount', 'median_amount', 'size_coverage_pct']])

        print("\nSummary statistics:")
        print(monthly[['deal_count', 'total_amount', 'median_amount']].describe())
    else:
        print(f"CSV not found: {csv_path}")
