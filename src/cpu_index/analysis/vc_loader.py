"""
VC Deal Data Loader

Loads and preprocesses ClimateTech/CleanTech VC deal data from PitchBook exports.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime


# VC-related deal types to include
VC_DEAL_TYPES = [
    'Seed Round',
    'Early Stage VC',
    'Later Stage VC',
    'Angel (individual)'
]

# Minimum date for analysis (to match CPU index availability)
MIN_DATE = '2008-01-01'


def load_vc_deals(
    csv_path: str | Path,
    min_date: str = MIN_DATE,
    deal_types: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Load and preprocess VC deal data from PitchBook CSV export.

    Args:
        csv_path: Path to ClimateTech_Deals.csv
        min_date: Minimum deal date to include (default: 2008-01-01)
        deal_types: List of deal types to include (default: VC_DEAL_TYPES)

    Returns:
        DataFrame with preprocessed VC deals
    """
    if deal_types is None:
        deal_types = VC_DEAL_TYPES

    # Load CSV, skip 6 header rows
    df = pd.read_csv(csv_path, skiprows=6, low_memory=False)

    # Select key columns
    columns_to_keep = [
        'Deal ID', 'Companies', 'Company ID',
        'Deal Date', 'Announced Date',
        'Deal Size', 'Deal Size Status',
        'Deal Type', 'Deal Type 2', 'Deal Type 3',
        'VC Round', 'Series',
        'Primary Industry Sector', 'Primary Industry Group',
        'Verticals', 'Keywords',
        'Deal Status', 'Current Financing Status',
        'HQ Location', 'Company State/Province', 'Company Country/Territory/Region',
        'Year Founded'
    ]

    # Keep only columns that exist
    existing_cols = [c for c in columns_to_keep if c in df.columns]
    df = df[existing_cols].copy()

    # Parse dates
    df['Deal Date'] = pd.to_datetime(df['Deal Date'], errors='coerce')

    # Filter by date
    df = df[df['Deal Date'] >= min_date]

    # Filter by deal type
    df = df[df['Deal Type'].isin(deal_types)]

    # Filter to completed deals
    df = df[df['Deal Status'] == 'Completed']

    # Parse deal size (remove commas, convert to float)
    df['Deal Size'] = pd.to_numeric(
        df['Deal Size'].astype(str).str.replace(',', ''),
        errors='coerce'
    )

    # Add derived columns
    df['Year'] = df['Deal Date'].dt.year
    df['Month'] = df['Deal Date'].dt.month
    df['YearMonth'] = df['Deal Date'].dt.to_period('M')

    # Categorize stage
    df['Stage'] = df['Deal Type'].map({
        'Seed Round': 'Seed',
        'Angel (individual)': 'Seed',
        'Early Stage VC': 'Early',
        'Later Stage VC': 'Late'
    })

    # Reset index
    df = df.reset_index(drop=True)

    return df


def get_deal_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for loaded VC deals.

    Args:
        df: DataFrame from load_vc_deals()

    Returns:
        Dictionary with summary statistics
    """
    deal_size_valid = df['Deal Size'].notna()

    return {
        'total_deals': len(df),
        'date_range': {
            'start': df['Deal Date'].min().strftime('%Y-%m-%d'),
            'end': df['Deal Date'].max().strftime('%Y-%m-%d')
        },
        'deals_with_size': deal_size_valid.sum(),
        'size_coverage_pct': 100 * deal_size_valid.sum() / len(df),
        'deal_types': df['Deal Type'].value_counts().to_dict(),
        'stages': df['Stage'].value_counts().to_dict(),
        'top_sectors': df['Primary Industry Sector'].value_counts().head(10).to_dict(),
        'deals_by_year': df.groupby('Year').size().to_dict()
    }


if __name__ == '__main__':
    # Test loading
    import sys

    csv_path = Path(__file__).parent.parent.parent.parent / 'ClimateTech_Deals.csv'

    if csv_path.exists():
        df = load_vc_deals(csv_path)
        summary = get_deal_summary(df)

        print("=== VC Deal Data Summary ===")
        print(f"Total VC deals: {summary['total_deals']:,}")
        print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"Deals with size data: {summary['deals_with_size']:,} ({summary['size_coverage_pct']:.1f}%)")
        print("\nDeals by type:")
        for dtype, count in summary['deal_types'].items():
            print(f"  {dtype}: {count:,}")
        print("\nDeals by stage:")
        for stage, count in summary['stages'].items():
            print(f"  {stage}: {count:,}")
    else:
        print(f"CSV not found: {csv_path}")
        sys.exit(1)
