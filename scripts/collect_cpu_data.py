#!/usr/bin/env python3
"""
Collect CPU index data directly to CSV (no database required).
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from cpu_index.collection import api
from cpu_index import config


def collect_month_counts(year: int, month: int) -> dict:
    """Collect counts for a single month from LexisNexis API."""
    start_date, end_date = api.build_month_dates(year, month)
    date_filter = api.build_date_filter(start_date, end_date)

    # Build queries
    query_denom = api.build_search_query(
        climate_terms=config.CLIMATE_TERMS,
        policy_terms=config.POLICY_TERMS,
    )
    query_cpu = api.build_search_query(
        climate_terms=config.CLIMATE_TERMS,
        policy_terms=config.POLICY_TERMS,
        uncertainty_terms=config.UNCERTAINTY_TERMS,
    )
    query_impl = api.build_search_query(
        climate_terms=config.CLIMATE_TERMS,
        policy_terms=config.POLICY_TERMS,
        uncertainty_terms=config.UNCERTAINTY_TERMS,
        direction_terms=config.IMPLEMENTATION_TERMS,
    )
    query_reversal = api.build_search_query(
        climate_terms=config.CLIMATE_TERMS,
        policy_terms=config.POLICY_TERMS,
        uncertainty_terms=config.UNCERTAINTY_TERMS,
        direction_terms=config.REVERSAL_TERMS,
    )

    # Rate limit between API calls
    delay = getattr(config, 'REQUEST_DELAY', 0.5)

    denom = api.fetch_count(query_denom, date_filter)
    time.sleep(delay)

    cpu = api.fetch_count(query_cpu, date_filter)
    time.sleep(delay)

    impl = api.fetch_count(query_impl, date_filter)
    time.sleep(delay)

    reversal = api.fetch_count(query_reversal, date_filter)

    return {
        "month": f"{year}-{month:02d}",
        "denominator": denom,
        "numerator_cpu": cpu,
        "numerator_impl": impl,
        "numerator_reversal": reversal,
    }


def iter_months(start_year: int, start_month: int, end_year: int, end_month: int):
    """Yield (year, month) tuples for each month in range."""
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month > 12:
            month = 1
            year += 1


def collect_all(start_year=2008, start_month=1, end_year=2025, end_month=5):
    """Collect counts for all months and save to CSV."""
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "cpu_counts.csv"

    # Load existing data if available
    if output_file.exists():
        existing = pd.read_csv(output_file)
        existing_months = set(existing['month'].tolist())
        print(f"Found existing data with {len(existing_months)} months")
    else:
        existing = pd.DataFrame()
        existing_months = set()

    results = []
    months_list = list(iter_months(start_year, start_month, end_year, end_month))
    total = len(months_list)

    print(f"Collecting counts for {total} months...")
    print(f"API credentials: {'OK' if os.environ.get('clientid') else 'MISSING'}")

    # Test API connection first
    print("\nTesting API connection...")
    token = api.get_token()
    if not token:
        print("ERROR: Could not get API token. Check credentials.")
        return None
    print("API connection OK\n")

    for i, (year, month) in enumerate(months_list):
        month_str = f"{year}-{month:02d}"

        # Skip if already collected
        if month_str in existing_months:
            print(f"[{i+1}/{total}] {month_str} - already collected, skipping")
            continue

        print(f"[{i+1}/{total}] Collecting {month_str}...", end=" ", flush=True)

        try:
            counts = collect_month_counts(year, month)
            results.append(counts)
            print(f"denom={counts['denominator']}, cpu={counts['numerator_cpu']}, "
                  f"impl={counts['numerator_impl']}, rev={counts['numerator_reversal']}")

            # Save incrementally every 6 months
            if len(results) % 6 == 0:
                _save_results(output_file, existing, results)
                print(f"  [Saved {len(results)} new months]")

        except Exception as e:
            print(f"ERROR: {e}")
            # Save what we have so far
            if results:
                _save_results(output_file, existing, results)
            raise

        # Small delay between months
        time.sleep(0.5)

    # Final save
    if results:
        _save_results(output_file, existing, results)

    # Return combined data
    final = pd.read_csv(output_file)
    print(f"\nDone! Total months: {len(final)}")
    return final


def _save_results(output_file, existing, new_results):
    """Save results to CSV."""
    new_df = pd.DataFrame(new_results)
    if len(existing) > 0:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.sort_values('month').drop_duplicates('month')
    combined.to_csv(output_file, index=False)


def build_cpu_index(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Build CPU index from counts data."""
    df = counts_df.copy()

    # Calculate raw ratios
    df['ratio_cpu'] = df['numerator_cpu'] / df['denominator']
    df['ratio_impl'] = df['numerator_impl'] / df['denominator']
    df['ratio_reversal'] = df['numerator_reversal'] / df['denominator']

    # Standardize to mean=100 (BBD methodology)
    mean_cpu = df['ratio_cpu'].mean()
    mean_impl = df['ratio_impl'].mean()
    mean_rev = df['ratio_reversal'].mean()

    df['cpu_index'] = (df['ratio_cpu'] / mean_cpu) * 100
    df['cpu_impl'] = (df['ratio_impl'] / mean_impl) * 100
    df['cpu_reversal'] = (df['ratio_reversal'] / mean_rev) * 100

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--end-month", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Print queries without calling API")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN - showing sample query...")
        query = api.build_search_query(
            climate_terms=config.CLIMATE_TERMS[:3],
            policy_terms=config.POLICY_TERMS[:3],
            uncertainty_terms=config.UNCERTAINTY_TERMS[:3],
        )
        print(f"Query: {query[:500]}...")
    else:
        df = collect_all(args.start_year, args.start_month, args.end_year, args.end_month)
        if df is not None:
            # Build index
            index_df = build_cpu_index(df)
            output_file = Path(__file__).parent.parent / "data" / "cpu_index.csv"
            index_df.to_csv(output_file, index=False)
            print(f"\nCPU Index saved to: {output_file}")
            print("\nIndex preview (last 12 months):")
            print(index_df[['month', 'cpu_index', 'cpu_impl', 'cpu_reversal']].tail(12))
