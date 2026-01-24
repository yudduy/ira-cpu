"""
Count-Based Collector for CPU Index

Collects article COUNTS (not full articles) for fast index building.
Uses ~4 API calls per month instead of 2000+.

Query structure:
- denominator: climate AND policy
- numerator_cpu: climate AND policy AND uncertainty
- numerator_impl: climate AND policy AND uncertainty AND implementation
- numerator_reversal: climate AND policy AND uncertainty AND reversal
"""

from typing import Callable, Optional

from cpu_index.collection import api
from cpu_index import config
from cpu_index import db_postgres


def parse_config_date(date_str: str) -> tuple[int, int]:
    """Parse YYYY-MM-DD date string into (year, month)."""
    parts = date_str.split("-")
    return int(parts[0]), int(parts[1])


def iter_months(start_year: int, start_month: int, end_year: int, end_month: int):
    """Yield (year, month) tuples for each month in range."""
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month > 12:
            month = 1
            year += 1


def collect_month_counts(year: int, month: int, dry_run: bool = False) -> dict:
    """
    Collect all counts for a single month.

    Args:
        year: Year to collect
        month: Month (1-12)
        dry_run: If True, return fake counts

    Returns:
        Dict with all counts for the month
    """
    start_date, end_date = api.build_month_dates(year, month)
    date_filter = api.build_date_filter(start_date, end_date)

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

    return {
        "month": f"{year}-{month:02d}",
        "denominator": api.fetch_count(query_denom, date_filter, dry_run=dry_run),
        "numerator_cpu": api.fetch_count(query_cpu, date_filter, dry_run=dry_run),
        "numerator_impl": api.fetch_count(query_impl, date_filter, dry_run=dry_run),
        "numerator_reversal": api.fetch_count(query_reversal, date_filter, dry_run=dry_run),
    }


def collect_all_counts(
    start_year: int = None,
    start_month: int = None,
    end_year: int = None,
    end_month: int = None,
    dry_run: bool = False,
    save: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    Collect counts for all months in date range.

    Args:
        start_year: Start year (default from config)
        start_month: Start month (default from config)
        end_year: End year (default from config)
        end_month: End month (default from config)
        dry_run: If True, don't make real API calls
        save: If True, save to database
        progress_callback: Optional callback(current, total, month_str)

    Returns:
        Summary dict with all results
    """
    if start_year is None:
        start_year, start_month = parse_config_date(config.START_DATE)
    if end_year is None:
        end_year, end_month = parse_config_date(config.END_DATE)

    completed = db_postgres.get_completed_count_months() if save else set()

    pending_months = [
        (y, m)
        for y, m in iter_months(start_year, start_month, end_year, end_month)
        if f"{y}-{m:02d}" not in completed
    ]

    results = []
    total = len(pending_months)

    for i, (y, m) in enumerate(pending_months):
        month_str = f"{y}-{m:02d}"

        if progress_callback:
            progress_callback(i + 1, total, month_str)

        try:
            counts = collect_month_counts(y, m, dry_run=dry_run)
            if save and not dry_run:
                db_postgres.save_monthly_counts(counts)
            results.append({"month": month_str, "status": "complete", **counts})
        except Exception as e:
            results.append({"month": month_str, "status": "error", "error": str(e)})

    return {
        "months_processed": len(results),
        "results": results,
    }


def get_collection_status() -> dict:
    """Get count collection status."""
    start_year, start_month = parse_config_date(config.START_DATE)
    end_year, end_month = parse_config_date(config.END_DATE)

    total_months = sum(1 for _ in iter_months(start_year, start_month, end_year, end_month))
    completed = db_postgres.get_completed_count_months()
    completed_count = len(completed)

    return {
        "date_range": f"{config.START_DATE} to {config.END_DATE}",
        "total_months": total_months,
        "completed_months": completed_count,
        "pending_months": total_months - completed_count,
        "percent_complete": completed_count / total_months if total_months else 0,
    }


def estimate_api_usage() -> dict:
    """Estimate API calls needed for pending months."""
    status = get_collection_status()
    pending = status["pending_months"]
    searches = pending * 4  # 4 queries per month (denom, cpu, impl, reversal)

    return {
        "months": pending,
        "searches": searches,
    }
