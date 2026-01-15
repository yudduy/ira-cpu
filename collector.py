"""
Data collection pipeline for CPU Index Builder

Orchestrates monthly data collection with:
- Automatic resume from where you left off
- Progress tracking
- API usage estimation
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta

import config
import db
import api


def generate_months(start_date: str, end_date: str) -> list:
    """Generate list of months between start and end dates."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    months = []
    current = start.replace(day=1)

    while current <= end:
        months.append(current.strftime("%Y-%m"))
        current += relativedelta(months=1)

    return months


def get_incomplete_months() -> list:
    """Get months that haven't been collected yet."""
    all_months = generate_months(config.START_DATE, config.END_DATE)
    completed = db.get_completed_months()
    return [m for m in all_months if m not in completed]


def estimate_api_usage() -> dict:
    """
    Estimate API calls needed before running collection.

    Returns dict with:
    - months: number of months to collect
    - searches: estimated API searches
    - percent_quota: percent of annual quota (24,000)
    """
    incomplete = get_incomplete_months()
    num_months = len(incomplete)

    # 2 searches per month (denominator + numerator)
    searches = num_months * 2

    return {
        "months": num_months,
        "searches": searches,
        "percent_quota": searches / 24000,  # Stanford annual limit
        "incomplete_months": incomplete,
    }


def collect_month(year: int, month: int, dry_run: bool = False) -> dict:
    """
    Collect data for a single month.

    Returns dict with:
    - month: "YYYY-MM"
    - denominator: count of all climate-policy articles
    - numerator: count of articles with uncertainty language
    - raw_ratio: numerator / denominator
    - status: "complete", "skipped", or "error"
    """
    month_str = f"{year:04d}-{month:02d}"

    # Check if already complete
    if month_str in db.get_completed_months():
        return {
            "month": month_str,
            "status": "skipped",
            "reason": "already complete",
        }

    try:
        # Get date range for this month
        start_date, end_date = api.build_month_dates(year, month)

        # Build denominator query (climate + policy)
        denom_query = api.build_search_query(
            climate_terms=config.CLIMATE_TERMS,
            policy_terms=config.POLICY_TERMS,
            start_date=start_date,
            end_date=end_date,
        )

        # Fetch denominator count
        denominator = api.fetch_count(denom_query, dry_run=dry_run)

        # Build numerator query (add uncertainty)
        numer_query = api.build_search_query(
            climate_terms=config.CLIMATE_TERMS,
            policy_terms=config.POLICY_TERMS,
            uncertainty_terms=config.UNCERTAINTY_TERMS,
            start_date=start_date,
            end_date=end_date,
        )

        # Fetch numerator count
        numerator = api.fetch_count(numer_query, dry_run=dry_run)

        # Calculate ratio (handle division by zero)
        raw_ratio = numerator / denominator if denominator > 0 else 0.0

        # Save to database
        if not dry_run:
            db.save_month_count(month_str, "denominator", denominator)
            db.save_month_count(month_str, "numerator", numerator)

        return {
            "month": month_str,
            "denominator": denominator,
            "numerator": numerator,
            "raw_ratio": raw_ratio,
            "status": "complete",
        }

    # TODO: Catch specific exceptions instead of broad Exception
    # Best practice: Catch only expected errors to avoid masking bugs.
    # Suggested: except (RuntimeError, requests.RequestException, requests.Timeout) as e:
    # See: https://qodo.ai/blog/6-best-practices-for-python-exception-handling/
    except Exception as e:
        return {
            "month": month_str,
            "status": "error",
            "error": str(e),
        }


def collect_all(dry_run: bool = False, progress_callback=None) -> dict:
    """
    Collect data for all incomplete months.

    Args:
        dry_run: If True, simulate without API calls
        progress_callback: Optional function(current, total, month) for progress updates

    Returns:
        Summary dict with totals and results
    """
    incomplete = get_incomplete_months()
    total = len(incomplete)

    results = []
    errors = []

    for i, month_str in enumerate(incomplete):
        year, month = int(month_str[:4]), int(month_str[5:7])

        if progress_callback:
            progress_callback(i + 1, total, month_str)

        result = collect_month(year, month, dry_run=dry_run)
        results.append(result)

        if result["status"] == "error":
            errors.append(result)

    return {
        "months_processed": len(results),
        "months_complete": len([r for r in results if r["status"] == "complete"]),
        "months_skipped": len([r for r in results if r["status"] == "skipped"]),
        "months_error": len(errors),
        "errors": errors,
        "results": results,
    }


def get_collection_status() -> dict:
    """Get current collection status for display."""
    all_months = generate_months(config.START_DATE, config.END_DATE)
    completed = db.get_completed_months()
    incomplete = get_incomplete_months()

    progress = db.get_all_progress()

    return {
        "date_range": f"{config.START_DATE} to {config.END_DATE}",
        "total_months": len(all_months),
        "completed_months": len(completed),
        "incomplete_months": len(incomplete),
        "percent_complete": len(completed) / len(all_months) if all_months else 0,
        "next_month": incomplete[0] if incomplete else None,
        "progress_records": progress,
    }
