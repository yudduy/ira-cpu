"""
Data collection pipeline for CPU Index Builder

Orchestrates monthly data collection with:
- Automatic resume from where you left off
- Progress tracking
- API usage estimation

Strategy: Fetch articles once, classify locally
1. Fetch all articles matching (climate AND policy) for a month
2. Store them in PostgreSQL with full metadata
3. Classify locally using keyword matching
4. Track progress for resumability

Benefits:
- No API quota consumed for keyword experiments
- Full data preserved for analysis
- Addresses Steve's critique: direction terms alone don't indicate uncertainty
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional

from cpu_index import config
from cpu_index import db_postgres
from cpu_index.collection import api


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
    completed = db_postgres.get_completed_months()
    return [m for m in all_months if m not in completed]


def estimate_api_usage() -> dict:
    """
    Estimate API calls needed before running collection.

    Returns dict with:
    - months: number of months to collect
    - searches: estimated API searches (1 per month)
    - percent_quota: percent of annual quota (24,000)
    """
    incomplete = get_incomplete_months()
    num_months = len(incomplete)

    # 1 search per month: fetch all climate+policy articles
    # Classification happens locally, no additional API calls needed
    searches = num_months

    return {
        "months": num_months,
        "searches": searches,
        "percent_quota": searches / 24000,  # Stanford annual limit
        "incomplete_months": incomplete,
    }


def get_collection_status() -> dict:
    """Get current collection status for display."""
    all_months = generate_months(config.START_DATE, config.END_DATE)
    completed = db_postgres.get_completed_months()
    incomplete = get_incomplete_months()

    progress = db_postgres.get_collection_progress()

    return {
        "date_range": f"{config.START_DATE} to {config.END_DATE}",
        "total_months": len(all_months),
        "completed_months": len(completed),
        "incomplete_months": len(incomplete),
        "percent_complete": len(completed) / len(all_months) if all_months else 0,
        "next_month": incomplete[0] if incomplete else None,
        "progress_records": progress,
    }


class ArticleCollector:
    """
    Collector that fetches full articles and stores them for local classification.

    This implements the "fetch once, classify locally" strategy:
    1. Fetch all articles matching (climate AND policy) for a month
    2. Store them in PostgreSQL with full metadata
    3. Classify locally using keyword matching
    4. Track progress for resumability

    Benefits:
    - No API quota consumed for keyword experiments
    - Full data preserved for analysis
    - Addresses Steve's critique: direction terms alone don't indicate uncertainty
    """

    def __init__(
        self,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
        dry_run: bool = False,
    ):
        """
        Initialize collector with date range.

        Args:
            start_year: Start year
            start_month: Start month (1-12)
            end_year: End year
            end_month: End month (1-12)
            dry_run: If True, don't make real API calls
        """
        # Validate months
        if not (1 <= start_month <= 12):
            raise ValueError(f"Month must be between 1 and 12, got start_month={start_month}")
        if not (1 <= end_month <= 12):
            raise ValueError(f"Month must be between 1 and 12, got end_month={end_month}")

        # Validate date range
        start_val = start_year * 12 + start_month
        end_val = end_year * 12 + end_month
        if end_val < start_val:
            raise ValueError(
                f"End date must be after start date: "
                f"{start_year}-{start_month:02d} > {end_year}-{end_month:02d}"
            )

        self.start_year = start_year
        self.start_month = start_month
        self.end_year = end_year
        self.end_month = end_month
        self.dry_run = dry_run

    def get_month_range(self):
        """
        Generate (year, month) tuples for the date range.

        Yields:
            Tuple of (year, month) for each month in range
        """
        year = self.start_year
        month = self.start_month

        while (year, month) <= (self.end_year, self.end_month):
            yield (year, month)
            month += 1
            if month > 12:
                month = 1
                year += 1

    def get_pending_months(self) -> list:
        """
        Get months that haven't been collected yet.

        Returns:
            List of (year, month) tuples for pending months
        """
        try:
            completed = db_postgres.get_completed_months()
        except Exception:
            # If DB not available, all months are pending
            completed = []

        pending = []
        for year, month in self.get_month_range():
            month_str = f"{year}-{month:02d}"
            if month_str not in completed:
                pending.append((year, month))

        return pending

    def get_status(self) -> dict:
        """
        Get current collection status.

        Returns:
            Dict with status information
        """
        all_months = list(self.get_month_range())

        try:
            completed = db_postgres.get_completed_months()
            progress = db_postgres.get_collection_progress()
        except Exception:
            completed = []
            progress = {}

        completed_in_range = [
            m for m in completed
            if f"{self.start_year}-{self.start_month:02d}" <= m <= f"{self.end_year}-{self.end_month:02d}"
        ]

        total_articles = sum(
            p.get("articles_fetched", 0)
            for m, p in progress.items()
            if m in completed_in_range
        )

        return {
            "total_months": len(all_months),
            "completed_months": len(completed_in_range),
            "pending_months": len(all_months) - len(completed_in_range),
            "total_articles_collected": total_articles,
        }

    def collect_month(
        self,
        year: int,
        month: int,
        store: bool = False,
        classify: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> list:
        """
        Collect articles for a single month.

        Args:
            year: Year to collect
            month: Month to collect
            store: If True, store articles in PostgreSQL
            classify: If True, classify articles with local keywords
            progress_callback: Optional callback for progress updates

        Returns:
            List of article dicts
        """
        articles, metadata = api.fetch_articles_for_month(
            year=year,
            month=month,
            dry_run=self.dry_run,
            progress_callback=progress_callback,
        )

        if store and not self.dry_run:
            from cpu_index.classification import local_classifier

            # Save articles to database
            db_postgres.save_articles_batch(articles)

            if classify:
                # Classify each article
                classifications = []
                for article in articles:
                    classification = local_classifier.classify_article(
                        title=article.get("title", ""),
                        snippet=article.get("snippet", ""),
                    )
                    classification["article_id"] = article["id"]
                    classifications.append(classification)

                # Save classifications
                db_postgres.save_keyword_classifications_batch(classifications)

            # Mark month complete
            month_str = f"{year}-{month:02d}"
            db_postgres.mark_month_complete(month_str, len(articles))

        return articles

    def collect_all(
        self,
        store: bool = False,
        classify: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """
        Collect articles for all pending months.

        Args:
            store: If True, store articles in PostgreSQL
            classify: If True, classify articles with local keywords
            progress_callback: Optional callback(month_num, total_months, month_str)

        Returns:
            Summary dict with collection results
        """
        pending = self.get_pending_months()
        total = len(pending)
        total_articles = 0
        results = []

        for i, (year, month) in enumerate(pending):
            month_str = f"{year}-{month:02d}"

            if progress_callback:
                progress_callback(i + 1, total, month_str)

            try:
                articles = self.collect_month(
                    year=year,
                    month=month,
                    store=store,
                    classify=classify,
                )
                total_articles += len(articles)
                results.append({
                    "month": month_str,
                    "status": "complete",
                    "articles": len(articles),
                })
            except Exception as e:
                results.append({
                    "month": month_str,
                    "status": "error",
                    "error": str(e),
                })

        return {
            "months_processed": len(results),
            "total_articles": total_articles,
            "results": results,
        }
