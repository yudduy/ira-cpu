"""
Tests for collector.py - Data collection pipeline

Uses mocked api module to prevent real API calls.
Tests month generation, progress tracking, and collection logic.
"""

from unittest.mock import MagicMock, patch

import pytest

from cpu_index.collection import collector
from cpu_index.collection.collector import ArticleCollector


class TestGenerateMonths:
    """Tests for month range generation."""

    def test_generate_months_single_month(self):
        """generate_months should handle single month range."""
        months = collector.generate_months("2024-01-01", "2024-01-31")
        assert months == ["2024-01"]

    def test_generate_months_multiple_months(self):
        """generate_months should generate multiple months."""
        months = collector.generate_months("2024-01-01", "2024-03-31")
        assert months == ["2024-01", "2024-02", "2024-03"]

    def test_generate_months_cross_year(self):
        """generate_months should handle year boundary."""
        months = collector.generate_months("2023-11-01", "2024-02-28")
        assert months == ["2023-11", "2023-12", "2024-01", "2024-02"]

    def test_generate_months_full_year(self):
        """generate_months should handle full year."""
        months = collector.generate_months("2024-01-01", "2024-12-31")
        assert len(months) == 12
        assert months[0] == "2024-01"
        assert months[-1] == "2024-12"

    def test_generate_months_partial_dates(self):
        """generate_months should work with mid-month dates."""
        months = collector.generate_months("2024-01-15", "2024-03-10")
        assert months == ["2024-01", "2024-02", "2024-03"]


class TestGetIncompleteMonths:
    """Tests for incomplete month detection."""

    @patch("cpu_index.collection.collector.db_postgres")
    def test_get_incomplete_months_all_incomplete(self, mock_db, monkeypatch):
        """get_incomplete_months should return all months when none complete."""
        monkeypatch.setattr("cpu_index.config.START_DATE", "2024-01-01")
        monkeypatch.setattr("cpu_index.config.END_DATE", "2024-03-31")

        # Mock: No months complete
        mock_db.get_completed_months.return_value = set()

        incomplete = collector.get_incomplete_months()
        assert incomplete == ["2024-01", "2024-02", "2024-03"]

    @patch("cpu_index.collection.collector.db_postgres")
    def test_get_incomplete_months_some_complete(self, mock_db, monkeypatch):
        """get_incomplete_months should exclude completed months."""
        monkeypatch.setattr("cpu_index.config.START_DATE", "2024-01-01")
        monkeypatch.setattr("cpu_index.config.END_DATE", "2024-03-31")

        # Mock: January is complete
        mock_db.get_completed_months.return_value = {"2024-01"}

        incomplete = collector.get_incomplete_months()
        assert "2024-01" not in incomplete
        assert "2024-02" in incomplete
        assert "2024-03" in incomplete

    @patch("cpu_index.collection.collector.db_postgres")
    def test_get_incomplete_months_all_complete(self, mock_db, monkeypatch):
        """get_incomplete_months should return empty when all complete."""
        monkeypatch.setattr("cpu_index.config.START_DATE", "2024-01-01")
        monkeypatch.setattr("cpu_index.config.END_DATE", "2024-02-28")

        # Mock: Both months complete
        mock_db.get_completed_months.return_value = {"2024-01", "2024-02"}

        incomplete = collector.get_incomplete_months()
        assert incomplete == []


class TestEstimateApiUsage:
    """Tests for API usage estimation."""

    @patch("cpu_index.collection.collector.db_postgres")
    def test_estimate_api_usage_structure(self, mock_db, monkeypatch):
        """estimate_api_usage should return dict with expected keys."""
        monkeypatch.setattr("cpu_index.config.START_DATE", "2024-01-01")
        monkeypatch.setattr("cpu_index.config.END_DATE", "2024-03-31")

        mock_db.get_completed_months.return_value = set()

        estimate = collector.estimate_api_usage()

        assert "months" in estimate
        assert "searches" in estimate
        assert "percent_quota" in estimate
        assert "incomplete_months" in estimate

    @patch("cpu_index.collection.collector.db_postgres")
    def test_estimate_api_usage_calculation(self, mock_db, monkeypatch):
        """estimate_api_usage should calculate correctly (1 search per month)."""
        monkeypatch.setattr("cpu_index.config.START_DATE", "2024-01-01")
        monkeypatch.setattr("cpu_index.config.END_DATE", "2024-06-30")

        mock_db.get_completed_months.return_value = set()

        estimate = collector.estimate_api_usage()

        assert estimate["months"] == 6
        # 1 search per month: fetch all climate+policy articles
        assert estimate["searches"] == 6
        assert estimate["percent_quota"] == 6 / 24000

    @patch("cpu_index.collection.collector.db_postgres")
    def test_estimate_api_usage_excludes_complete(self, mock_db, monkeypatch):
        """estimate_api_usage should exclude completed months."""
        monkeypatch.setattr("cpu_index.config.START_DATE", "2024-01-01")
        monkeypatch.setattr("cpu_index.config.END_DATE", "2024-03-31")

        # Mock: 2 months complete
        mock_db.get_completed_months.return_value = {"2024-01", "2024-02"}

        estimate = collector.estimate_api_usage()

        assert estimate["months"] == 1  # Only March incomplete
        # 1 search per month for incomplete months
        assert estimate["searches"] == 1


class TestGetCollectionStatus:
    """Tests for collection status reporting."""

    @patch("cpu_index.collection.collector.db_postgres")
    def test_get_collection_status_structure(self, mock_db, monkeypatch):
        """get_collection_status should return expected structure."""
        monkeypatch.setattr("cpu_index.config.START_DATE", "2024-01-01")
        monkeypatch.setattr("cpu_index.config.END_DATE", "2024-03-31")

        mock_db.get_completed_months.return_value = set()
        mock_db.get_collection_progress.return_value = {}

        status = collector.get_collection_status()

        assert "date_range" in status
        assert "total_months" in status
        assert "completed_months" in status
        assert "incomplete_months" in status
        assert "percent_complete" in status
        assert "next_month" in status

    @patch("cpu_index.collection.collector.db_postgres")
    def test_get_collection_status_values(self, mock_db, monkeypatch):
        """get_collection_status should calculate correct values."""
        monkeypatch.setattr("cpu_index.config.START_DATE", "2024-01-01")
        monkeypatch.setattr("cpu_index.config.END_DATE", "2024-04-30")

        # Mock: 2 of 4 months complete
        mock_db.get_completed_months.return_value = {"2024-01", "2024-02"}
        mock_db.get_collection_progress.return_value = {
            "2024-01": {"articles_fetched": 100},
            "2024-02": {"articles_fetched": 110},
        }

        status = collector.get_collection_status()

        assert status["total_months"] == 4
        assert status["completed_months"] == 2
        assert status["incomplete_months"] == 2
        assert status["percent_complete"] == 0.5
        assert status["next_month"] == "2024-03"

    @patch("cpu_index.collection.collector.db_postgres")
    def test_get_collection_status_all_complete(self, mock_db, monkeypatch):
        """get_collection_status should handle all complete."""
        monkeypatch.setattr("cpu_index.config.START_DATE", "2024-01-01")
        monkeypatch.setattr("cpu_index.config.END_DATE", "2024-01-31")

        mock_db.get_completed_months.return_value = {"2024-01"}
        mock_db.get_collection_progress.return_value = {
            "2024-01": {"articles_fetched": 100},
        }

        status = collector.get_collection_status()

        assert status["percent_complete"] == 1.0
        assert status["next_month"] is None


# =============================================================================
# Tests for ArticleCollector (fetch once, classify locally)
# =============================================================================

class TestArticleCollectorInit:
    """Tests for ArticleCollector initialization."""

    def test_collector_initialization(self):
        """Collector should initialize with date range."""
        coll = ArticleCollector(
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=6,
        )

        assert coll.start_year == 2024
        assert coll.start_month == 1
        assert coll.end_year == 2024
        assert coll.end_month == 6

    def test_collector_validates_month_range(self):
        """Collector should validate month values."""
        with pytest.raises(ValueError, match="Month must be between"):
            ArticleCollector(
                start_year=2024,
                start_month=13,
                end_year=2024,
                end_month=1,
            )

    def test_collector_validates_date_range(self):
        """Collector should validate that end date is after start date."""
        with pytest.raises(ValueError, match="End date must be after"):
            ArticleCollector(
                start_year=2025,
                start_month=1,
                end_year=2024,
                end_month=12,
            )


class TestArticleCollectorMonthRange:
    """Tests for ArticleCollector month range generation."""

    def test_collector_generates_month_range(self):
        """Collector should generate correct month range."""
        coll = ArticleCollector(
            start_year=2024,
            start_month=10,
            end_year=2025,
            end_month=2,
        )

        months = list(coll.get_month_range())

        assert len(months) == 5
        assert months[0] == (2024, 10)
        assert months[1] == (2024, 11)
        assert months[2] == (2024, 12)
        assert months[3] == (2025, 1)
        assert months[4] == (2025, 2)

    def test_collector_generates_single_month(self):
        """Collector should handle single month range."""
        coll = ArticleCollector(
            start_year=2024,
            start_month=6,
            end_year=2024,
            end_month=6,
        )

        months = list(coll.get_month_range())

        assert len(months) == 1
        assert months[0] == (2024, 6)


class TestArticleCollectorDryRun:
    """Tests for ArticleCollector in dry run mode."""

    def test_collect_month_dry_run(self):
        """collect_month with dry_run should return fake articles."""
        coll = ArticleCollector(
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
            dry_run=True,
        )

        articles = coll.collect_month(2024, 1)

        assert len(articles) > 0
        assert all("id" in a for a in articles)
        assert all("month" in a for a in articles)

    def test_collect_all_dry_run(self):
        """collect_all with dry_run should process all months."""
        coll = ArticleCollector(
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=3,
            dry_run=True,
        )

        result = coll.collect_all()

        assert result["months_processed"] == 3
        assert result["total_articles"] > 0


class TestArticleCollectorWithMockedAPI:
    """Tests for ArticleCollector with mocked API calls."""

    def test_collect_month_calls_api(self, mock_env_vars):
        """collect_month should call API and return articles."""
        coll = ArticleCollector(
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
        )

        mock_articles = [
            {"id": "art_1", "title": "Climate Policy", "date": "2024-01-15",
             "source": "Reuters", "snippet": "Article about climate policy...",
             "month": "2024-01"},
            {"id": "art_2", "title": "Energy Regulation", "date": "2024-01-20",
             "source": "AP", "snippet": "New energy regulations...",
             "month": "2024-01"},
        ]
        mock_metadata = {
            "query_hash": "abc123",
            "month": "2024-01",
            "total_count": 2,
            "fetched_at": "2024-01-31T12:00:00",
        }

        with patch("cpu_index.collection.collector.api.fetch_articles_for_month") as mock_fetch:
            mock_fetch.return_value = (mock_articles, mock_metadata)
            articles = coll.collect_month(2024, 1)

        assert len(articles) == 2
        mock_fetch.assert_called_once_with(
            year=2024,
            month=1,
            dry_run=False,
            progress_callback=None,
        )

    def test_collect_month_with_storage(self, mock_env_vars):
        """collect_month should store articles when db is available."""
        coll = ArticleCollector(
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
        )

        mock_articles = [
            {"id": "art_1", "title": "Test", "date": "2024-01-15",
             "source": "Test", "snippet": "Test", "month": "2024-01"},
        ]
        mock_metadata = {"query_hash": "abc", "month": "2024-01", "total_count": 1}

        with patch("cpu_index.collection.collector.api.fetch_articles_for_month") as mock_fetch:
            with patch("cpu_index.collection.collector.db_postgres.save_articles_batch") as mock_save:
                with patch("cpu_index.collection.collector.db_postgres.mark_month_complete") as mock_mark:
                    mock_fetch.return_value = (mock_articles, mock_metadata)
                    mock_save.return_value = 1
                    coll.collect_month(2024, 1, store=True)

        mock_save.assert_called_once()
        mock_mark.assert_called_once_with("2024-01", 1)


class TestArticleCollectorClassification:
    """Tests for local classification during collection."""

    def test_collector_classifies_after_fetch(self, mock_env_vars):
        """Collector should classify articles after fetching."""
        coll = ArticleCollector(
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
        )

        mock_articles = [
            {"id": "art_1", "title": "Uncertain climate policy",
             "date": "2024-01-15", "source": "Test",
             "snippet": "Rollback of regulations uncertain", "month": "2024-01"},
        ]
        mock_metadata = {"query_hash": "abc", "month": "2024-01", "total_count": 1}

        with patch("cpu_index.collection.collector.api.fetch_articles_for_month") as mock_fetch:
            with patch("cpu_index.collection.collector.db_postgres.save_articles_batch"):
                with patch("cpu_index.collection.collector.db_postgres.save_keyword_classifications_batch") as mock_classify:
                    with patch("cpu_index.collection.collector.db_postgres.mark_month_complete"):
                        mock_fetch.return_value = (mock_articles, mock_metadata)
                        coll.collect_month(2024, 1, store=True, classify=True)

        # Should have saved classifications
        mock_classify.assert_called_once()
        classifications = mock_classify.call_args[0][0]
        assert len(classifications) == 1
        assert classifications[0]["article_id"] == "art_1"


class TestArticleCollectorPendingMonths:
    """Tests for pending months tracking."""

    @patch("cpu_index.collection.collector.db_postgres")
    def test_get_pending_months_excludes_completed(self, mock_db):
        """get_pending_months should exclude completed months."""
        coll = ArticleCollector(
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=3,
        )

        mock_db.get_completed_months.return_value = {"2024-01", "2024-02"}

        pending = coll.get_pending_months()

        assert len(pending) == 1
        assert pending[0] == (2024, 3)

    @patch("cpu_index.collection.collector.db_postgres")
    def test_get_pending_months_all_pending(self, mock_db):
        """get_pending_months should return all when none completed."""
        coll = ArticleCollector(
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=3,
        )

        mock_db.get_completed_months.return_value = set()

        pending = coll.get_pending_months()

        assert len(pending) == 3
        assert pending == [(2024, 1), (2024, 2), (2024, 3)]


class TestArticleCollectorStatus:
    """Tests for ArticleCollector status reporting."""

    @patch("cpu_index.collection.collector.db_postgres")
    def test_get_status_structure(self, mock_db):
        """get_status should return expected structure."""
        coll = ArticleCollector(
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=3,
        )

        mock_db.get_completed_months.return_value = {"2024-01"}
        mock_db.get_collection_progress.return_value = {
            "2024-01": {"articles_fetched": 150}
        }

        status = coll.get_status()

        assert "total_months" in status
        assert "completed_months" in status
        assert "pending_months" in status
        assert "total_articles_collected" in status
        assert status["total_months"] == 3
        assert status["completed_months"] == 1
        assert status["pending_months"] == 2
        assert status["total_articles_collected"] == 150
