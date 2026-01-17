"""
Tests for collector.py - Data collection pipeline

Uses mocked api module to prevent real API calls.
Tests month generation, progress tracking, and collection logic.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGenerateMonths:
    """Tests for month range generation."""

    def test_generate_months_single_month(self):
        """generate_months should handle single month range."""
        import collector

        months = collector.generate_months("2024-01-01", "2024-01-31")
        assert months == ["2024-01"]

    def test_generate_months_multiple_months(self):
        """generate_months should generate multiple months."""
        import collector

        months = collector.generate_months("2024-01-01", "2024-03-31")
        assert months == ["2024-01", "2024-02", "2024-03"]

    def test_generate_months_cross_year(self):
        """generate_months should handle year boundary."""
        import collector

        months = collector.generate_months("2023-11-01", "2024-02-28")
        assert months == ["2023-11", "2023-12", "2024-01", "2024-02"]

    def test_generate_months_full_year(self):
        """generate_months should handle full year."""
        import collector

        months = collector.generate_months("2024-01-01", "2024-12-31")
        assert len(months) == 12
        assert months[0] == "2024-01"
        assert months[-1] == "2024-12"

    def test_generate_months_partial_dates(self):
        """generate_months should work with mid-month dates."""
        import collector

        months = collector.generate_months("2024-01-15", "2024-03-10")
        assert months == ["2024-01", "2024-02", "2024-03"]


class TestGetIncompleteMonths:
    """Tests for incomplete month detection."""

    def test_get_incomplete_months_all_incomplete(self, initialized_db, monkeypatch):
        """get_incomplete_months should return all months when none complete."""
        import collector

        monkeypatch.setattr("config.START_DATE", "2024-01-01")
        monkeypatch.setattr("config.END_DATE", "2024-03-31")

        incomplete = collector.get_incomplete_months()
        assert incomplete == ["2024-01", "2024-02", "2024-03"]

    def test_get_incomplete_months_some_complete(self, initialized_db, monkeypatch):
        """get_incomplete_months should exclude completed months."""
        import collector
        import db

        monkeypatch.setattr("config.START_DATE", "2024-01-01")
        monkeypatch.setattr("config.END_DATE", "2024-03-31")

        # Mark January as complete
        db.save_month_count("2024-01", "denominator", 100)
        db.save_month_count("2024-01", "numerator", 20)

        incomplete = collector.get_incomplete_months()
        assert "2024-01" not in incomplete
        assert "2024-02" in incomplete
        assert "2024-03" in incomplete

    def test_get_incomplete_months_all_complete(self, initialized_db, monkeypatch):
        """get_incomplete_months should return empty when all complete."""
        import collector
        import db

        monkeypatch.setattr("config.START_DATE", "2024-01-01")
        monkeypatch.setattr("config.END_DATE", "2024-02-28")

        # Mark both months complete
        db.save_month_count("2024-01", "denominator", 100)
        db.save_month_count("2024-01", "numerator", 20)
        db.save_month_count("2024-02", "denominator", 110)
        db.save_month_count("2024-02", "numerator", 25)

        incomplete = collector.get_incomplete_months()
        assert incomplete == []


class TestEstimateApiUsage:
    """Tests for API usage estimation."""

    def test_estimate_api_usage_structure(self, initialized_db, monkeypatch):
        """estimate_api_usage should return dict with expected keys."""
        import collector

        monkeypatch.setattr("config.START_DATE", "2024-01-01")
        monkeypatch.setattr("config.END_DATE", "2024-03-31")

        estimate = collector.estimate_api_usage()

        assert "months" in estimate
        assert "searches" in estimate
        assert "percent_quota" in estimate
        assert "incomplete_months" in estimate

    def test_estimate_api_usage_calculation(self, initialized_db, monkeypatch):
        """estimate_api_usage should calculate correctly."""
        import collector

        monkeypatch.setattr("config.START_DATE", "2024-01-01")
        monkeypatch.setattr("config.END_DATE", "2024-06-30")

        estimate = collector.estimate_api_usage()

        assert estimate["months"] == 6
        # 4 searches per month: denominator, numerator, numerator_down, numerator_up
        assert estimate["searches"] == 24
        assert estimate["percent_quota"] == 24 / 24000

    def test_estimate_api_usage_excludes_complete(self, initialized_db, monkeypatch):
        """estimate_api_usage should exclude completed months."""
        import collector
        import db

        monkeypatch.setattr("config.START_DATE", "2024-01-01")
        monkeypatch.setattr("config.END_DATE", "2024-03-31")

        # Complete 2 months (legacy format with 2 query types)
        db.save_month_count("2024-01", "denominator", 100)
        db.save_month_count("2024-01", "numerator", 20)
        db.save_month_count("2024-02", "denominator", 110)
        db.save_month_count("2024-02", "numerator", 25)

        estimate = collector.estimate_api_usage()

        assert estimate["months"] == 1  # Only March incomplete
        # 4 searches per month for incomplete months
        assert estimate["searches"] == 4


class TestCollectMonth:
    """Tests for single month data collection."""

    def test_collect_month_dry_run(self, initialized_db):
        """collect_month with dry_run should return fake data."""
        import collector

        result = collector.collect_month(2024, 6, dry_run=True)

        assert result["status"] == "complete"
        assert result["month"] == "2024-06"
        assert "denominator" in result
        assert "numerator" in result
        assert "numerator_down" in result
        assert "numerator_up" in result
        # Note: raw_ratio is calculated in indexer, not collector

    def test_collect_month_skips_complete(self, initialized_db):
        """collect_month should skip already complete months (legacy format)."""
        import collector
        import db

        # Mark month as complete (legacy format with 2 query types)
        db.save_month_count("2024-07", "denominator", 100)
        db.save_month_count("2024-07", "numerator", 20)

        result = collector.collect_month(2024, 7, dry_run=True)

        assert result["status"] == "skipped"
        assert "already complete" in result.get("reason", "")

    def test_collect_month_calls_api(self, initialized_db):
        """collect_month should call API with correct queries."""
        import collector

        with patch("collector.api.build_month_dates", return_value=("2024-08-01", "2024-08-31")):
            with patch("collector.api.build_search_query") as mock_build:
                with patch("collector.api.fetch_count", return_value=100):
                    result = collector.collect_month(2024, 8, dry_run=False)

        # Should call build_search_query 4 times:
        # 1. denominator (climate + policy)
        # 2. numerator (climate + policy + uncertainty)
        # 3. numerator_down (+ downside)
        # 4. numerator_up (+ upside)
        assert mock_build.call_count == 4

    def test_collect_month_returns_counts(self, initialized_db):
        """collect_month should return correct counts."""
        import collector

        with patch("collector.api.build_month_dates", return_value=("2024-09-01", "2024-09-30")):
            with patch("collector.api.build_search_query", return_value="query"):
                # 4 values: denom, numer, numer_down, numer_up
                with patch("collector.api.fetch_count", side_effect=[200, 50, 30, 20]):
                    result = collector.collect_month(2024, 9, dry_run=False)

        assert result["denominator"] == 200
        assert result["numerator"] == 50
        assert result["numerator_down"] == 30
        assert result["numerator_up"] == 20
        # Note: raw_ratio is calculated in indexer, not collector

    def test_collect_month_handles_zero_denominator(self, initialized_db):
        """collect_month should handle zero denominator gracefully."""
        import collector

        with patch("collector.api.build_month_dates", return_value=("2024-10-01", "2024-10-31")):
            with patch("collector.api.build_search_query", return_value="query"):
                # 4 zeros for the 4 fetch_count calls
                with patch("collector.api.fetch_count", side_effect=[0, 0, 0, 0]):
                    result = collector.collect_month(2024, 10, dry_run=False)

        assert result["denominator"] == 0
        assert result["numerator"] == 0
        # No crash on zero denominator - indexer handles ratio calculation

    def test_collect_month_saves_to_db(self, initialized_db):
        """collect_month should save results to database."""
        import collector
        import db

        with patch("collector.api.build_month_dates", return_value=("2024-11-01", "2024-11-30")):
            with patch("collector.api.build_search_query", return_value="query"):
                # 4 values: denom, numer, numer_down, numer_up
                with patch("collector.api.fetch_count", side_effect=[150, 30, 20, 10]):
                    collector.collect_month(2024, 11, dry_run=False)

        assert db.get_month_count("2024-11", "denominator") == 150
        assert db.get_month_count("2024-11", "numerator") == 30
        assert db.get_month_count("2024-11", "numerator_down") == 20
        assert db.get_month_count("2024-11", "numerator_up") == 10

    def test_collect_month_handles_api_error(self, initialized_db):
        """collect_month should handle API errors gracefully."""
        import collector

        with patch("collector.api.build_month_dates", return_value=("2024-12-01", "2024-12-31")):
            with patch("collector.api.build_search_query", return_value="query"):
                with patch("collector.api.fetch_count", side_effect=RuntimeError("API Error")):
                    result = collector.collect_month(2024, 12, dry_run=False)

        assert result["status"] == "error"
        assert "API Error" in result["error"]


class TestCollectAll:
    """Tests for batch collection."""

    def test_collect_all_dry_run(self, initialized_db, monkeypatch):
        """collect_all with dry_run should process without API calls."""
        import collector

        monkeypatch.setattr("config.START_DATE", "2024-01-01")
        monkeypatch.setattr("config.END_DATE", "2024-02-28")

        result = collector.collect_all(dry_run=True)

        assert result["months_processed"] == 2
        assert result["months_complete"] == 2
        assert result["months_error"] == 0

    def test_collect_all_with_progress_callback(self, initialized_db, monkeypatch):
        """collect_all should call progress callback."""
        import collector

        monkeypatch.setattr("config.START_DATE", "2024-01-01")
        monkeypatch.setattr("config.END_DATE", "2024-02-28")

        callback = MagicMock()
        collector.collect_all(dry_run=True, progress_callback=callback)

        assert callback.call_count == 2
        callback.assert_any_call(1, 2, "2024-01")
        callback.assert_any_call(2, 2, "2024-02")

    def test_collect_all_skips_complete(self, initialized_db, monkeypatch):
        """collect_all should skip already complete months."""
        import collector
        import db

        monkeypatch.setattr("config.START_DATE", "2024-01-01")
        monkeypatch.setattr("config.END_DATE", "2024-03-31")

        # Complete one month
        db.save_month_count("2024-01", "denominator", 100)
        db.save_month_count("2024-01", "numerator", 20)

        result = collector.collect_all(dry_run=True)

        # Should process 2 incomplete months, 0 skipped (they won't even be iterated)
        assert result["months_processed"] == 2
        assert "2024-01" not in [r["month"] for r in result["results"]]


class TestGetCollectionStatus:
    """Tests for collection status reporting."""

    def test_get_collection_status_structure(self, initialized_db, monkeypatch):
        """get_collection_status should return expected structure."""
        import collector

        monkeypatch.setattr("config.START_DATE", "2024-01-01")
        monkeypatch.setattr("config.END_DATE", "2024-03-31")

        status = collector.get_collection_status()

        assert "date_range" in status
        assert "total_months" in status
        assert "completed_months" in status
        assert "incomplete_months" in status
        assert "percent_complete" in status
        assert "next_month" in status

    def test_get_collection_status_values(self, initialized_db, monkeypatch):
        """get_collection_status should calculate correct values."""
        import collector
        import db

        monkeypatch.setattr("config.START_DATE", "2024-01-01")
        monkeypatch.setattr("config.END_DATE", "2024-04-30")

        # Complete 2 of 4 months
        db.save_month_count("2024-01", "denominator", 100)
        db.save_month_count("2024-01", "numerator", 20)
        db.save_month_count("2024-02", "denominator", 110)
        db.save_month_count("2024-02", "numerator", 25)

        status = collector.get_collection_status()

        assert status["total_months"] == 4
        assert status["completed_months"] == 2
        assert status["incomplete_months"] == 2
        assert status["percent_complete"] == 0.5
        assert status["next_month"] == "2024-03"

    def test_get_collection_status_all_complete(self, initialized_db, monkeypatch):
        """get_collection_status should handle all complete."""
        import collector
        import db

        monkeypatch.setattr("config.START_DATE", "2024-01-01")
        monkeypatch.setattr("config.END_DATE", "2024-01-31")

        db.save_month_count("2024-01", "denominator", 100)
        db.save_month_count("2024-01", "numerator", 20)

        status = collector.get_collection_status()

        assert status["percent_complete"] == 1.0
        assert status["next_month"] is None
