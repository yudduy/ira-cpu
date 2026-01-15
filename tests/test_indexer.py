"""
Tests for indexer.py - CPU Index calculation

Tests index calculation and normalization with mocked database data.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCalculateRawIndex:
    """Tests for raw index calculation from database."""

    def test_calculate_raw_index_empty_db(self, initialized_db):
        """calculate_raw_index should return empty list for empty database."""
        import indexer

        result = indexer.calculate_raw_index()
        assert result == []

    def test_calculate_raw_index_with_data(self, populated_db):
        """calculate_raw_index should calculate ratios from progress data."""
        import indexer

        result = indexer.calculate_raw_index()

        assert len(result) == 3
        assert result[0]["month"] == "2024-01"
        assert result[0]["denominator"] == 150
        assert result[0]["numerator"] == 30
        assert result[0]["raw_ratio"] == 0.20  # 30/150

    def test_calculate_raw_index_sorted_by_month(self, initialized_db):
        """calculate_raw_index should return results sorted by month."""
        import indexer
        import db

        # Add out of order
        db.save_month_count("2024-03", "denominator", 160)
        db.save_month_count("2024-03", "numerator", 32)
        db.save_month_count("2024-01", "denominator", 150)
        db.save_month_count("2024-01", "numerator", 30)

        result = indexer.calculate_raw_index()
        months = [r["month"] for r in result]
        assert months == sorted(months)

    def test_calculate_raw_index_handles_zero_denominator(self, initialized_db):
        """calculate_raw_index should handle zero denominator."""
        import indexer
        import db

        db.save_month_count("2024-05", "denominator", 0)
        db.save_month_count("2024-05", "numerator", 0)

        result = indexer.calculate_raw_index()

        assert len(result) == 1
        assert result[0]["raw_ratio"] == 0.0

    def test_calculate_raw_index_ignores_incomplete(self, initialized_db):
        """calculate_raw_index should ignore months missing numerator or denominator."""
        import indexer
        import db

        # Complete month
        db.save_month_count("2024-06", "denominator", 200)
        db.save_month_count("2024-06", "numerator", 50)

        # Incomplete month (only denominator)
        db.save_month_count("2024-07", "denominator", 180)

        result = indexer.calculate_raw_index()

        assert len(result) == 1
        assert result[0]["month"] == "2024-06"


class TestNormalizeIndex:
    """Tests for index normalization."""

    def test_normalize_index_empty_list(self):
        """normalize_index should handle empty input."""
        import indexer

        result = indexer.normalize_index([])
        assert result == []

    def test_normalize_index_single_value(self):
        """normalize_index should normalize single value to 100."""
        import indexer

        raw = [{"month": "2024-01", "raw_ratio": 0.25}]
        result = indexer.normalize_index(raw)

        assert result[0]["normalized"] == 100.0

    def test_normalize_index_mean_equals_100(self, sample_index_values):
        """normalize_index should set mean to approximately 100."""
        import indexer
        import statistics

        result = indexer.normalize_index(sample_index_values)

        normalized_values = [r["normalized"] for r in result]
        mean = statistics.mean(normalized_values)
        assert abs(mean - 100) < 0.1  # Should be very close to 100

    def test_normalize_index_preserves_relative_values(self):
        """normalize_index should preserve relative differences."""
        import indexer

        raw = [
            {"month": "2024-01", "raw_ratio": 0.10},  # Low
            {"month": "2024-02", "raw_ratio": 0.20},  # Mean
            {"month": "2024-03", "raw_ratio": 0.30},  # High
        ]

        result = indexer.normalize_index(raw)

        # Month with 0.20 should be at 100 (mean)
        # Month with 0.10 should be at 50 (half)
        # Month with 0.30 should be at 150 (1.5x)
        # Use approximate comparison for floating point
        assert abs(result[0]["normalized"] - 50.0) < 0.01
        assert abs(result[1]["normalized"] - 100.0) < 0.01
        assert abs(result[2]["normalized"] - 150.0) < 0.01

    def test_normalize_index_with_base_period(self):
        """normalize_index should use base period for mean calculation."""
        import indexer

        raw = [
            {"month": "2024-01", "raw_ratio": 0.10},
            {"month": "2024-02", "raw_ratio": 0.10},
            {"month": "2024-03", "raw_ratio": 0.20},  # Outside base period
        ]

        # Use only Jan-Feb as base (mean = 0.10)
        result = indexer.normalize_index(raw, base_start="2024-01", base_end="2024-02")

        # 0.10 should be 100 (it's the base mean)
        # 0.20 should be 200 (double the base mean)
        assert result[0]["normalized"] == 100.0
        assert result[1]["normalized"] == 100.0
        assert result[2]["normalized"] == 200.0

    def test_normalize_index_handles_zero_mean(self):
        """normalize_index should handle all-zero ratios."""
        import indexer

        raw = [
            {"month": "2024-01", "raw_ratio": 0.0},
            {"month": "2024-02", "raw_ratio": 0.0},
        ]

        result = indexer.normalize_index(raw)

        # Should not crash, normalized should be 0
        assert result[0]["normalized"] == 0.0
        assert result[1]["normalized"] == 0.0


class TestBuildIndex:
    """Tests for complete index building."""

    def test_build_index_no_data(self, initialized_db):
        """build_index should return error with no data."""
        import indexer

        result = indexer.build_index()

        assert result["status"] == "error"
        assert "No data" in result["message"]

    def test_build_index_success(self, populated_db):
        """build_index should return success with populated data."""
        import indexer

        result = indexer.build_index()

        assert result["status"] == "success"
        assert "metadata" in result
        assert "series" in result

    def test_build_index_metadata(self, populated_db):
        """build_index should include correct metadata."""
        import indexer

        result = indexer.build_index()

        metadata = result["metadata"]
        assert "period" in metadata
        assert "num_months" in metadata
        assert "mean_raw_ratio" in metadata
        assert "mean_normalized" in metadata

    def test_build_index_saves_to_database(self, populated_db):
        """build_index should save normalized values to database."""
        import indexer
        import db

        # Clear any existing index values first
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM index_values")
        conn.commit()
        conn.close()

        indexer.build_index()

        # Check values were saved
        values = db.get_all_index_values()
        assert len(values) == 3  # 3 months in populated_db fixture

    def test_build_index_with_base_period(self, populated_db):
        """build_index should apply base period normalization."""
        import indexer

        result = indexer.build_index(base_start="2024-01", base_end="2024-02")

        assert "2024-01 to 2024-02" in result["metadata"]["base_period"]


class TestGetPriorMonth:
    """Tests for prior month calculation helper."""

    def test_get_prior_month_regular(self):
        """_get_prior_month should handle regular months."""
        import indexer

        assert indexer._get_prior_month("2024-06") == "2024-05"
        assert indexer._get_prior_month("2024-12") == "2024-11"

    def test_get_prior_month_january(self):
        """_get_prior_month should handle January (year boundary)."""
        import indexer

        assert indexer._get_prior_month("2024-01") == "2023-12"
        assert indexer._get_prior_month("2025-01") == "2024-12"


class TestValidateAgainstEvents:
    """Tests for event validation."""

    def test_validate_against_events_no_data(self, initialized_db):
        """validate_against_events should return error with no index."""
        import indexer

        result = indexer.validate_against_events()

        assert result["status"] == "error"
        assert "No index values" in result["message"]

    def test_validate_against_events_structure(self, populated_db):
        """validate_against_events should return expected structure."""
        import indexer
        import db

        # Build index first
        indexer.build_index()

        result = indexer.validate_against_events()

        assert "events" in result
        assert "summary" in result
        assert isinstance(result["events"], list)

    def test_validate_against_events_missing_data(self, initialized_db):
        """validate_against_events should handle missing months."""
        import indexer
        import db

        # Add only one month of data
        db.save_index_value("2024-01", 150, 30, 0.20, 100.0)

        result = indexer.validate_against_events()

        # Most events should show NO DATA
        no_data_count = sum(1 for e in result["events"] if e.get("result") == "NO DATA")
        assert no_data_count > 0


class TestGetIndexSummary:
    """Tests for index summary statistics."""

    def test_get_index_summary_empty(self, initialized_db):
        """get_index_summary should handle empty database."""
        import indexer

        result = indexer.get_index_summary()

        assert result["status"] == "empty"

    def test_get_index_summary_with_data(self, populated_db):
        """get_index_summary should return statistics."""
        import indexer

        result = indexer.get_index_summary()

        assert result["status"] == "ready"
        assert "period" in result
        assert "num_months" in result
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result

    def test_get_index_summary_peaks_and_troughs(self, populated_db):
        """get_index_summary should identify peaks and troughs."""
        import indexer

        result = indexer.get_index_summary()

        assert "top_3_peaks" in result
        assert "top_3_troughs" in result
        assert len(result["top_3_peaks"]) <= 3
        assert len(result["top_3_troughs"]) <= 3

    def test_get_index_summary_statistics_accuracy(self, initialized_db):
        """get_index_summary should calculate accurate statistics."""
        import indexer
        import db

        # Add specific values for predictable statistics
        db.save_index_value("2024-01", 100, 10, 0.10, 80.0)
        db.save_index_value("2024-02", 100, 10, 0.10, 100.0)
        db.save_index_value("2024-03", 100, 10, 0.10, 120.0)

        result = indexer.get_index_summary()

        assert result["mean"] == 100.0
        assert result["min"] == 80.0
        assert result["max"] == 120.0
