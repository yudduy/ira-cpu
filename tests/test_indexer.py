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


class TestCalculateRawIndexValues:
    """Tests for raw index value calculation."""

    def test_calculate_raw_index_values_basic(self):
        """calculate_raw_index_values should calculate ratios correctly."""
        import indexer

        raw_counts = [
            {
                "month": "2024-01",
                "total_articles": 100,
                "uncertainty_count": 20,
                "implementation_uncertainty_count": 10,
                "reversal_uncertainty_count": 8,
                "ira_count": 5,
                "obbba_count": 2,
            }
        ]

        result = indexer.calculate_raw_index_values(raw_counts)

        assert len(result) == 1
        assert result[0]["month"] == "2024-01"
        assert result[0]["denominator"] == 100
        assert result[0]["raw_ratio_cpu"] == 0.20  # 20/100
        assert result[0]["raw_ratio_impl"] == 0.10  # 10/100
        assert result[0]["raw_ratio_reversal"] == 0.08  # 8/100

    def test_calculate_raw_index_values_direction_metric(self):
        """calculate_raw_index_values should compute direction correctly."""
        import indexer

        raw_counts = [
            {
                "month": "2024-01",
                "total_articles": 100,
                "uncertainty_count": 20,
                "implementation_uncertainty_count": 15,  # More impl
                "reversal_uncertainty_count": 5,  # Less reversal
                "ira_count": 0,
                "obbba_count": 0,
            }
        ]

        result = indexer.calculate_raw_index_values(raw_counts)

        # Direction = (15 - 5) / (15 + 5) = 10 / 20 = 0.5
        assert result[0]["cpu_direction"] == pytest.approx(0.5)

    def test_calculate_raw_index_values_zero_denominator(self):
        """calculate_raw_index_values should skip zero denominator."""
        import indexer

        raw_counts = [
            {"month": "2024-01", "total_articles": 0},
            {"month": "2024-02", "total_articles": 100, "uncertainty_count": 10},
        ]

        result = indexer.calculate_raw_index_values(raw_counts)

        assert len(result) == 1
        assert result[0]["month"] == "2024-02"

    def test_calculate_raw_index_values_empty_input(self):
        """calculate_raw_index_values should handle empty input."""
        import indexer

        result = indexer.calculate_raw_index_values([])
        assert result == []


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

        raw = [{"month": "2024-01", "raw_ratio_cpu": 0.25}]
        result = indexer.normalize_index(raw)

        assert result[0]["normalized_cpu"] == 100.0

    def test_normalize_index_preserves_relative_values(self):
        """normalize_index should preserve relative differences."""
        import indexer

        raw = [
            {"month": "2024-01", "raw_ratio_cpu": 0.10},  # Low
            {"month": "2024-02", "raw_ratio_cpu": 0.20},  # Mean
            {"month": "2024-03", "raw_ratio_cpu": 0.30},  # High
        ]

        result = indexer.normalize_index(raw)

        # Mean = 0.20, so 0.10 -> 50, 0.20 -> 100, 0.30 -> 150
        assert abs(result[0]["normalized_cpu"] - 50.0) < 0.01
        assert abs(result[1]["normalized_cpu"] - 100.0) < 0.01
        assert abs(result[2]["normalized_cpu"] - 150.0) < 0.01

    def test_normalize_index_with_base_period(self):
        """normalize_index should use base period for mean calculation."""
        import indexer

        raw = [
            {"month": "2024-01", "raw_ratio_cpu": 0.10},
            {"month": "2024-02", "raw_ratio_cpu": 0.10},
            {"month": "2024-03", "raw_ratio_cpu": 0.20},  # Outside base
        ]

        result = indexer.normalize_index(raw, base_start="2024-01", base_end="2024-02")

        # Base mean = 0.10, so 0.20 -> 200
        assert result[0]["normalized_cpu"] == 100.0
        assert result[1]["normalized_cpu"] == 100.0
        assert result[2]["normalized_cpu"] == 200.0

    def test_normalize_index_handles_zero_mean(self):
        """normalize_index should handle all-zero ratios."""
        import indexer

        raw = [
            {"month": "2024-01", "raw_ratio_cpu": 0.0},
            {"month": "2024-02", "raw_ratio_cpu": 0.0},
        ]

        result = indexer.normalize_index(raw)

        assert result[0]["normalized_cpu"] == 0.0
        assert result[1]["normalized_cpu"] == 0.0

    def test_normalize_index_all_types(self):
        """normalize_index should normalize all index types."""
        import indexer

        raw = [
            {
                "month": "2024-01",
                "raw_ratio_cpu": 0.10,
                "raw_ratio_impl": 0.05,
                "raw_ratio_reversal": 0.03,
                "raw_ratio_ira": 0.02,
                "raw_ratio_obbba": 0.01,
            },
            {
                "month": "2024-02",
                "raw_ratio_cpu": 0.20,
                "raw_ratio_impl": 0.10,
                "raw_ratio_reversal": 0.06,
                "raw_ratio_ira": 0.04,
                "raw_ratio_obbba": 0.02,
            },
        ]

        result = indexer.normalize_index(raw)

        # All types should be normalized
        assert "normalized_cpu" in result[0]
        assert "normalized_impl" in result[0]
        assert "normalized_reversal" in result[0]
        assert "normalized_ira" in result[0]
        assert "normalized_obbba" in result[0]


class TestBuildIndex:
    """Tests for complete index building."""

    @patch("indexer.db_postgres.get_classification_counts_by_month")
    @patch("indexer.db_postgres.save_index_value")
    def test_build_index_no_data(self, mock_save, mock_get_counts):
        """build_index should return error with no data."""
        import indexer

        mock_get_counts.return_value = []

        result = indexer.build_index()

        assert result["status"] == "error"
        assert "No data" in result["message"]

    @patch("indexer.db_postgres.get_classification_counts_by_month")
    @patch("indexer.db_postgres.save_index_value")
    def test_build_index_success(self, mock_save, mock_get_counts):
        """build_index should return success with data."""
        import indexer

        mock_get_counts.return_value = [
            {
                "month": "2024-01",
                "total_articles": 100,
                "uncertainty_count": 20,
                "implementation_uncertainty_count": 10,
                "reversal_uncertainty_count": 5,
                "ira_count": 3,
                "obbba_count": 1,
            },
            {
                "month": "2024-02",
                "total_articles": 120,
                "uncertainty_count": 30,
                "implementation_uncertainty_count": 15,
                "reversal_uncertainty_count": 8,
                "ira_count": 5,
                "obbba_count": 2,
            },
        ]

        result = indexer.build_index()

        assert result["status"] == "success"
        assert "metadata" in result
        assert "series" in result
        assert result["metadata"]["num_months"] == 2

    @patch("indexer.db_postgres.get_classification_counts_by_month")
    @patch("indexer.db_postgres.save_index_value")
    def test_build_index_saves_to_database(self, mock_save, mock_get_counts):
        """build_index should save values to database."""
        import indexer

        mock_get_counts.return_value = [
            {
                "month": "2024-01",
                "total_articles": 100,
                "uncertainty_count": 20,
                "implementation_uncertainty_count": 10,
                "reversal_uncertainty_count": 5,
                "ira_count": 3,
                "obbba_count": 1,
            },
        ]

        indexer.build_index(save_to_db=True)

        # Should save multiple index types
        assert mock_save.call_count >= 5  # CPU, impl, reversal, ira, obbba

    @patch("indexer.db_postgres.get_classification_counts_by_month")
    @patch("indexer.db_postgres.save_index_value")
    def test_build_index_no_save_option(self, mock_save, mock_get_counts):
        """build_index should not save when save_to_db=False."""
        import indexer

        mock_get_counts.return_value = [
            {
                "month": "2024-01",
                "total_articles": 100,
                "uncertainty_count": 20,
            },
        ]

        indexer.build_index(save_to_db=False)

        mock_save.assert_not_called()


class TestBuildOutletLevelIndex:
    """Tests for outlet-level index building."""

    @patch("indexer.db_postgres.get_classification_counts_by_outlet")
    @patch("indexer.db_postgres.save_index_value")
    def test_build_outlet_level_no_data(self, mock_save, mock_get_outlet):
        """build_outlet_level_index should return error with no data."""
        import indexer

        mock_get_outlet.return_value = []

        result = indexer.build_outlet_level_index()

        assert result["status"] == "error"

    @patch("indexer.db_postgres.get_classification_counts_by_outlet")
    @patch("indexer.db_postgres.save_index_value")
    def test_build_outlet_level_success(self, mock_save, mock_get_outlet):
        """build_outlet_level_index should process multiple outlets."""
        import indexer

        mock_get_outlet.return_value = [
            {"outlet": "NYT", "month": "2024-01", "total_articles": 100, "uncertainty_count": 20},
            {"outlet": "NYT", "month": "2024-02", "total_articles": 110, "uncertainty_count": 25},
            {"outlet": "WSJ", "month": "2024-01", "total_articles": 80, "uncertainty_count": 15},
            {"outlet": "WSJ", "month": "2024-02", "total_articles": 90, "uncertainty_count": 18},
        ]

        result = indexer.build_outlet_level_index()

        assert result["status"] == "success"
        assert result["num_outlets"] == 2
        assert "NYT" in result["outlets"]
        assert "WSJ" in result["outlets"]

    @patch("indexer.db_postgres.get_classification_counts_by_outlet")
    @patch("indexer.db_postgres.save_index_value")
    def test_build_outlet_level_filter_outlets(self, mock_save, mock_get_outlet):
        """build_outlet_level_index should filter to specified outlets."""
        import indexer

        mock_get_outlet.return_value = [
            {"outlet": "NYT", "month": "2024-01", "total_articles": 100, "uncertainty_count": 20},
            {"outlet": "WSJ", "month": "2024-01", "total_articles": 80, "uncertainty_count": 15},
            {"outlet": "Reuters", "month": "2024-01", "total_articles": 60, "uncertainty_count": 10},
        ]

        result = indexer.build_outlet_level_index(outlets=["NYT", "WSJ"])

        assert result["status"] == "success"
        assert result["num_outlets"] == 2
        assert "Reuters" not in result["outlets"]


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

    @patch("indexer.db_postgres.get_index_values")
    def test_validate_against_events_no_data(self, mock_get_values):
        """validate_against_events should return error with no index."""
        import indexer

        mock_get_values.return_value = []

        result = indexer.validate_against_events()

        assert result["status"] == "error"
        assert "No index values" in result["message"]

    @patch("indexer.db_postgres.get_index_values")
    def test_validate_against_events_structure(self, mock_get_values):
        """validate_against_events should return expected structure."""
        import indexer

        mock_get_values.return_value = [
            {"month": "2022-06", "normalized": 100.0},
            {"month": "2022-07", "normalized": 150.0},  # Spike expected
            {"month": "2022-08", "normalized": 80.0},  # Drop expected
        ]

        result = indexer.validate_against_events()

        assert "events" in result
        assert "summary" in result
        assert isinstance(result["events"], list)


class TestGetIndexSummary:
    """Tests for index summary statistics."""

    @patch("indexer.db_postgres.get_index_values")
    def test_get_index_summary_empty(self, mock_get_values):
        """get_index_summary should handle empty database."""
        import indexer

        mock_get_values.return_value = []

        result = indexer.get_index_summary()

        assert result["status"] == "empty"

    @patch("indexer.db_postgres.get_index_values")
    def test_get_index_summary_with_data(self, mock_get_values):
        """get_index_summary should return statistics."""
        import indexer

        mock_get_values.return_value = [
            {"month": "2024-01", "normalized": 80.0},
            {"month": "2024-02", "normalized": 100.0},
            {"month": "2024-03", "normalized": 120.0},
        ]

        result = indexer.get_index_summary()

        assert result["status"] == "ready"
        assert result["mean"] == 100.0
        assert result["min"] == 80.0
        assert result["max"] == 120.0

    @patch("indexer.db_postgres.get_index_values")
    def test_get_index_summary_peaks_and_troughs(self, mock_get_values):
        """get_index_summary should identify peaks and troughs."""
        import indexer

        mock_get_values.return_value = [
            {"month": "2024-01", "normalized": 80.0},
            {"month": "2024-02", "normalized": 100.0},
            {"month": "2024-03", "normalized": 120.0},
        ]

        result = indexer.get_index_summary()

        assert "top_3_peaks" in result
        assert "top_3_troughs" in result
        assert result["top_3_peaks"][0]["month"] == "2024-03"
        assert result["top_3_troughs"][0]["month"] == "2024-01"


class TestCompareIndexTypes:
    """Tests for index type comparison."""

    @patch("indexer.db_postgres.get_index_values")
    def test_compare_index_types_no_data(self, mock_get_values):
        """compare_index_types should return error with missing data."""
        import indexer

        mock_get_values.return_value = []

        result = indexer.compare_index_types()

        assert result["status"] == "error"

    @patch("indexer.db_postgres.get_index_values")
    def test_compare_index_types_success(self, mock_get_values):
        """compare_index_types should calculate correlations."""
        import indexer

        # Mock returns for different index types
        def side_effect(index_type):
            if index_type == "CPU":
                return [
                    {"month": "2024-01", "normalized": 100.0},
                    {"month": "2024-02", "normalized": 120.0},
                ]
            elif index_type == "CPU_impl":
                return [
                    {"month": "2024-01", "normalized": 90.0},
                    {"month": "2024-02", "normalized": 110.0},
                ]
            elif index_type == "CPU_reversal":
                return [
                    {"month": "2024-01", "normalized": 110.0},
                    {"month": "2024-02", "normalized": 130.0},
                ]
            return []

        mock_get_values.side_effect = side_effect

        result = indexer.compare_index_types()

        assert result["status"] == "success"
        assert "correlations" in result
        assert "cpu_impl" in result["correlations"]
        assert "cpu_reversal" in result["correlations"]


class TestLegacyCompatibility:
    """Tests for legacy function compatibility."""

    @patch("indexer.db_postgres.get_classification_counts_by_month")
    def test_calculate_raw_index_legacy(self, mock_get_counts):
        """calculate_raw_index should return legacy format."""
        import indexer

        mock_get_counts.return_value = [
            {
                "month": "2024-01",
                "total_articles": 100,
                "uncertainty_count": 20,
                "implementation_uncertainty_count": 10,
                "reversal_uncertainty_count": 5,
            },
        ]

        result = indexer.calculate_raw_index()

        # Legacy format should have old field names
        assert len(result) == 1
        assert result[0]["numerator"] == 20  # was uncertainty_count
        assert result[0]["raw_ratio"] == 0.20
        assert result[0]["numerator_up"] == 10  # impl mapped to up
        assert result[0]["numerator_down"] == 5  # reversal mapped to down

    @patch("indexer.db_postgres.get_index_values")
    def test_get_all_index_values_legacy(self, mock_get_values):
        """get_all_index_values should call db_postgres correctly."""
        import indexer

        mock_get_values.return_value = [{"month": "2024-01", "normalized": 100.0}]

        result = indexer.get_all_index_values()

        mock_get_values.assert_called_once_with("CPU")
        assert len(result) == 1


class TestIndexConstants:
    """Tests for index type constants."""

    def test_index_constants_defined(self):
        """Index type constants should be defined."""
        import indexer

        assert indexer.INDEX_CPU == "CPU"
        assert indexer.INDEX_CPU_IMPL == "CPU_impl"
        assert indexer.INDEX_CPU_REVERSAL == "CPU_reversal"
        assert indexer.INDEX_SALIENCE_IRA == "salience_ira"
        assert indexer.INDEX_SALIENCE_OBBBA == "salience_obbba"
