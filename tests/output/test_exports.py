"""
Tests for exports.py - CSV export functions for CPU Index deliverables.

Following BBD/policyuncertainty.com CSV conventions.
"""

import csv
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cpu_index.output.exports import (
    export_monthly_index,
    export_decomposition,
    export_salience,
    export_outlet_robustness,
    export_all_csvs,
)


class TestExportMonthlyIndex:
    """Tests for export_monthly_index function."""

    def test_export_creates_csv_file(self, tmp_path):
        """export_monthly_index should create a CSV file."""
        index_data = {
            "2021-01": {"cpu": 100.0, "denominator": 1000},
            "2021-02": {"cpu": 105.5, "denominator": 1100},
        }

        output_path = tmp_path / "cpu_monthly.csv"
        result = export_monthly_index(index_data, output_path)

        assert output_path.exists()
        assert result == output_path

    def test_export_csv_has_correct_columns(self, tmp_path):
        """CSV should have date,cpu,denominator columns."""
        index_data = {
            "2021-01": {"cpu": 100.0, "denominator": 1000},
        }

        output_path = tmp_path / "cpu_monthly.csv"
        export_monthly_index(index_data, output_path)

        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert "date" in row
            assert "cpu" in row
            assert "denominator" in row

    def test_export_csv_values_correct(self, tmp_path):
        """CSV values should match input data."""
        index_data = {
            "2021-01": {"cpu": 98.5, "denominator": 1250},
            "2021-02": {"cpu": 102.3, "denominator": 1300},
        }

        output_path = tmp_path / "cpu_monthly.csv"
        export_monthly_index(index_data, output_path)

        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["date"] == "2021-01"
        assert float(rows[0]["cpu"]) == 98.5
        assert int(rows[0]["denominator"]) == 1250

    def test_export_csv_sorted_by_date(self, tmp_path):
        """CSV rows should be sorted by date."""
        # Input in wrong order
        index_data = {
            "2021-03": {"cpu": 110.0, "denominator": 1200},
            "2021-01": {"cpu": 100.0, "denominator": 1000},
            "2021-02": {"cpu": 105.0, "denominator": 1100},
        }

        output_path = tmp_path / "cpu_monthly.csv"
        export_monthly_index(index_data, output_path)

        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            dates = [row["date"] for row in reader]

        assert dates == ["2021-01", "2021-02", "2021-03"]

    def test_export_handles_empty_data(self, tmp_path):
        """Should handle empty data gracefully."""
        output_path = tmp_path / "cpu_monthly.csv"
        result = export_monthly_index({}, output_path)

        assert output_path.exists()
        with open(output_path, "r") as f:
            content = f.read()
        # Should have header only
        assert "date" in content


class TestExportDecomposition:
    """Tests for export_decomposition function."""

    def test_export_decomposition_creates_file(self, tmp_path):
        """export_decomposition should create CSV with impl/reversal breakdown."""
        index_data = {
            "2021-01": {"cpu": 100.0, "cpu_impl": 45.0, "cpu_reversal": 55.0, "denominator": 1000},
        }

        output_path = tmp_path / "cpu_decomposition.csv"
        result = export_decomposition(index_data, output_path)

        assert output_path.exists()
        assert result == output_path

    def test_export_decomposition_columns(self, tmp_path):
        """CSV should have date,cpu,cpu_impl,cpu_reversal,denominator."""
        index_data = {
            "2021-01": {"cpu": 100.0, "cpu_impl": 45.0, "cpu_reversal": 55.0, "denominator": 1000},
        }

        output_path = tmp_path / "cpu_decomposition.csv"
        export_decomposition(index_data, output_path)

        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert "date" in row
        assert "cpu" in row
        assert "cpu_impl" in row
        assert "cpu_reversal" in row
        assert "denominator" in row


class TestExportSalience:
    """Tests for export_salience function."""

    def test_export_salience_creates_file(self, tmp_path):
        """export_salience should create CSV with IRA/OBBBA counts."""
        index_data = {
            "2021-01": {"salience_ira": 15, "salience_obbba": 0, "denominator": 1000},
            "2025-02": {"salience_ira": 25, "salience_obbba": 45, "denominator": 1200},
        }

        output_path = tmp_path / "cpu_salience.csv"
        result = export_salience(index_data, output_path)

        assert output_path.exists()

    def test_export_salience_columns(self, tmp_path):
        """CSV should have date,salience_ira,salience_obbba,denominator."""
        index_data = {
            "2021-01": {"salience_ira": 15, "salience_obbba": 0, "denominator": 1000},
        }

        output_path = tmp_path / "cpu_salience.csv"
        export_salience(index_data, output_path)

        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert "date" in row
        assert "salience_ira" in row
        assert "salience_obbba" in row


class TestExportOutletRobustness:
    """Tests for export_outlet_robustness function."""

    def test_export_outlet_robustness_creates_file(self, tmp_path):
        """export_outlet_robustness should create CSV with outlet-level indices."""
        outlet_data = {
            "New York Times": {"2021-01": 98.5, "2021-02": 102.3},
            "Wall Street Journal": {"2021-01": 101.2, "2021-02": 99.8},
        }

        output_path = tmp_path / "cpu_robustness.csv"
        result = export_outlet_robustness(outlet_data, output_path)

        assert output_path.exists()

    def test_export_outlet_robustness_wide_format(self, tmp_path):
        """CSV should have date column + one column per outlet."""
        outlet_data = {
            "NYT": {"2021-01": 98.5},
            "WSJ": {"2021-01": 101.2},
        }

        output_path = tmp_path / "cpu_robustness.csv"
        export_outlet_robustness(outlet_data, output_path)

        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

        assert "date" in fieldnames
        assert "NYT" in fieldnames
        assert "WSJ" in fieldnames


class TestExportAllCSVs:
    """Tests for export_all_csvs convenience function."""

    def test_export_all_csvs_creates_all_files(self, tmp_path):
        """export_all_csvs should create all 4 CSV files."""
        index_data = {
            "2021-01": {
                "cpu": 100.0,
                "cpu_impl": 45.0,
                "cpu_reversal": 55.0,
                "salience_ira": 10,
                "salience_obbba": 0,
                "denominator": 1000,
            },
        }
        outlet_data = {"NYT": {"2021-01": 100.0}}

        result = export_all_csvs(index_data, outlet_data, tmp_path)

        assert (tmp_path / "cpu_monthly.csv").exists()
        assert (tmp_path / "cpu_decomposition.csv").exists()
        assert (tmp_path / "cpu_salience.csv").exists()
        assert (tmp_path / "cpu_robustness.csv").exists()
        assert len(result) == 4


class TestUTF8Encoding:
    """Tests for proper UTF-8 encoding."""

    def test_export_uses_utf8(self, tmp_path):
        """All exports should use UTF-8 encoding."""
        index_data = {"2021-01": {"cpu": 100.0, "denominator": 1000}}
        output_path = tmp_path / "test.csv"

        export_monthly_index(index_data, output_path)

        # Should be readable as UTF-8
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "date" in content


class TestNaNHandling:
    """Tests for handling missing/NaN values."""

    def test_export_handles_nan_values(self, tmp_path):
        """Should handle NaN values gracefully."""
        import math

        index_data = {
            "2021-01": {"cpu": 100.0, "denominator": 1000},
            "2021-02": {"cpu": float("nan"), "denominator": 0},
        }

        output_path = tmp_path / "test.csv"
        export_monthly_index(index_data, output_path)

        with open(output_path, "r") as f:
            content = f.read()
        # NaN should be exported as empty or "NaN"
        assert output_path.exists()
