"""
Tests for report_generator.py - Full deliverable package generation.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGenerateFullReport:
    """Tests for generate_full_report function."""

    def test_creates_all_outputs(self, tmp_path):
        """generate_full_report should create all expected files."""
        from report_generator import generate_full_report

        index_data = {
            "2021-01": {"cpu": 100.0, "cpu_impl": 45.0, "cpu_reversal": 55.0,
                       "salience_ira": 0, "salience_obbba": 0, "denominator": 1000},
            "2021-02": {"cpu": 105.0, "cpu_impl": 50.0, "cpu_reversal": 55.0,
                       "salience_ira": 5, "salience_obbba": 0, "denominator": 1100},
        }

        outlet_indices = {
            "NYT": {"2021-01": 98.0, "2021-02": 107.0},
            "WSJ": {"2021-01": 102.0, "2021-02": 103.0},
        }

        result = generate_full_report(
            index_data=index_data,
            outlet_indices=outlet_indices,
            output_dir=tmp_path,
        )

        # Check CSVs
        assert len(result["csvs"]) == 4
        assert (tmp_path / "csv" / "cpu_monthly.csv").exists()
        assert (tmp_path / "csv" / "cpu_decomposition.csv").exists()
        assert (tmp_path / "csv" / "cpu_salience.csv").exists()
        assert (tmp_path / "csv" / "cpu_robustness.csv").exists()

        # Check figures
        assert len(result["figures"]) >= 5  # 3 main + 2 appendix minimum
        assert (tmp_path / "figures" / "fig1_cpu_timeseries.png").exists()
        assert (tmp_path / "figures" / "fig2_cpu_decomposition.png").exists()
        assert (tmp_path / "figures" / "fig3_direction_balance.png").exists()

        # Check memo
        assert result["memo"] is not None
        assert (tmp_path / "cpu_methodology_memo.md").exists()

    def test_returns_dict_with_paths(self, tmp_path):
        """Result should be dict with csvs, figures, memo keys."""
        from report_generator import generate_full_report

        index_data = {"2021-01": {"cpu": 100.0, "denominator": 1000}}
        outlet_indices = {"NYT": {"2021-01": 100.0}}

        result = generate_full_report(
            index_data=index_data,
            outlet_indices=outlet_indices,
            output_dir=tmp_path,
        )

        assert "csvs" in result
        assert "figures" in result
        assert "memo" in result
        assert isinstance(result["csvs"], list)
        assert isinstance(result["figures"], list)
        assert isinstance(result["memo"], str)

    def test_creates_output_directories(self, tmp_path):
        """Should create csv and figures subdirectories."""
        from report_generator import generate_full_report

        index_data = {"2021-01": {"cpu": 100.0, "denominator": 1000}}
        outlet_indices = {"NYT": {"2021-01": 100.0}}

        generate_full_report(
            index_data=index_data,
            outlet_indices=outlet_indices,
            output_dir=tmp_path,
        )

        assert (tmp_path / "csv").is_dir()
        assert (tmp_path / "figures").is_dir()

    def test_memo_contains_keyword_lists(self, tmp_path):
        """Generated memo should include keyword lists."""
        from report_generator import generate_full_report

        index_data = {"2021-01": {"cpu": 100.0, "denominator": 1000}}
        outlet_indices = {"NYT": {"2021-01": 100.0}}

        generate_full_report(
            index_data=index_data,
            outlet_indices=outlet_indices,
            output_dir=tmp_path,
        )

        memo_path = tmp_path / "cpu_methodology_memo.md"
        with open(memo_path, "r") as f:
            content = f.read()

        # Should contain actual keywords, not placeholders
        assert "climate" in content.lower()
        assert "policy" in content.lower()
        assert "{{" not in content  # No unfilled placeholders for keywords

    def test_handles_optional_data(self, tmp_path):
        """Should work with optional sensitivity/placebo/llm data."""
        from report_generator import generate_full_report

        index_data = {
            "2021-01": {"cpu": 100.0, "cpu_impl": 50.0, "cpu_reversal": 50.0, "denominator": 1000},
            "2021-02": {"cpu": 110.0, "cpu_impl": 55.0, "cpu_reversal": 55.0, "denominator": 1100},
        }
        outlet_indices = {"NYT": {"2021-01": 100.0, "2021-02": 110.0}}

        sensitivity = {"drop_climate": 0.98, "drop_risk": 0.97}
        placebo = {"tpu": {"2021-01": 80.0, "2021-02": 85.0}, "mpu": {"2021-01": 90.0, "2021-02": 95.0}}
        llm = {"keyword_cpu": {"2021-01": 100.0, "2021-02": 110.0}, "llm_cpu": {"2021-01": 98.0, "2021-02": 108.0}}

        result = generate_full_report(
            index_data=index_data,
            outlet_indices=outlet_indices,
            output_dir=tmp_path,
            sensitivity_results=sensitivity,
            placebo_data=placebo,
            llm_data=llm,
        )

        # Should have additional figures
        assert len(result["figures"]) >= 7  # 3 main + 4 appendix
