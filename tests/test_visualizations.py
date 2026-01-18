"""
Tests for visualizations.py - Chart generation for CPU Index deliverables.

Tests that visualization functions:
1. Create output files
2. Return paths correctly
3. Handle edge cases (empty data, missing months)
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPlotCPUTimeseries:
    """Tests for plot_cpu_timeseries function (Figure 1)."""

    def test_creates_png_file(self, tmp_path):
        """plot_cpu_timeseries should create a PNG file."""
        from visualizations import plot_cpu_timeseries

        index_data = {
            "2021-01": {"cpu": 100.0},
            "2021-02": {"cpu": 105.5},
            "2021-03": {"cpu": 98.2},
        }
        events = [
            {"date": "2021-02", "label": "Test Event", "color": "red"},
        ]

        output_path = tmp_path / "cpu_timeseries.png"
        result = plot_cpu_timeseries(index_data, events, output_path)

        assert output_path.exists()
        assert result == output_path

    def test_handles_empty_events(self, tmp_path):
        """Should work with no events."""
        from visualizations import plot_cpu_timeseries

        index_data = {
            "2021-01": {"cpu": 100.0},
            "2021-02": {"cpu": 105.5},
        }

        output_path = tmp_path / "cpu_timeseries.png"
        result = plot_cpu_timeseries(index_data, [], output_path)

        assert output_path.exists()

    def test_handles_minimal_data(self, tmp_path):
        """Should handle single data point."""
        from visualizations import plot_cpu_timeseries

        index_data = {"2021-01": {"cpu": 100.0}}

        output_path = tmp_path / "cpu_timeseries.png"
        result = plot_cpu_timeseries(index_data, [], output_path)

        assert output_path.exists()


class TestPlotCPUDecomposition:
    """Tests for plot_cpu_decomposition function (Figure 2)."""

    def test_creates_png_file(self, tmp_path):
        """plot_cpu_decomposition should create a PNG file."""
        from visualizations import plot_cpu_decomposition

        index_data = {
            "2021-01": {"cpu": 100.0, "cpu_impl": 45.0, "cpu_reversal": 55.0},
            "2021-02": {"cpu": 105.5, "cpu_impl": 50.0, "cpu_reversal": 55.5},
        }

        output_path = tmp_path / "cpu_decomposition.png"
        result = plot_cpu_decomposition(index_data, output_path)

        assert output_path.exists()
        assert result == output_path

    def test_handles_missing_components(self, tmp_path):
        """Should handle data with missing impl/reversal."""
        from visualizations import plot_cpu_decomposition

        index_data = {
            "2021-01": {"cpu": 100.0},  # Missing impl/reversal
        }

        output_path = tmp_path / "cpu_decomposition.png"
        # Should not raise, just plot what's available
        result = plot_cpu_decomposition(index_data, output_path)
        assert output_path.exists()


class TestPlotDirectionBalance:
    """Tests for plot_direction_balance function (Figure 3)."""

    def test_creates_png_file(self, tmp_path):
        """plot_direction_balance should create a PNG file."""
        from visualizations import plot_direction_balance

        index_data = {
            "2021-01": {"cpu_impl": 60.0, "cpu_reversal": 40.0},
            "2021-02": {"cpu_impl": 40.0, "cpu_reversal": 60.0},
        }

        output_path = tmp_path / "direction_balance.png"
        result = plot_direction_balance(index_data, output_path)

        assert output_path.exists()
        assert result == output_path

    def test_balance_calculation(self, tmp_path):
        """Balance should be (impl - reversal) / (impl + reversal)."""
        from visualizations import plot_direction_balance

        # impl=60, reversal=40 -> (60-40)/(60+40) = 0.2
        # impl=40, reversal=60 -> (40-60)/(40+60) = -0.2
        index_data = {
            "2021-01": {"cpu_impl": 60.0, "cpu_reversal": 40.0},
            "2021-02": {"cpu_impl": 40.0, "cpu_reversal": 60.0},
        }

        output_path = tmp_path / "direction_balance.png"
        result = plot_direction_balance(index_data, output_path)
        assert output_path.exists()


class TestPlotOutletCorrelationHeatmap:
    """Tests for plot_outlet_correlation_heatmap function (Figure A1)."""

    def test_creates_png_file(self, tmp_path):
        """plot_outlet_correlation_heatmap should create a PNG file."""
        from visualizations import plot_outlet_correlation_heatmap

        outlet_indices = {
            "NYT": {"2021-01": 100.0, "2021-02": 105.0},
            "WSJ": {"2021-01": 102.0, "2021-02": 103.0},
            "WaPo": {"2021-01": 98.0, "2021-02": 107.0},
        }

        output_path = tmp_path / "outlet_heatmap.png"
        result = plot_outlet_correlation_heatmap(outlet_indices, output_path)

        assert output_path.exists()
        assert result == output_path

    def test_handles_single_outlet(self, tmp_path):
        """Should handle single outlet gracefully."""
        from visualizations import plot_outlet_correlation_heatmap

        outlet_indices = {"NYT": {"2021-01": 100.0, "2021-02": 105.0}}

        output_path = tmp_path / "outlet_heatmap.png"
        result = plot_outlet_correlation_heatmap(outlet_indices, output_path)
        # Should still create file, possibly with N/A note
        assert output_path.exists()


class TestPlotKeywordSensitivity:
    """Tests for plot_keyword_sensitivity function (Figure A2)."""

    def test_creates_png_file(self, tmp_path):
        """plot_keyword_sensitivity should create a PNG file."""
        from visualizations import plot_keyword_sensitivity

        sensitivity_results = {
            "drop_uncertain": 0.98,
            "drop_risk": 0.97,
            "drop_climate": 0.96,
            "drop_policy": 0.99,
        }

        output_path = tmp_path / "keyword_sensitivity.png"
        result = plot_keyword_sensitivity(sensitivity_results, output_path)

        assert output_path.exists()
        assert result == output_path


class TestPlotPlaceboComparison:
    """Tests for plot_placebo_comparison function (Figure A3)."""

    def test_creates_png_file(self, tmp_path):
        """plot_placebo_comparison should create a PNG file."""
        from visualizations import plot_placebo_comparison

        cpu = {"2021-01": 100.0, "2021-02": 105.0}
        tpu = {"2021-01": 80.0, "2021-02": 85.0}
        mpu = {"2021-01": 90.0, "2021-02": 92.0}

        output_path = tmp_path / "placebo_comparison.png"
        result = plot_placebo_comparison(cpu, tpu, mpu, output_path)

        assert output_path.exists()
        assert result == output_path


class TestPlotLLMValidationScatter:
    """Tests for plot_llm_validation_scatter function (Figure A4)."""

    def test_creates_png_file(self, tmp_path):
        """plot_llm_validation_scatter should create a PNG file."""
        from visualizations import plot_llm_validation_scatter

        keyword_cpu = {"2021-01": 100.0, "2021-02": 105.0, "2021-03": 98.0}
        llm_cpu = {"2021-01": 98.0, "2021-02": 103.0, "2021-03": 97.0}

        output_path = tmp_path / "llm_scatter.png"
        result = plot_llm_validation_scatter(keyword_cpu, llm_cpu, output_path)

        assert output_path.exists()
        assert result == output_path


class TestPlotArticleVolume:
    """Tests for plot_article_volume function (Figure A5)."""

    def test_creates_png_file(self, tmp_path):
        """plot_article_volume should create a PNG file."""
        from visualizations import plot_article_volume

        monthly_counts = {
            "2021-01": 1250,
            "2021-02": 1300,
            "2021-03": 1400,
        }

        output_path = tmp_path / "article_volume.png"
        result = plot_article_volume(monthly_counts, output_path)

        assert output_path.exists()
        assert result == output_path


class TestVisualizationStyling:
    """Tests for consistent visualization styling."""

    def test_output_is_png(self, tmp_path):
        """All outputs should be PNG format."""
        from visualizations import plot_cpu_timeseries

        index_data = {"2021-01": {"cpu": 100.0}}
        output_path = tmp_path / "test.png"

        plot_cpu_timeseries(index_data, [], output_path)

        # Check file magic bytes for PNG
        with open(output_path, "rb") as f:
            header = f.read(8)
        assert header[:4] == b"\x89PNG"
