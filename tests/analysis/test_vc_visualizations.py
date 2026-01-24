"""
Tests for vc_visualizations.py - CPU-VC correlation visualization module

Tests cover:
1. plot_cpu_vc_timeseries() - Dual-axis time series plot
2. plot_cross_correlation() - CCF bar plot with confidence bands
3. plot_rolling_correlation() - Rolling correlation with uncertainty
4. plot_stage_distribution() - Stacked area chart of deal stages
5. save_all_visualizations() - Batch export of all visualizations

Tests verify:
- Correct return types (Figure vs Path)
- File creation when output_path is provided
- Handling of different input formats (DataFrame, dict)
- Edge cases and error conditions
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from cpu_index.analysis.vc_visualizations import (
    plot_cpu_vc_timeseries,
    plot_cross_correlation,
    plot_rolling_correlation,
    plot_stage_distribution,
    save_all_visualizations,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_cpu_series():
    """Sample CPU index series with datetime index."""
    dates = pd.date_range("2020-01", periods=24, freq="MS")
    np.random.seed(42)
    values = 100 + np.cumsum(np.random.randn(24) * 5)
    return pd.Series(values, index=dates, name="cpu_index")


@pytest.fixture
def sample_vc_series():
    """Sample VC deal count series with datetime index."""
    dates = pd.date_range("2020-01", periods=24, freq="MS")
    np.random.seed(43)
    values = np.random.poisson(50, 24)
    return pd.Series(values, index=dates, name="vc_deal_count")


@pytest.fixture
def sample_ccf_dataframe():
    """Sample cross-correlation results DataFrame."""
    lags = list(range(-12, 13))
    np.random.seed(44)
    correlations = [np.random.uniform(-0.5, 0.5) for _ in lags]
    return pd.DataFrame({
        'lag': lags,
        'correlation': correlations,
        'n_observations': [40] * len(lags),
        'interpretation': ['test'] * len(lags),
    })


@pytest.fixture
def sample_ccf_dict(sample_ccf_dataframe):
    """Sample cross-correlation results as dict (from analyze_cpu_vc_correlation)."""
    return {
        'results': sample_ccf_dataframe.to_dict('records'),
        'summary': {
            'max_correlation': 0.45,
            'optimal_lag': 3,
        }
    }


@pytest.fixture
def sample_monthly_data():
    """Sample monthly VC data with stage columns."""
    dates = pd.date_range("2020-01", periods=24, freq="MS")
    np.random.seed(45)
    return pd.DataFrame({
        'deal_count': np.random.poisson(50, 24),
        'seed_count': np.random.poisson(20, 24),
        'early_count': np.random.poisson(20, 24),
        'late_count': np.random.poisson(10, 24),
        'total_amount': np.random.uniform(100, 500, 24),
    }, index=dates)


@pytest.fixture
def sample_monthly_data_with_period_index(sample_monthly_data):
    """Sample monthly VC data with PeriodIndex instead of DatetimeIndex."""
    df = sample_monthly_data.copy()
    df.index = df.index.to_period('M')
    return df


# =============================================================================
# Tests for plot_cpu_vc_timeseries()
# =============================================================================


class TestPlotCpuVcTimeseries:
    """Tests for CPU-VC time series visualization."""

    def test_returns_figure_when_no_output_path(self, sample_cpu_series, sample_vc_series):
        """plot_cpu_vc_timeseries should return Figure when output_path is None."""
        result = plot_cpu_vc_timeseries(sample_cpu_series, sample_vc_series)

        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_saves_to_file_when_path_given(
        self, sample_cpu_series, sample_vc_series, tmp_path
    ):
        """plot_cpu_vc_timeseries should save to file and return Path."""
        output_file = tmp_path / "timeseries.png"

        result = plot_cpu_vc_timeseries(
            sample_cpu_series, sample_vc_series, output_path=output_file
        )

        assert result == output_file
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_creates_parent_directories(
        self, sample_cpu_series, sample_vc_series, tmp_path
    ):
        """plot_cpu_vc_timeseries should create parent directories if needed."""
        output_file = tmp_path / "subdir" / "nested" / "timeseries.png"

        result = plot_cpu_vc_timeseries(
            sample_cpu_series, sample_vc_series, output_path=output_file
        )

        assert result == output_file
        assert output_file.exists()

    def test_with_custom_title(self, sample_cpu_series, sample_vc_series):
        """plot_cpu_vc_timeseries should use custom title."""
        fig = plot_cpu_vc_timeseries(
            sample_cpu_series,
            sample_vc_series,
            title="Custom Title"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_show_trend_false(self, sample_cpu_series, sample_vc_series):
        """plot_cpu_vc_timeseries should work with show_trend=False."""
        fig = plot_cpu_vc_timeseries(
            sample_cpu_series,
            sample_vc_series,
            show_trend=False
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_raises_on_no_overlap(self):
        """plot_cpu_vc_timeseries should raise ValueError when no overlapping data."""
        cpu = pd.Series([100, 110], index=pd.date_range("2020-01", periods=2, freq="MS"))
        vc = pd.Series([50, 60], index=pd.date_range("2025-01", periods=2, freq="MS"))

        with pytest.raises(ValueError, match="No overlapping data"):
            plot_cpu_vc_timeseries(cpu, vc)

    def test_handles_period_index(self, sample_cpu_series, sample_vc_series):
        """plot_cpu_vc_timeseries should handle PeriodIndex conversion."""
        cpu_period = sample_cpu_series.copy()
        cpu_period.index = cpu_period.index.to_period('M')

        fig = plot_cpu_vc_timeseries(cpu_period, sample_vc_series)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_handles_string_output_path(
        self, sample_cpu_series, sample_vc_series, tmp_path
    ):
        """plot_cpu_vc_timeseries should accept string output_path."""
        output_file = str(tmp_path / "timeseries.png")

        result = plot_cpu_vc_timeseries(
            sample_cpu_series, sample_vc_series, output_path=output_file
        )

        assert isinstance(result, Path)
        assert result.exists()


# =============================================================================
# Tests for plot_cross_correlation()
# =============================================================================


class TestPlotCrossCorrelation:
    """Tests for cross-correlation visualization."""

    def test_returns_figure_with_dataframe_input(self, sample_ccf_dataframe):
        """plot_cross_correlation should return Figure with DataFrame input."""
        result = plot_cross_correlation(sample_ccf_dataframe)

        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_returns_figure_with_dict_input(self, sample_ccf_dict):
        """plot_cross_correlation should return Figure with dict input."""
        result = plot_cross_correlation(sample_ccf_dict)

        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_saves_to_file_when_path_given(self, sample_ccf_dataframe, tmp_path):
        """plot_cross_correlation should save to file and return Path."""
        output_file = tmp_path / "ccf.png"

        result = plot_cross_correlation(sample_ccf_dataframe, output_path=output_file)

        assert result == output_file
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_dict_input_saves_to_file(self, sample_ccf_dict, tmp_path):
        """plot_cross_correlation with dict input should save to file."""
        output_file = tmp_path / "ccf_dict.png"

        result = plot_cross_correlation(sample_ccf_dict, output_path=output_file)

        assert result == output_file
        assert output_file.exists()

    def test_with_custom_title(self, sample_ccf_dataframe):
        """plot_cross_correlation should use custom title."""
        fig = plot_cross_correlation(
            sample_ccf_dataframe,
            title="Custom CCF Title"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_alpha(self, sample_ccf_dataframe):
        """plot_cross_correlation should use custom significance level."""
        fig = plot_cross_correlation(sample_ccf_dataframe, alpha=0.01)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_raises_on_empty_dataframe(self):
        """plot_cross_correlation should raise ValueError on empty input."""
        empty_df = pd.DataFrame({'lag': [], 'correlation': []})

        with pytest.raises(ValueError, match="Empty cross-correlation results"):
            plot_cross_correlation(empty_df)

    def test_raises_on_dict_without_results_key(self):
        """plot_cross_correlation should raise ValueError on invalid dict."""
        invalid_dict = {'summary': {'max_corr': 0.5}}

        with pytest.raises(ValueError, match="Dict must contain 'results' key"):
            plot_cross_correlation(invalid_dict)

    def test_handles_n_observations_column_missing(self, sample_ccf_dataframe):
        """plot_cross_correlation should handle missing n_observations column."""
        df = sample_ccf_dataframe.drop(columns=['n_observations'])

        fig = plot_cross_correlation(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Tests for plot_rolling_correlation()
# =============================================================================


class TestPlotRollingCorrelation:
    """Tests for rolling correlation visualization."""

    def test_returns_figure_with_default_window(
        self, sample_cpu_series, sample_vc_series
    ):
        """plot_rolling_correlation should return Figure with default window."""
        result = plot_rolling_correlation(sample_cpu_series, sample_vc_series)

        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_saves_to_file_when_path_given(
        self, sample_cpu_series, sample_vc_series, tmp_path
    ):
        """plot_rolling_correlation should save to file and return Path."""
        output_file = tmp_path / "rolling_corr.png"

        result = plot_rolling_correlation(
            sample_cpu_series, sample_vc_series, output_path=output_file
        )

        assert result == output_file
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_with_window_6(self, sample_cpu_series, sample_vc_series):
        """plot_rolling_correlation should work with window=6."""
        fig = plot_rolling_correlation(
            sample_cpu_series, sample_vc_series, window=6
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_window_24(self, sample_cpu_series, sample_vc_series):
        """plot_rolling_correlation should work with larger window."""
        fig = plot_rolling_correlation(
            sample_cpu_series, sample_vc_series, window=24
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_window_3(self, sample_cpu_series, sample_vc_series):
        """plot_rolling_correlation should work with small window=3."""
        fig = plot_rolling_correlation(
            sample_cpu_series, sample_vc_series, window=3
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_title(self, sample_cpu_series, sample_vc_series):
        """plot_rolling_correlation should use custom title."""
        fig = plot_rolling_correlation(
            sample_cpu_series,
            sample_vc_series,
            title="Custom Rolling Correlation"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_raises_on_insufficient_data(self, sample_cpu_series, sample_vc_series):
        """plot_rolling_correlation should raise on insufficient data for window."""
        # Use only 5 data points with window=12
        short_cpu = sample_cpu_series.iloc[:5]
        short_vc = sample_vc_series.iloc[:5]

        with pytest.raises(ValueError, match="Insufficient data for window"):
            plot_rolling_correlation(short_cpu, short_vc, window=12)

    def test_handles_period_index(self, sample_cpu_series, sample_vc_series):
        """plot_rolling_correlation should handle PeriodIndex conversion."""
        cpu_period = sample_cpu_series.copy()
        cpu_period.index = cpu_period.index.to_period('M')

        fig = plot_rolling_correlation(cpu_period, sample_vc_series, window=6)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Tests for plot_stage_distribution()
# =============================================================================


class TestPlotStageDistribution:
    """Tests for stage distribution visualization."""

    def test_returns_figure_when_no_output_path(self, sample_monthly_data):
        """plot_stage_distribution should return Figure when output_path is None."""
        result = plot_stage_distribution(sample_monthly_data)

        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_saves_to_file_when_path_given(self, sample_monthly_data, tmp_path):
        """plot_stage_distribution should save to file and return Path."""
        output_file = tmp_path / "stage_dist.png"

        result = plot_stage_distribution(sample_monthly_data, output_path=output_file)

        assert result == output_file
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_with_custom_title(self, sample_monthly_data):
        """plot_stage_distribution should use custom title."""
        fig = plot_stage_distribution(
            sample_monthly_data,
            title="Custom Stage Distribution"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_handles_period_index(self, sample_monthly_data_with_period_index):
        """plot_stage_distribution should handle PeriodIndex conversion."""
        fig = plot_stage_distribution(sample_monthly_data_with_period_index)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_raises_on_missing_columns(self, sample_monthly_data):
        """plot_stage_distribution should raise ValueError on missing columns."""
        incomplete_data = sample_monthly_data.drop(columns=['seed_count'])

        with pytest.raises(ValueError, match="Missing required columns"):
            plot_stage_distribution(incomplete_data)

    def test_raises_on_multiple_missing_columns(self, sample_monthly_data):
        """plot_stage_distribution should list all missing columns."""
        incomplete_data = sample_monthly_data.drop(
            columns=['seed_count', 'early_count']
        )

        with pytest.raises(ValueError, match="seed_count"):
            plot_stage_distribution(incomplete_data)

    def test_with_string_index(self, sample_monthly_data):
        """plot_stage_distribution should convert string index to datetime."""
        df = sample_monthly_data.copy()
        df.index = df.index.strftime('%Y-%m-%d')

        fig = plot_stage_distribution(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================================
# Tests for save_all_visualizations()
# =============================================================================


class TestSaveAllVisualizations:
    """Tests for batch visualization export."""

    def test_creates_all_expected_files(
        self,
        sample_cpu_series,
        sample_monthly_data,
        sample_ccf_dataframe,
        tmp_path
    ):
        """save_all_visualizations should create all expected files."""
        output_dir = tmp_path / "figures"

        result = save_all_visualizations(
            sample_cpu_series,
            sample_monthly_data,
            sample_ccf_dataframe,
            output_dir
        )

        # Check all expected keys are present
        expected_keys = [
            'timeseries',
            'cross_correlation',
            'rolling_correlation',
            'stage_distribution'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        # Check all files were created
        assert (output_dir / "cpu_vc_timeseries.png").exists()
        assert (output_dir / "cpu_vc_ccf.png").exists()
        assert (output_dir / "cpu_vc_rolling_corr.png").exists()
        assert (output_dir / "vc_stage_distribution.png").exists()

    def test_creates_output_directory(
        self,
        sample_cpu_series,
        sample_monthly_data,
        sample_ccf_dataframe,
        tmp_path
    ):
        """save_all_visualizations should create output directory if needed."""
        output_dir = tmp_path / "new_dir" / "figures"

        result = save_all_visualizations(
            sample_cpu_series,
            sample_monthly_data,
            sample_ccf_dataframe,
            output_dir
        )

        assert output_dir.exists()
        assert isinstance(result['timeseries'], Path)

    def test_returns_paths_dict(
        self,
        sample_cpu_series,
        sample_monthly_data,
        sample_ccf_dataframe,
        tmp_path
    ):
        """save_all_visualizations should return dict mapping names to paths."""
        output_dir = tmp_path / "figures"

        result = save_all_visualizations(
            sample_cpu_series,
            sample_monthly_data,
            sample_ccf_dataframe,
            output_dir
        )

        assert isinstance(result, dict)
        for key, value in result.items():
            if not isinstance(value, str):  # Error messages are strings
                assert isinstance(value, Path)
                assert value.exists()

    def test_with_dict_ccf_input(
        self,
        sample_cpu_series,
        sample_monthly_data,
        sample_ccf_dict,
        tmp_path
    ):
        """save_all_visualizations should accept dict ccf_results."""
        output_dir = tmp_path / "figures"

        result = save_all_visualizations(
            sample_cpu_series,
            sample_monthly_data,
            sample_ccf_dict,
            output_dir
        )

        assert (output_dir / "cpu_vc_ccf.png").exists()

    def test_with_custom_window(
        self,
        sample_cpu_series,
        sample_monthly_data,
        sample_ccf_dataframe,
        tmp_path
    ):
        """save_all_visualizations should use custom rolling window."""
        output_dir = tmp_path / "figures"

        result = save_all_visualizations(
            sample_cpu_series,
            sample_monthly_data,
            sample_ccf_dataframe,
            output_dir,
            window=6
        )

        assert (output_dir / "cpu_vc_rolling_corr.png").exists()

    def test_accepts_string_output_dir(
        self,
        sample_cpu_series,
        sample_monthly_data,
        sample_ccf_dataframe,
        tmp_path
    ):
        """save_all_visualizations should accept string output_dir."""
        output_dir = str(tmp_path / "figures")

        result = save_all_visualizations(
            sample_cpu_series,
            sample_monthly_data,
            sample_ccf_dataframe,
            output_dir
        )

        assert Path(output_dir).exists()

    def test_handles_partial_failures_gracefully(
        self,
        sample_cpu_series,
        sample_ccf_dataframe,
        tmp_path
    ):
        """save_all_visualizations should handle partial failures."""
        output_dir = tmp_path / "figures"

        # Monthly data missing stage columns - stage_distribution will fail
        incomplete_monthly = pd.DataFrame({
            'deal_count': np.random.poisson(50, 24),
            # Missing seed_count, early_count, late_count
        }, index=pd.date_range("2020-01", periods=24, freq="MS"))

        result = save_all_visualizations(
            sample_cpu_series,
            incomplete_monthly,
            sample_ccf_dataframe,
            output_dir
        )

        # Other visualizations should succeed
        assert (output_dir / "cpu_vc_timeseries.png").exists()
        assert (output_dir / "cpu_vc_ccf.png").exists()
        assert (output_dir / "cpu_vc_rolling_corr.png").exists()

        # stage_distribution should report error
        assert 'Error' in str(result['stage_distribution'])

    def test_uses_deal_count_column_from_monthly(
        self,
        sample_cpu_series,
        sample_monthly_data,
        sample_ccf_dataframe,
        tmp_path
    ):
        """save_all_visualizations should extract deal_count from monthly data."""
        output_dir = tmp_path / "figures"

        result = save_all_visualizations(
            sample_cpu_series,
            sample_monthly_data,
            sample_ccf_dataframe,
            output_dir
        )

        # Timeseries should be created using deal_count
        assert isinstance(result['timeseries'], Path)


# =============================================================================
# Tests with mocked matplotlib (for speed)
# =============================================================================


class TestWithMockedMatplotlib:
    """Tests using mocked matplotlib for faster execution."""

    @patch('cpu_index.analysis.vc_visualizations.plt')
    def test_plot_cpu_vc_timeseries_calls_savefig(
        self, mock_plt, sample_cpu_series, sample_vc_series, tmp_path
    ):
        """plot_cpu_vc_timeseries should call savefig when output_path given."""
        # Set up mock
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax2 = MagicMock()
        mock_ax.twinx.return_value = mock_ax2
        mock_ax.xaxis = MagicMock()
        mock_ax.spines = {'top': MagicMock()}
        mock_ax2.spines = {'top': MagicMock()}
        mock_ax.get_legend_handles_labels.return_value = ([], [])
        mock_ax2.get_legend_handles_labels.return_value = ([], [])
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        output_file = tmp_path / "test.png"

        plot_cpu_vc_timeseries(
            sample_cpu_series, sample_vc_series, output_path=output_file
        )

        mock_fig.savefig.assert_called_once()
        mock_plt.close.assert_called_with(mock_fig)

    @patch('cpu_index.analysis.vc_visualizations.plt')
    def test_plot_cross_correlation_calls_subplots(
        self, mock_plt, sample_ccf_dataframe
    ):
        """plot_cross_correlation should call plt.subplots."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.spines = {'top': MagicMock(), 'right': MagicMock()}
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_cross_correlation(sample_ccf_dataframe)

        mock_plt.subplots.assert_called_once()

    @patch('cpu_index.analysis.vc_visualizations.plt')
    def test_plot_rolling_correlation_calls_tight_layout(
        self, mock_plt, sample_cpu_series, sample_vc_series
    ):
        """plot_rolling_correlation should call tight_layout."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.xaxis = MagicMock()
        mock_ax.spines = {'top': MagicMock(), 'right': MagicMock()}
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_rolling_correlation(sample_cpu_series, sample_vc_series)

        mock_plt.tight_layout.assert_called_once()


# =============================================================================
# Edge case and integration tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for visualization functions."""

    def test_timeseries_with_minimal_data(self):
        """plot_cpu_vc_timeseries should work with minimal (3 point) data."""
        dates = pd.date_range("2020-01", periods=3, freq="MS")
        cpu = pd.Series([100, 105, 110], index=dates)
        vc = pd.Series([50, 55, 60], index=dates)

        fig = plot_cpu_vc_timeseries(cpu, vc)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_timeseries_with_two_points_no_trend(self):
        """plot_cpu_vc_timeseries with 2 points should work without trend."""
        dates = pd.date_range("2020-01", periods=2, freq="MS")
        cpu = pd.Series([100, 105], index=dates)
        vc = pd.Series([50, 55], index=dates)

        fig = plot_cpu_vc_timeseries(cpu, vc, show_trend=False)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ccf_with_single_lag(self):
        """plot_cross_correlation should work with single lag."""
        df = pd.DataFrame({
            'lag': [0],
            'correlation': [0.5],
            'n_observations': [50]
        })

        fig = plot_cross_correlation(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ccf_with_high_correlation_values(self):
        """plot_cross_correlation should handle correlations near +/-1."""
        df = pd.DataFrame({
            'lag': [-1, 0, 1],
            'correlation': [-0.95, 0.99, -0.98],
            'n_observations': [50, 50, 50]
        })

        fig = plot_cross_correlation(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_rolling_with_window_equals_data_length(
        self, sample_cpu_series, sample_vc_series
    ):
        """plot_rolling_correlation should work when window equals data length."""
        # 24 data points, window=24
        fig = plot_rolling_correlation(
            sample_cpu_series, sample_vc_series, window=24
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_stage_distribution_with_zeros(self):
        """plot_stage_distribution should handle zero counts."""
        dates = pd.date_range("2020-01", periods=6, freq="MS")
        df = pd.DataFrame({
            'seed_count': [0, 0, 0, 0, 0, 0],
            'early_count': [10, 10, 10, 10, 10, 10],
            'late_count': [5, 5, 5, 5, 5, 5],
        }, index=dates)

        fig = plot_stage_distribution(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_visualizations_file_sizes(
        self,
        sample_cpu_series,
        sample_monthly_data,
        sample_ccf_dataframe,
        tmp_path
    ):
        """All saved visualizations should have reasonable file sizes."""
        output_dir = tmp_path / "figures"

        save_all_visualizations(
            sample_cpu_series,
            sample_monthly_data,
            sample_ccf_dataframe,
            output_dir
        )

        # Check all files have content (reasonable minimum size)
        min_size = 1000  # At least 1KB for a valid PNG
        assert (output_dir / "cpu_vc_timeseries.png").stat().st_size > min_size
        assert (output_dir / "cpu_vc_ccf.png").stat().st_size > min_size
        assert (output_dir / "cpu_vc_rolling_corr.png").stat().st_size > min_size
        assert (output_dir / "vc_stage_distribution.png").stat().st_size > min_size
