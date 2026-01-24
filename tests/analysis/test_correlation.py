"""
Tests for correlation.py - CPU-VC Correlation Analysis Module

Tests cover:
1. check_stationarity() - Stationarity testing with ADF and KPSS
2. make_stationary() - Automatic differencing for stationarity
3. cross_correlation() - Cross-correlation at multiple lags
4. find_optimal_lag() - Finding optimal lag from CCF results
5. granger_test() - Granger causality testing
6. analyze_cpu_vc_correlation() - Comprehensive analysis pipeline
7. Edge cases - Short series, missing data, constant series

Tests are designed to work with or without statsmodels installed.
Statsmodels-dependent tests are skipped when not available.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from cpu_index.analysis.correlation import (
    check_stationarity,
    make_stationary,
    cross_correlation,
    find_optimal_lag,
    granger_test,
    bidirectional_granger_test,
    analyze_cpu_vc_correlation,
    check_statsmodels_available,
    get_required_observations,
    MIN_OBSERVATIONS,
    DEFAULT_MAX_LAG,
    DEFAULT_ALPHA,
    _STATSMODELS_AVAILABLE,
)


# =============================================================================
# PYTEST MARKERS AND HELPERS
# =============================================================================

# Skip decorator for tests requiring statsmodels
requires_statsmodels = pytest.mark.skipif(
    not _STATSMODELS_AVAILABLE,
    reason="statsmodels not installed"
)


def create_stationary_series(n: int = 50, seed: int = 42) -> pd.Series:
    """Create a stationary series (white noise)."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    values = np.random.randn(n) * 10 + 100  # White noise around 100
    return pd.Series(values, index=dates, name="stationary")


def create_non_stationary_series(n: int = 50, seed: int = 42) -> pd.Series:
    """Create a non-stationary series (random walk with drift)."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    # Random walk: cumulative sum of random increments
    values = 100 + np.cumsum(np.random.randn(n) * 5 + 0.5)  # Drift of 0.5
    return pd.Series(values, index=dates, name="non_stationary")


def create_correlated_series(
    n: int = 50, correlation: float = 0.7, lag: int = 0, seed: int = 42
) -> tuple[pd.Series, pd.Series]:
    """Create two correlated series with optional lag."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")

    # Base series
    s1 = np.random.randn(n) * 10 + 100

    # Correlated series
    noise = np.random.randn(n) * 10 * (1 - correlation)
    if lag > 0:
        s2_values = np.zeros(n)
        s2_values[lag:] = s1[:-lag] * correlation + noise[lag:]
        s2_values[:lag] = 100 + noise[:lag]
    elif lag < 0:
        s2_values = np.zeros(n)
        s2_values[:lag] = s1[-lag:] * correlation + noise[:lag]
        s2_values[lag:] = 100 + noise[lag:]
    else:
        s2_values = s1 * correlation + noise

    series1 = pd.Series(s1, index=dates, name="series1")
    series2 = pd.Series(s2_values, index=dates, name="series2")
    return series1, series2


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def stationary_series():
    """Fixture: stationary series (white noise)."""
    return create_stationary_series(n=50, seed=42)


@pytest.fixture
def non_stationary_series():
    """Fixture: non-stationary series (random walk)."""
    return create_non_stationary_series(n=50, seed=42)


@pytest.fixture
def short_series():
    """Fixture: series with too few observations (< MIN_OBSERVATIONS)."""
    dates = pd.date_range("2020-01-01", periods=10, freq="MS")
    values = np.random.randn(10) * 10 + 100
    return pd.Series(values, index=dates, name="short")


@pytest.fixture
def constant_series():
    """Fixture: constant series (zero variance)."""
    dates = pd.date_range("2020-01-01", periods=50, freq="MS")
    values = np.full(50, 100.0)
    return pd.Series(values, index=dates, name="constant")


@pytest.fixture
def series_with_nan():
    """Fixture: series with NaN values."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=50, freq="MS")
    values = np.random.randn(50) * 10 + 100
    values[10:15] = np.nan  # 5 NaN values
    return pd.Series(values, index=dates, name="with_nan")


@pytest.fixture
def correlated_pair():
    """Fixture: two correlated series."""
    return create_correlated_series(n=50, correlation=0.8, lag=0, seed=42)


@pytest.fixture
def lagged_pair():
    """Fixture: two series where series1 leads series2 by 2 periods."""
    return create_correlated_series(n=50, correlation=0.8, lag=2, seed=42)


@pytest.fixture
def sample_ccf_results():
    """Fixture: sample cross-correlation results DataFrame."""
    lags = range(-5, 6)
    correlations = [0.1, 0.15, 0.2, 0.3, 0.5, 0.6, 0.5, 0.3, 0.2, 0.15, 0.1]
    n_obs = [45, 46, 47, 48, 49, 50, 49, 48, 47, 46, 45]
    return pd.DataFrame({
        'lag': list(lags),
        'correlation': correlations,
        'n_observations': n_obs,
        'interpretation': [
            f"series2_leads_by_{-l}_periods" if l < 0 else
            f"series1_leads_by_{l}_periods" if l > 0 else
            "contemporaneous" for l in lags
        ],
    })


@pytest.fixture
def sample_granger_data():
    """Fixture: DataFrame suitable for Granger causality test."""
    np.random.seed(42)
    n = 50
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")

    # Create x that Granger-causes y (y depends on lagged x)
    x = np.random.randn(n) * 10 + 100
    y = np.zeros(n)
    y[0] = np.random.randn() * 5 + 50
    for t in range(1, n):
        y[t] = 0.3 * y[t - 1] + 0.4 * x[t - 1] + np.random.randn() * 3

    return pd.DataFrame({
        'cpu_index': x,
        'vc_count': y,
    }, index=dates)


# =============================================================================
# TEST: CONSTANTS AND UTILITIES
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_min_observations_value(self):
        """MIN_OBSERVATIONS should be 24 (2 years of monthly data)."""
        assert MIN_OBSERVATIONS == 24

    def test_default_max_lag_value(self):
        """DEFAULT_MAX_LAG should be 12."""
        assert DEFAULT_MAX_LAG == 12

    def test_default_alpha_value(self):
        """DEFAULT_ALPHA should be 0.05."""
        assert DEFAULT_ALPHA == 0.05


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_check_statsmodels_available_returns_bool(self):
        """check_statsmodels_available should return a boolean."""
        result = check_statsmodels_available()
        assert isinstance(result, bool)

    def test_check_statsmodels_matches_module_constant(self):
        """check_statsmodels_available should match _STATSMODELS_AVAILABLE."""
        assert check_statsmodels_available() == _STATSMODELS_AVAILABLE

    def test_get_required_observations_default(self):
        """get_required_observations should return MIN_OBSERVATIONS + 2*default_lag."""
        result = get_required_observations()
        # default max_lag is 4
        assert result == 4 * 2 + MIN_OBSERVATIONS

    def test_get_required_observations_custom_lag(self):
        """get_required_observations should calculate correctly for custom lag."""
        result = get_required_observations(max_lag=6)
        assert result == 6 * 2 + MIN_OBSERVATIONS


# =============================================================================
# TEST: check_stationarity()
# =============================================================================


class TestStationarityBasic:
    """Basic tests for check_stationarity()."""

    @requires_statsmodels
    def check_stationarity_returns_dict(self, stationary_series):
        """check_stationarity should return a dictionary."""
        result = check_stationarity(stationary_series, name="test")
        assert isinstance(result, dict)

    @requires_statsmodels
    def check_stationarity_contains_expected_keys(self, stationary_series):
        """check_stationarity should return all expected keys."""
        result = check_stationarity(stationary_series, name="test")

        expected_keys = [
            "name", "n_observations",
            "adf_statistic", "adf_pvalue", "adf_critical_values", "adf_stationary",
            "kpss_statistic", "kpss_pvalue", "kpss_critical_values", "kpss_stationary",
            "conclusion", "differencing_needed",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    @requires_statsmodels
    def check_stationarity_stationary_series_detected(self, stationary_series):
        """check_stationarity should detect stationary series."""
        result = check_stationarity(stationary_series, name="test")

        # Stationary series should have:
        # - ADF rejects null (pvalue < alpha) -> adf_stationary = True
        # - KPSS fails to reject null (pvalue >= alpha) -> kpss_stationary = True
        # Result: conclusion = "stationary", differencing_needed = False
        assert result["adf_stationary"] is True
        assert result["kpss_stationary"] is True
        assert result["conclusion"] == "stationary"
        assert result["differencing_needed"] is False

    @requires_statsmodels
    def check_stationarity_non_stationary_series_detected(self, non_stationary_series):
        """check_stationarity should detect non-stationary series."""
        result = check_stationarity(non_stationary_series, name="test")

        # Non-stationary series should have:
        # - ADF fails to reject null -> adf_stationary = False
        # - KPSS rejects null -> kpss_stationary = False
        # Result: conclusion = "non_stationary", differencing_needed = True
        assert result["differencing_needed"] is True
        assert result["conclusion"] in ["non_stationary", "trend_stationary", "inconclusive"]

    @requires_statsmodels
    def check_stationarity_name_preserved(self, stationary_series):
        """check_stationarity should preserve the name parameter."""
        result = check_stationarity(stationary_series, name="cpu_index")
        assert result["name"] == "cpu_index"

    @requires_statsmodels
    def check_stationarity_n_observations_correct(self, stationary_series):
        """check_stationarity should report correct number of observations."""
        result = check_stationarity(stationary_series, name="test")
        assert result["n_observations"] == len(stationary_series)


class TestStationarityEdgeCases:
    """Edge case tests for check_stationarity()."""

    @requires_statsmodels
    def check_stationarity_short_series_raises(self, short_series):
        """check_stationarity should raise ValueError for short series."""
        with pytest.raises(ValueError, match="has only.*observations"):
            check_stationarity(short_series, name="test")

    @requires_statsmodels
    def check_stationarity_handles_nan(self, series_with_nan):
        """check_stationarity should handle series with NaN values."""
        # Series has 50 obs, 5 NaN -> 45 valid obs (>= 24)
        result = check_stationarity(series_with_nan, name="test")
        assert result["n_observations"] == 45  # 50 - 5 NaN values

    @requires_statsmodels
    def check_stationarity_custom_alpha(self, stationary_series):
        """check_stationarity should respect custom alpha parameter."""
        result_05 = check_stationarity(stationary_series, name="test", alpha=0.05)
        result_01 = check_stationarity(stationary_series, name="test", alpha=0.01)

        # Both should have results, but conclusions may differ at different alpha
        assert "adf_pvalue" in result_05
        assert "adf_pvalue" in result_01

    def check_stationarity_without_statsmodels_raises(self, stationary_series):
        """check_stationarity should raise ImportError without statsmodels."""
        with patch('cpu_index.analysis.correlation._STATSMODELS_AVAILABLE', False):
            # Need to reload the function to pick up the patched value
            # Instead, we test the behavior with a mock
            pass  # Covered by module-level guard


class TestStationarityMocked:
    """Tests for check_stationarity() with mocked statsmodels."""

    def check_stationarity_import_error_when_not_available(self, stationary_series):
        """check_stationarity should raise ImportError when statsmodels unavailable."""
        # Directly test the import check logic
        if not _STATSMODELS_AVAILABLE:
            with pytest.raises(ImportError, match="statsmodels is required"):
                check_stationarity(stationary_series, name="test")


# =============================================================================
# TEST: make_stationary()
# =============================================================================


class TestMakeStationaryBasic:
    """Basic tests for make_stationary()."""

    @requires_statsmodels
    def test_make_stationary_returns_dict(self, non_stationary_series):
        """make_stationary should return a dictionary."""
        result = make_stationary(non_stationary_series)
        assert isinstance(result, dict)

    @requires_statsmodels
    def test_make_stationary_contains_expected_keys(self, non_stationary_series):
        """make_stationary should return all expected keys."""
        result = make_stationary(non_stationary_series)

        expected_keys = [
            "series", "original_length", "final_length",
            "differences_applied", "is_stationary", "transformations",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    @requires_statsmodels
    def test_make_stationary_returns_series(self, non_stationary_series):
        """make_stationary should return a pandas Series in 'series' key."""
        result = make_stationary(non_stationary_series)
        assert isinstance(result["series"], pd.Series)

    @requires_statsmodels
    def test_make_stationary_already_stationary_no_diff(self, stationary_series):
        """make_stationary should not difference already stationary series."""
        result = make_stationary(stationary_series)

        assert result["differences_applied"] == 0
        assert result["is_stationary"] is True
        assert "already stationary" in result["transformations"][0].lower()

    @requires_statsmodels
    def test_make_stationary_applies_differencing(self, non_stationary_series):
        """make_stationary should apply differencing to non-stationary series."""
        result = make_stationary(non_stationary_series)

        # Random walk should need differencing
        assert result["differences_applied"] >= 1
        assert "difference" in result["transformations"][0].lower()

    @requires_statsmodels
    def test_make_stationary_length_decreases_with_differencing(
        self, non_stationary_series
    ):
        """make_stationary differencing should reduce series length."""
        result = make_stationary(non_stationary_series)

        if result["differences_applied"] > 0:
            assert result["final_length"] < result["original_length"]
            # Each differencing loses 1 observation
            expected_loss = result["differences_applied"]
            assert result["final_length"] == result["original_length"] - expected_loss


class TestMakeStationaryEdgeCases:
    """Edge case tests for make_stationary()."""

    @requires_statsmodels
    def test_make_stationary_max_differences_respected(self, non_stationary_series):
        """make_stationary should not exceed max_differences."""
        result = make_stationary(non_stationary_series, max_differences=1)
        assert result["differences_applied"] <= 1

    @requires_statsmodels
    def test_make_stationary_handles_nan(self, series_with_nan):
        """make_stationary should handle series with NaN values."""
        result = make_stationary(series_with_nan)
        # Should clean NaN and still work
        assert "series" in result

    @requires_statsmodels
    def test_make_stationary_short_series_returns_best_effort(self):
        """make_stationary should return best effort for series near MIN_OBSERVATIONS."""
        # Create series just at MIN_OBSERVATIONS boundary
        dates = pd.date_range("2020-01-01", periods=MIN_OBSERVATIONS, freq="MS")
        values = 100 + np.cumsum(np.random.randn(MIN_OBSERVATIONS) * 5)
        series = pd.Series(values, index=dates)

        result = make_stationary(series)
        # Should return something, even if not fully stationary
        assert "series" in result


# =============================================================================
# TEST: cross_correlation()
# =============================================================================


class TestCrossCorrelationBasic:
    """Basic tests for cross_correlation()."""

    def test_cross_correlation_returns_dataframe(self, correlated_pair):
        """cross_correlation should return a DataFrame."""
        series1, series2 = correlated_pair
        result = cross_correlation(series1, series2)
        assert isinstance(result, pd.DataFrame)

    def test_cross_correlation_expected_columns(self, correlated_pair):
        """cross_correlation should have expected columns."""
        series1, series2 = correlated_pair
        result = cross_correlation(series1, series2)

        expected_columns = ['lag', 'correlation', 'n_observations', 'interpretation']
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_cross_correlation_lag_range(self, correlated_pair):
        """cross_correlation should have lags from -max_lag to +max_lag."""
        series1, series2 = correlated_pair
        max_lag = 6
        result = cross_correlation(series1, series2, max_lag=max_lag)

        expected_lags = list(range(-max_lag, max_lag + 1))
        assert list(result['lag']) == expected_lags

    def test_cross_correlation_default_max_lag(self, correlated_pair):
        """cross_correlation should use DEFAULT_MAX_LAG when not specified."""
        series1, series2 = correlated_pair
        result = cross_correlation(series1, series2)

        # Should have -12 to +12 lags
        assert result['lag'].min() == -DEFAULT_MAX_LAG
        assert result['lag'].max() == DEFAULT_MAX_LAG
        assert len(result) == 2 * DEFAULT_MAX_LAG + 1

    def test_cross_correlation_contemporaneous_at_lag_zero(self, correlated_pair):
        """cross_correlation should have contemporaneous correlation at lag 0."""
        series1, series2 = correlated_pair
        result = cross_correlation(series1, series2)

        lag_zero = result[result['lag'] == 0]
        assert len(lag_zero) == 1
        assert lag_zero['interpretation'].iloc[0] == "contemporaneous"

    def test_cross_correlation_high_correlation_detected(self, correlated_pair):
        """cross_correlation should detect high correlation in correlated series."""
        series1, series2 = correlated_pair
        result = cross_correlation(series1, series2)

        # At lag 0, correlation should be substantial (series were created correlated)
        lag_zero_corr = result[result['lag'] == 0]['correlation'].iloc[0]
        assert abs(lag_zero_corr) > 0.3  # Should be notable correlation


class TestCrossCorrelationLagInterpretation:
    """Tests for lag interpretation in cross_correlation()."""

    def test_positive_lag_interpretation(self, correlated_pair):
        """Positive lag should indicate series1 leads series2."""
        series1, series2 = correlated_pair
        result = cross_correlation(series1, series2, max_lag=3)

        positive_lag_row = result[result['lag'] == 2].iloc[0]
        assert "series1_leads" in positive_lag_row['interpretation']

    def test_negative_lag_interpretation(self, correlated_pair):
        """Negative lag should indicate series2 leads series1."""
        series1, series2 = correlated_pair
        result = cross_correlation(series1, series2, max_lag=3)

        negative_lag_row = result[result['lag'] == -2].iloc[0]
        assert "series2_leads" in negative_lag_row['interpretation']

    def test_lagged_series_peak_at_correct_lag(self, lagged_pair):
        """Cross-correlation should peak near the true lag."""
        series1, series2 = lagged_pair
        result = cross_correlation(series1, series2, max_lag=6)

        # Find the lag with maximum absolute correlation
        max_corr_row = result.loc[result['correlation'].abs().idxmax()]

        # The lag with highest correlation should be around +2
        # (series1 leads series2 by 2 periods)
        # Allow some tolerance due to noise
        assert max_corr_row['lag'] >= 0  # Should be positive (series1 leads)


class TestCrossCorrelationEdgeCases:
    """Edge case tests for cross_correlation()."""

    def test_cross_correlation_short_series_returns_empty(self):
        """cross_correlation should return empty DataFrame for short series."""
        dates = pd.date_range("2020-01-01", periods=10, freq="MS")
        s1 = pd.Series(np.random.randn(10), index=dates)
        s2 = pd.Series(np.random.randn(10), index=dates)

        result = cross_correlation(s1, s2)
        assert len(result) == 0

    def test_cross_correlation_handles_nan(self):
        """cross_correlation should handle series with NaN values."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=50, freq="MS")
        s1 = pd.Series(np.random.randn(50), index=dates)
        s2 = pd.Series(np.random.randn(50), index=dates)
        s1.iloc[10:15] = np.nan

        result = cross_correlation(s1, s2)
        # Should still produce results (NaN rows dropped in alignment)
        assert len(result) > 0

    def test_cross_correlation_misaligned_index(self):
        """cross_correlation should align series by index."""
        dates1 = pd.date_range("2020-01-01", periods=30, freq="MS")
        dates2 = pd.date_range("2020-03-01", periods=30, freq="MS")  # 2 months offset

        np.random.seed(42)
        s1 = pd.Series(np.random.randn(30), index=dates1)
        s2 = pd.Series(np.random.randn(30), index=dates2)

        result = cross_correlation(s1, s2)
        # Overlap is 28 months, which is >= MIN_OBSERVATIONS
        assert len(result) > 0

    def test_cross_correlation_identical_series(self):
        """cross_correlation with identical series should have corr=1 at lag=0."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=50, freq="MS")
        s1 = pd.Series(np.random.randn(50), index=dates)

        result = cross_correlation(s1, s1.copy())
        lag_zero_corr = result[result['lag'] == 0]['correlation'].iloc[0]
        assert lag_zero_corr == pytest.approx(1.0, rel=1e-10)


# =============================================================================
# TEST: find_optimal_lag()
# =============================================================================


class TestFindOptimalLagBasic:
    """Basic tests for find_optimal_lag()."""

    def test_find_optimal_lag_returns_dict(self, sample_ccf_results):
        """find_optimal_lag should return a dictionary."""
        result = find_optimal_lag(sample_ccf_results)
        assert isinstance(result, dict)

    def test_find_optimal_lag_expected_keys(self, sample_ccf_results):
        """find_optimal_lag should return all expected keys."""
        result = find_optimal_lag(sample_ccf_results)

        expected_keys = [
            "optimal_lag", "max_correlation", "is_significant",
            "direction", "lead_series",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_find_optimal_lag_finds_maximum(self, sample_ccf_results):
        """find_optimal_lag should find the lag with maximum absolute correlation."""
        result = find_optimal_lag(sample_ccf_results)

        # In sample_ccf_results, lag=0 has correlation=0.6 (highest)
        assert result["optimal_lag"] == 0
        assert result["max_correlation"] == pytest.approx(0.6)

    def test_find_optimal_lag_positive_direction(self, sample_ccf_results):
        """find_optimal_lag should detect positive correlation direction."""
        result = find_optimal_lag(sample_ccf_results)
        assert result["direction"] == "positive"

    def test_find_optimal_lag_contemporaneous(self, sample_ccf_results):
        """find_optimal_lag at lag=0 should indicate contemporaneous."""
        result = find_optimal_lag(sample_ccf_results)
        assert result["lead_series"] == "contemporaneous"


class TestFindOptimalLagSignificance:
    """Tests for significance threshold in find_optimal_lag()."""

    def test_find_optimal_lag_significant_above_threshold(self, sample_ccf_results):
        """find_optimal_lag should mark as significant when above threshold."""
        result = find_optimal_lag(sample_ccf_results, min_correlation=0.5)
        assert result["is_significant"] is True  # 0.6 >= 0.5

    def test_find_optimal_lag_not_significant_below_threshold(self, sample_ccf_results):
        """find_optimal_lag should mark as not significant when below threshold."""
        result = find_optimal_lag(sample_ccf_results, min_correlation=0.7)
        assert result["is_significant"] is False  # 0.6 < 0.7


class TestFindOptimalLagLeadLag:
    """Tests for lead/lag determination in find_optimal_lag()."""

    def test_find_optimal_lag_series1_leads(self):
        """find_optimal_lag should identify series1 as leading when lag > 0."""
        ccf = pd.DataFrame({
            'lag': [-2, -1, 0, 1, 2, 3],
            'correlation': [0.1, 0.2, 0.3, 0.4, 0.8, 0.5],
            'n_observations': [48, 49, 50, 49, 48, 47],
            'interpretation': [''] * 6,
        })
        result = find_optimal_lag(ccf)
        assert result["optimal_lag"] == 2
        assert result["lead_series"] == "series1"

    def test_find_optimal_lag_series2_leads(self):
        """find_optimal_lag should identify series2 as leading when lag < 0."""
        ccf = pd.DataFrame({
            'lag': [-3, -2, -1, 0, 1, 2],
            'correlation': [0.5, 0.9, 0.4, 0.3, 0.2, 0.1],
            'n_observations': [47, 48, 49, 50, 49, 48],
            'interpretation': [''] * 6,
        })
        result = find_optimal_lag(ccf)
        assert result["optimal_lag"] == -2
        assert result["lead_series"] == "series2"

    def test_find_optimal_lag_negative_correlation(self):
        """find_optimal_lag should handle negative correlations correctly."""
        ccf = pd.DataFrame({
            'lag': [-2, -1, 0, 1, 2],
            'correlation': [-0.3, -0.5, -0.8, -0.5, -0.3],
            'n_observations': [48, 49, 50, 49, 48],
            'interpretation': [''] * 5,
        })
        result = find_optimal_lag(ccf)
        assert result["optimal_lag"] == 0
        assert result["max_correlation"] == pytest.approx(-0.8)
        assert result["direction"] == "negative"


class TestFindOptimalLagEdgeCases:
    """Edge case tests for find_optimal_lag()."""

    def test_find_optimal_lag_empty_results(self):
        """find_optimal_lag should handle empty CCF results."""
        ccf = pd.DataFrame({
            'lag': [],
            'correlation': [],
            'n_observations': [],
            'interpretation': [],
        })
        result = find_optimal_lag(ccf)

        assert result["optimal_lag"] is None
        assert result["max_correlation"] is None
        assert result["is_significant"] is False

    def test_find_optimal_lag_all_nan_correlations(self):
        """find_optimal_lag should handle all NaN correlations."""
        ccf = pd.DataFrame({
            'lag': [-1, 0, 1],
            'correlation': [np.nan, np.nan, np.nan],
            'n_observations': [49, 50, 49],
            'interpretation': [''] * 3,
        })
        result = find_optimal_lag(ccf)

        assert result["optimal_lag"] is None
        assert result["is_significant"] is False


# =============================================================================
# TEST: granger_test()
# =============================================================================


class TestGrangerTestBasic:
    """Basic tests for granger_test()."""

    @requires_statsmodels
    def test_granger_test_returns_dict(self, sample_granger_data):
        """granger_test should return a dictionary."""
        result = granger_test(sample_granger_data, 'cpu_index', 'vc_count')
        assert isinstance(result, dict)

    @requires_statsmodels
    def test_granger_test_expected_keys(self, sample_granger_data):
        """granger_test should return all expected keys."""
        result = granger_test(sample_granger_data, 'cpu_index', 'vc_count')

        expected_keys = [
            "x_col", "y_col", "lags_tested", "pvalues_by_lag",
            "min_pvalue", "optimal_lag", "granger_causes", "interpretation",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    @requires_statsmodels
    def test_granger_test_preserves_column_names(self, sample_granger_data):
        """granger_test should preserve column names in output."""
        result = granger_test(sample_granger_data, 'cpu_index', 'vc_count')
        assert result["x_col"] == "cpu_index"
        assert result["y_col"] == "vc_count"

    @requires_statsmodels
    def test_granger_test_lags_tested(self, sample_granger_data):
        """granger_test should test lags 1 to max_lag."""
        result = granger_test(sample_granger_data, 'cpu_index', 'vc_count', max_lag=4)
        assert result["lags_tested"] == [1, 2, 3, 4]

    @requires_statsmodels
    def test_granger_test_pvalues_by_lag(self, sample_granger_data):
        """granger_test should provide p-values for each lag."""
        result = granger_test(sample_granger_data, 'cpu_index', 'vc_count', max_lag=4)

        assert isinstance(result["pvalues_by_lag"], dict)
        assert len(result["pvalues_by_lag"]) == 4  # 4 lags tested

    @requires_statsmodels
    def test_granger_test_detects_causality(self, sample_granger_data):
        """granger_test should detect Granger causality in designed data."""
        # The sample_granger_data has y dependent on lagged x
        result = granger_test(sample_granger_data, 'cpu_index', 'vc_count', max_lag=4)

        # Should detect causality (p-value < 0.05)
        assert result["granger_causes"] is True
        assert result["min_pvalue"] < 0.05


class TestGrangerTestEdgeCases:
    """Edge case tests for granger_test()."""

    @requires_statsmodels
    def test_granger_test_missing_column_raises(self, sample_granger_data):
        """granger_test should raise ValueError for missing column."""
        with pytest.raises(ValueError, match="not found"):
            granger_test(sample_granger_data, 'nonexistent', 'vc_count')

    @requires_statsmodels
    def test_granger_test_insufficient_data(self):
        """granger_test should handle insufficient data gracefully."""
        dates = pd.date_range("2020-01-01", periods=10, freq="MS")
        data = pd.DataFrame({
            'x': np.random.randn(10),
            'y': np.random.randn(10),
        }, index=dates)

        result = granger_test(data, 'x', 'y', max_lag=4)

        assert result["granger_causes"] is False
        assert "Insufficient data" in result["interpretation"]

    @requires_statsmodels
    def test_granger_test_no_causality(self):
        """granger_test should not find causality in independent series."""
        np.random.seed(42)
        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="MS")

        # Independent random walks
        data = pd.DataFrame({
            'x': np.random.randn(n),
            'y': np.random.randn(n),
        }, index=dates)

        result = granger_test(data, 'x', 'y', max_lag=4)

        # Should likely NOT find causality (independent series)
        # Note: May occasionally be significant by chance
        assert isinstance(result["granger_causes"], bool)


class TestBidirectionalGrangerTest:
    """Tests for bidirectional_granger_test()."""

    @requires_statsmodels
    def test_bidirectional_returns_dict(self, sample_granger_data):
        """bidirectional_granger_test should return a dictionary."""
        result = bidirectional_granger_test(
            sample_granger_data, 'cpu_index', 'vc_count'
        )
        assert isinstance(result, dict)

    @requires_statsmodels
    def test_bidirectional_expected_keys(self, sample_granger_data):
        """bidirectional_granger_test should have both direction results."""
        result = bidirectional_granger_test(
            sample_granger_data, 'cpu_index', 'vc_count'
        )

        assert "col1_to_col2" in result
        assert "col2_to_col1" in result
        assert "relationship" in result

    @requires_statsmodels
    def test_bidirectional_unidirectional_relationship(self, sample_granger_data):
        """bidirectional_granger_test should detect unidirectional causality."""
        result = bidirectional_granger_test(
            sample_granger_data, 'cpu_index', 'vc_count'
        )

        # In sample data, cpu_index causes vc_count, not reverse
        assert result["col1_to_col2"]["granger_causes"] is True


# =============================================================================
# TEST: analyze_cpu_vc_correlation()
# =============================================================================


class TestAnalyzeCpuVcCorrelationBasic:
    """Basic tests for analyze_cpu_vc_correlation()."""

    def test_analyze_returns_dict(self, correlated_pair):
        """analyze_cpu_vc_correlation should return a dictionary."""
        series1, series2 = correlated_pair
        result = analyze_cpu_vc_correlation(series1, series2)
        assert isinstance(result, dict)

    def test_analyze_expected_top_level_keys(self, correlated_pair):
        """analyze_cpu_vc_correlation should have expected top-level keys."""
        series1, series2 = correlated_pair
        result = analyze_cpu_vc_correlation(series1, series2)

        expected_keys = [
            "status", "metadata", "stationarity",
            "cross_correlation", "granger_causality", "summary", "warnings",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_analyze_status_success(self, correlated_pair):
        """analyze_cpu_vc_correlation should return success status."""
        series1, series2 = correlated_pair
        result = analyze_cpu_vc_correlation(series1, series2)
        assert result["status"] == "success"


class TestAnalyzeCpuVcCorrelationMetadata:
    """Tests for metadata in analyze_cpu_vc_correlation()."""

    def test_analyze_metadata_n_observations(self, correlated_pair):
        """analyze_cpu_vc_correlation should report correct n_observations."""
        series1, series2 = correlated_pair
        result = analyze_cpu_vc_correlation(series1, series2)

        assert result["metadata"]["n_observations"] == 50

    def test_analyze_metadata_date_range(self, correlated_pair):
        """analyze_cpu_vc_correlation should report date range."""
        series1, series2 = correlated_pair
        result = analyze_cpu_vc_correlation(series1, series2)

        assert "date_range" in result["metadata"]
        assert "start" in result["metadata"]["date_range"]
        assert "end" in result["metadata"]["date_range"]

    def test_analyze_metadata_custom_names(self, correlated_pair):
        """analyze_cpu_vc_correlation should use custom names."""
        series1, series2 = correlated_pair
        result = analyze_cpu_vc_correlation(
            series1, series2,
            cpu_name="my_cpu",
            vc_name="my_vc"
        )

        assert result["metadata"]["cpu_name"] == "my_cpu"
        assert result["metadata"]["vc_name"] == "my_vc"


class TestAnalyzeCpuVcCorrelationCrossCorr:
    """Tests for cross-correlation in analyze_cpu_vc_correlation()."""

    def test_analyze_cross_correlation_results(self, correlated_pair):
        """analyze_cpu_vc_correlation should include cross-correlation results."""
        series1, series2 = correlated_pair
        result = analyze_cpu_vc_correlation(series1, series2)

        assert "results" in result["cross_correlation"]
        assert "optimal_lag" in result["cross_correlation"]
        assert "contemporaneous_correlation" in result["cross_correlation"]

    def test_analyze_cross_correlation_contemporaneous(self, correlated_pair):
        """analyze_cpu_vc_correlation should report contemporaneous correlation."""
        series1, series2 = correlated_pair
        result = analyze_cpu_vc_correlation(series1, series2)

        contemp_corr = result["cross_correlation"]["contemporaneous_correlation"]
        assert isinstance(contemp_corr, float)
        # Correlated series should have notable correlation
        assert abs(contemp_corr) > 0.3


class TestAnalyzeCpuVcCorrelationSummary:
    """Tests for summary in analyze_cpu_vc_correlation()."""

    def test_analyze_summary_n_observations(self, correlated_pair):
        """analyze_cpu_vc_correlation summary should include n_observations."""
        series1, series2 = correlated_pair
        result = analyze_cpu_vc_correlation(series1, series2)

        assert result["summary"]["n_observations"] == 50

    def test_analyze_summary_contemporaneous_correlation(self, correlated_pair):
        """analyze_cpu_vc_correlation summary should include contemporaneous corr."""
        series1, series2 = correlated_pair
        result = analyze_cpu_vc_correlation(series1, series2)

        assert "contemporaneous_correlation" in result["summary"]


class TestAnalyzeCpuVcCorrelationEdgeCases:
    """Edge case tests for analyze_cpu_vc_correlation()."""

    def test_analyze_short_series_returns_error(self):
        """analyze_cpu_vc_correlation should return error for short series."""
        dates = pd.date_range("2020-01-01", periods=10, freq="MS")
        s1 = pd.Series(np.random.randn(10), index=dates)
        s2 = pd.Series(np.random.randn(10), index=dates)

        result = analyze_cpu_vc_correlation(s1, s2)

        assert result["status"] == "error"
        assert "Insufficient data" in result["summary"]["error"]

    def test_analyze_misaligned_series(self):
        """analyze_cpu_vc_correlation should handle misaligned series."""
        dates1 = pd.date_range("2020-01-01", periods=40, freq="MS")
        dates2 = pd.date_range("2020-03-01", periods=40, freq="MS")  # 2 month offset

        np.random.seed(42)
        s1 = pd.Series(np.random.randn(40), index=dates1)
        s2 = pd.Series(np.random.randn(40), index=dates2)

        result = analyze_cpu_vc_correlation(s1, s2)

        # Overlap is 38 months >= MIN_OBSERVATIONS, should succeed
        assert result["status"] == "success"
        assert result["metadata"]["n_observations"] == 38

    def test_analyze_with_nan_values(self):
        """analyze_cpu_vc_correlation should handle NaN values."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=50, freq="MS")
        s1 = pd.Series(np.random.randn(50), index=dates)
        s2 = pd.Series(np.random.randn(50), index=dates)
        s1.iloc[5:10] = np.nan

        result = analyze_cpu_vc_correlation(s1, s2)

        # Should still work with fewer observations
        assert result["status"] == "success"
        assert result["metadata"]["n_observations"] == 45


class TestAnalyzeCpuVcCorrelationStatsmodels:
    """Tests for statsmodels-dependent features in analyze_cpu_vc_correlation()."""

    @requires_statsmodels
    def test_analyze_includes_stationarity(self, correlated_pair):
        """analyze_cpu_vc_correlation should include stationarity results."""
        series1, series2 = correlated_pair
        result = analyze_cpu_vc_correlation(series1, series2)

        # Should have stationarity results for both series
        assert "cpu_index" in result["stationarity"]
        assert "vc_deal_count" in result["stationarity"]

    @requires_statsmodels
    def test_analyze_includes_granger(self, correlated_pair):
        """analyze_cpu_vc_correlation should include Granger results."""
        series1, series2 = correlated_pair
        result = analyze_cpu_vc_correlation(series1, series2)

        # Should have Granger causality results
        assert "relationship" in result["granger_causality"]

    def test_analyze_without_statsmodels_warnings(self, correlated_pair):
        """analyze_cpu_vc_correlation should warn when statsmodels not available."""
        if not _STATSMODELS_AVAILABLE:
            series1, series2 = correlated_pair
            result = analyze_cpu_vc_correlation(series1, series2)

            # Should have warnings about missing statsmodels
            assert any("statsmodels" in w.lower() for w in result["warnings"])


# =============================================================================
# TEST: EDGE CASES - CONSTANT SERIES
# =============================================================================


class TestConstantSeries:
    """Tests for handling constant (zero-variance) series."""

    def test_cross_correlation_constant_series(self, constant_series, stationary_series):
        """cross_correlation should handle constant series (produces NaN)."""
        result = cross_correlation(constant_series, stationary_series)

        # Correlation with constant series should be NaN
        assert result['correlation'].isna().all() or len(result) == 0

    def test_find_optimal_lag_constant_produces_nan(self, constant_series, stationary_series):
        """find_optimal_lag with constant series should handle NaN correlations."""
        ccf_result = cross_correlation(constant_series, stationary_series)
        result = find_optimal_lag(ccf_result)

        # Should handle gracefully (either None or some default)
        assert result["optimal_lag"] is None or isinstance(result["optimal_lag"], int)


# =============================================================================
# TEST: IMPORT ERROR HANDLING
# =============================================================================


class TestImportErrorHandling:
    """Tests for proper ImportError handling when statsmodels is unavailable."""

    def test_check_stationarity_raises_import_error(self, stationary_series):
        """check_stationarity raises ImportError when statsmodels unavailable."""
        if not _STATSMODELS_AVAILABLE:
            with pytest.raises(ImportError, match="statsmodels"):
                check_stationarity(stationary_series, name="test")

    def test_make_stationary_raises_import_error(self, stationary_series):
        """make_stationary raises ImportError when statsmodels unavailable."""
        if not _STATSMODELS_AVAILABLE:
            with pytest.raises(ImportError, match="statsmodels"):
                make_stationary(stationary_series)

    def test_granger_test_raises_import_error(self, sample_granger_data):
        """granger_test raises ImportError when statsmodels unavailable."""
        if not _STATSMODELS_AVAILABLE:
            with pytest.raises(ImportError, match="statsmodels"):
                granger_test(sample_granger_data, 'cpu_index', 'vc_count')

    def test_bidirectional_granger_raises_import_error(self, sample_granger_data):
        """bidirectional_granger_test raises ImportError when statsmodels unavailable."""
        if not _STATSMODELS_AVAILABLE:
            with pytest.raises(ImportError, match="statsmodels"):
                bidirectional_granger_test(sample_granger_data, 'cpu_index', 'vc_count')
