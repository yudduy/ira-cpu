"""
Tests for vc_aggregator.py - VC deal aggregation for time series analysis

Tests cover:
1. aggregate_monthly() - grouping deals by month with various metrics
2. aggregate_by_sector() - sector-level aggregation
3. create_analysis_dataset() - merging VC data with CPU index
4. compute_rolling_stats() - rolling window calculations
5. get_monthly_metrics() - integration of loading and aggregation
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from cpu_index.analysis.vc_aggregator import (
    aggregate_monthly,
    aggregate_by_sector,
    create_analysis_dataset,
    compute_rolling_stats,
    get_monthly_metrics,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_vc_deals():
    """Sample VC deal DataFrame mimicking load_vc_deals() output."""
    return pd.DataFrame({
        'Deal ID': ['D001', 'D002', 'D003', 'D004', 'D005', 'D006'],
        'Companies': ['CompanyA', 'CompanyB', 'CompanyC', 'CompanyD', 'CompanyE', 'CompanyF'],
        'Deal Date': pd.to_datetime([
            '2024-01-15', '2024-01-20', '2024-02-10',
            '2024-02-25', '2024-03-05', '2024-03-15'
        ]),
        'Deal Size': [10.0, 25.0, 15.0, np.nan, 30.0, 20.0],
        'Deal Type': [
            'Seed Round', 'Early Stage VC', 'Later Stage VC',
            'Seed Round', 'Early Stage VC', 'Later Stage VC'
        ],
        'Stage': ['Seed', 'Early', 'Late', 'Seed', 'Early', 'Late'],
        'Primary Industry Sector': [
            'Energy', 'Energy', 'Materials',
            'Energy', 'Energy', 'Materials'
        ],
        'YearMonth': pd.to_datetime([
            '2024-01-15', '2024-01-20', '2024-02-10',
            '2024-02-25', '2024-03-05', '2024-03-15'
        ]).to_period('M'),
    })


@pytest.fixture
def sample_vc_deals_with_gap():
    """Sample VC deal DataFrame with a gap month (no deals in Feb)."""
    return pd.DataFrame({
        'Deal ID': ['D001', 'D002', 'D003', 'D004'],
        'Companies': ['CompanyA', 'CompanyB', 'CompanyC', 'CompanyD'],
        'Deal Date': pd.to_datetime([
            '2024-01-15', '2024-01-20', '2024-03-05', '2024-03-15'
        ]),
        'Deal Size': [10.0, 20.0, 30.0, 40.0],
        'Deal Type': [
            'Seed Round', 'Early Stage VC', 'Early Stage VC', 'Later Stage VC'
        ],
        'Stage': ['Seed', 'Early', 'Early', 'Late'],
        'Primary Industry Sector': ['Energy', 'Energy', 'Materials', 'Energy'],
        'YearMonth': pd.to_datetime([
            '2024-01-15', '2024-01-20', '2024-03-05', '2024-03-15'
        ]).to_period('M'),
    })


@pytest.fixture
def sample_monthly_aggregates():
    """Sample monthly aggregated VC metrics."""
    dates = pd.date_range('2024-01-01', periods=6, freq='MS')
    return pd.DataFrame({
        'deal_count': [5, 8, 6, 10, 7, 9],
        'total_amount': [100.0, 150.0, 120.0, 200.0, 140.0, 180.0],
        'median_amount': [20.0, 18.0, 22.0, 19.0, 21.0, 20.0],
        'mean_amount': [20.0, 18.75, 20.0, 20.0, 20.0, 20.0],
        'deals_with_size': [5, 8, 6, 10, 7, 9],
        'seed_count': [2, 3, 2, 4, 3, 3],
        'early_count': [2, 3, 2, 4, 2, 4],
        'late_count': [1, 2, 2, 2, 2, 2],
        'size_coverage_pct': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    }, index=dates)


@pytest.fixture
def sample_cpu_index_with_column():
    """Sample CPU index DataFrame with 'month' column."""
    return pd.DataFrame({
        'month': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01',
                                  '2024-04-01', '2024-05-01', '2024-06-01']),
        'normalized_index': [100.0, 105.0, 98.0, 110.0, 102.0, 108.0],
        'raw_ratio': [0.20, 0.21, 0.19, 0.22, 0.20, 0.21],
    })


@pytest.fixture
def sample_cpu_index_with_datetime_index():
    """Sample CPU index DataFrame with datetime index (no 'month' column)."""
    dates = pd.date_range('2024-01-01', periods=6, freq='MS')
    return pd.DataFrame({
        'normalized_index': [100.0, 105.0, 98.0, 110.0, 102.0, 108.0],
        'raw_ratio': [0.20, 0.21, 0.19, 0.22, 0.20, 0.21],
    }, index=dates)


# =============================================================================
# Tests for aggregate_monthly()
# =============================================================================


class TestAggregateMonthly:
    """Tests for monthly aggregation of VC deals."""

    def test_aggregate_monthly_basic(self, sample_vc_deals):
        """aggregate_monthly should group deals by month and compute metrics."""
        result = aggregate_monthly(sample_vc_deals)

        # Should have 3 months of data
        assert len(result) == 3

        # Check index is datetime
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == 'month'

    def test_aggregate_monthly_deal_count(self, sample_vc_deals):
        """aggregate_monthly should correctly count deals per month."""
        result = aggregate_monthly(sample_vc_deals)

        # January: D001, D002 = 2 deals
        # February: D003, D004 = 2 deals
        # March: D005, D006 = 2 deals
        assert result.loc['2024-01-01', 'deal_count'] == 2
        assert result.loc['2024-02-01', 'deal_count'] == 2
        assert result.loc['2024-03-01', 'deal_count'] == 2

    def test_aggregate_monthly_total_amount(self, sample_vc_deals):
        """aggregate_monthly should sum deal sizes correctly."""
        result = aggregate_monthly(sample_vc_deals)

        # January: 10 + 25 = 35
        # February: 15 + NaN = 15
        # March: 30 + 20 = 50
        assert result.loc['2024-01-01', 'total_amount'] == pytest.approx(35.0)
        assert result.loc['2024-02-01', 'total_amount'] == pytest.approx(15.0)
        assert result.loc['2024-03-01', 'total_amount'] == pytest.approx(50.0)

    def test_aggregate_monthly_median_amount(self, sample_vc_deals):
        """aggregate_monthly should compute median deal size."""
        result = aggregate_monthly(sample_vc_deals)

        # January: median of [10, 25] = 17.5
        # February: median of [15] = 15 (NaN excluded)
        # March: median of [30, 20] = 25
        assert result.loc['2024-01-01', 'median_amount'] == pytest.approx(17.5)
        assert result.loc['2024-02-01', 'median_amount'] == pytest.approx(15.0)
        assert result.loc['2024-03-01', 'median_amount'] == pytest.approx(25.0)

    def test_aggregate_monthly_stage_counts(self, sample_vc_deals):
        """aggregate_monthly should count deals by stage."""
        result = aggregate_monthly(sample_vc_deals)

        # January: Seed (D001), Early (D002) -> seed=1, early=1, late=0
        assert result.loc['2024-01-01', 'seed_count'] == 1
        assert result.loc['2024-01-01', 'early_count'] == 1
        assert result.loc['2024-01-01', 'late_count'] == 0

        # February: Late (D003), Seed (D004) -> seed=1, early=0, late=1
        assert result.loc['2024-02-01', 'seed_count'] == 1
        assert result.loc['2024-02-01', 'early_count'] == 0
        assert result.loc['2024-02-01', 'late_count'] == 1

    def test_aggregate_monthly_expected_columns(self, sample_vc_deals):
        """aggregate_monthly should return all expected columns."""
        result = aggregate_monthly(sample_vc_deals)

        expected_columns = [
            'deal_count', 'total_amount', 'median_amount', 'mean_amount',
            'deals_with_size', 'seed_count', 'early_count', 'late_count',
            'size_coverage_pct'
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_aggregate_monthly_size_coverage(self, sample_vc_deals):
        """aggregate_monthly should compute size coverage percentage."""
        result = aggregate_monthly(sample_vc_deals)

        # January: 2 deals, 2 with size = 100%
        # February: 2 deals, 1 with size = 50%
        # March: 2 deals, 2 with size = 100%
        assert result.loc['2024-01-01', 'size_coverage_pct'] == pytest.approx(100.0)
        assert result.loc['2024-02-01', 'size_coverage_pct'] == pytest.approx(50.0)
        assert result.loc['2024-03-01', 'size_coverage_pct'] == pytest.approx(100.0)

    def test_aggregate_monthly_fills_gap_months_with_zeros(self, sample_vc_deals_with_gap):
        """aggregate_monthly should fill months with no deals with zeros."""
        result = aggregate_monthly(sample_vc_deals_with_gap)

        # Should have 3 months: Jan, Feb (filled), Mar
        assert len(result) == 3

        # February should be filled with zeros
        assert result.loc['2024-02-01', 'deal_count'] == 0
        assert result.loc['2024-02-01', 'total_amount'] == 0
        assert result.loc['2024-02-01', 'seed_count'] == 0
        assert result.loc['2024-02-01', 'early_count'] == 0
        assert result.loc['2024-02-01', 'late_count'] == 0

    def test_aggregate_monthly_preserves_chronological_order(self, sample_vc_deals):
        """aggregate_monthly should return data in chronological order."""
        result = aggregate_monthly(sample_vc_deals)

        # Index should be sorted
        assert result.index.is_monotonic_increasing

    def test_aggregate_monthly_deals_with_size_count(self, sample_vc_deals):
        """aggregate_monthly should count deals that have size data."""
        result = aggregate_monthly(sample_vc_deals)

        # January: 2 deals, both have size
        # February: 2 deals, only 1 has size (D004 is NaN)
        # March: 2 deals, both have size
        assert result.loc['2024-01-01', 'deals_with_size'] == 2
        assert result.loc['2024-02-01', 'deals_with_size'] == 1
        assert result.loc['2024-03-01', 'deals_with_size'] == 2


class TestAggregateMonthlyEdgeCases:
    """Edge case tests for aggregate_monthly()."""

    def test_aggregate_monthly_single_deal(self):
        """aggregate_monthly should handle single deal DataFrame."""
        df = pd.DataFrame({
            'Deal ID': ['D001'],
            'Companies': ['CompanyA'],
            'Deal Date': pd.to_datetime(['2024-01-15']),
            'Deal Size': [50.0],
            'Deal Type': ['Seed Round'],
            'Stage': ['Seed'],
            'Primary Industry Sector': ['Energy'],
            'YearMonth': pd.to_datetime(['2024-01-15']).to_period('M'),
        })

        result = aggregate_monthly(df)

        assert len(result) == 1
        assert result.loc['2024-01-01', 'deal_count'] == 1
        assert result.loc['2024-01-01', 'total_amount'] == pytest.approx(50.0)
        assert result.loc['2024-01-01', 'seed_count'] == 1

    def test_aggregate_monthly_all_nan_sizes(self):
        """aggregate_monthly should handle all NaN deal sizes."""
        df = pd.DataFrame({
            'Deal ID': ['D001', 'D002'],
            'Companies': ['CompanyA', 'CompanyB'],
            'Deal Date': pd.to_datetime(['2024-01-15', '2024-01-20']),
            'Deal Size': [np.nan, np.nan],
            'Deal Type': ['Seed Round', 'Early Stage VC'],
            'Stage': ['Seed', 'Early'],
            'Primary Industry Sector': ['Energy', 'Energy'],
            'YearMonth': pd.to_datetime(['2024-01-15', '2024-01-20']).to_period('M'),
        })

        result = aggregate_monthly(df)

        assert result.loc['2024-01-01', 'deal_count'] == 2
        assert result.loc['2024-01-01', 'deals_with_size'] == 0
        assert result.loc['2024-01-01', 'size_coverage_pct'] == pytest.approx(0.0)
        # Total amount should be 0 (NaN summed)
        assert result.loc['2024-01-01', 'total_amount'] == pytest.approx(0.0)

    def test_aggregate_monthly_large_time_gap(self):
        """aggregate_monthly should fill large gaps between months."""
        df = pd.DataFrame({
            'Deal ID': ['D001', 'D002'],
            'Companies': ['CompanyA', 'CompanyB'],
            'Deal Date': pd.to_datetime(['2024-01-15', '2024-06-15']),
            'Deal Size': [10.0, 20.0],
            'Deal Type': ['Seed Round', 'Seed Round'],
            'Stage': ['Seed', 'Seed'],
            'Primary Industry Sector': ['Energy', 'Energy'],
            'YearMonth': pd.to_datetime(['2024-01-15', '2024-06-15']).to_period('M'),
        })

        result = aggregate_monthly(df)

        # Should have 6 months: Jan, Feb, Mar, Apr, May, Jun
        assert len(result) == 6

        # Months Feb-May should be filled with zeros
        for month in ['2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01']:
            assert result.loc[month, 'deal_count'] == 0


# =============================================================================
# Tests for aggregate_by_sector()
# =============================================================================


class TestAggregateBySector:
    """Tests for sector-level aggregation."""

    def test_aggregate_by_sector_basic(self, sample_vc_deals):
        """aggregate_by_sector should group deals by month and sector."""
        result = aggregate_by_sector(sample_vc_deals)

        # Should have rows for each month-sector combination
        assert 'YearMonth' in result.columns
        assert 'Primary Industry Sector' in result.columns
        assert 'deal_count' in result.columns
        assert 'total_amount' in result.columns

    def test_aggregate_by_sector_counts(self, sample_vc_deals):
        """aggregate_by_sector should correctly count deals per sector."""
        result = aggregate_by_sector(sample_vc_deals)

        # Filter for January Energy deals
        jan_energy = result[
            (result['YearMonth'] == pd.Timestamp('2024-01-01')) &
            (result['Primary Industry Sector'] == 'Energy')
        ]
        assert len(jan_energy) == 1
        assert jan_energy['deal_count'].values[0] == 2

    def test_aggregate_by_sector_timestamp_conversion(self, sample_vc_deals):
        """aggregate_by_sector should convert YearMonth to timestamp."""
        result = aggregate_by_sector(sample_vc_deals)

        # YearMonth should be datetime, not Period
        assert pd.api.types.is_datetime64_any_dtype(result['YearMonth'])


# =============================================================================
# Tests for create_analysis_dataset()
# =============================================================================


class TestCreateAnalysisDataset:
    """Tests for merging VC data with CPU index."""

    def test_create_analysis_dataset_basic(
        self, sample_monthly_aggregates, sample_cpu_index_with_column
    ):
        """create_analysis_dataset should merge VC and CPU data."""
        result = create_analysis_dataset(
            sample_monthly_aggregates,
            sample_cpu_index_with_column,
            cpu_column='normalized_index'
        )

        # Should contain both VC and CPU columns
        assert 'deal_count' in result.columns
        assert 'total_amount' in result.columns
        assert 'cpu_index' in result.columns

    def test_create_analysis_dataset_inner_join(
        self, sample_monthly_aggregates, sample_cpu_index_with_column
    ):
        """create_analysis_dataset should perform inner join on months."""
        result = create_analysis_dataset(
            sample_monthly_aggregates,
            sample_cpu_index_with_column
        )

        # Should have same number of rows as overlap
        assert len(result) == 6  # Both have 6 months

    def test_create_analysis_dataset_cpu_column_renamed(
        self, sample_monthly_aggregates, sample_cpu_index_with_column
    ):
        """create_analysis_dataset should rename CPU column to 'cpu_index'."""
        result = create_analysis_dataset(
            sample_monthly_aggregates,
            sample_cpu_index_with_column,
            cpu_column='normalized_index'
        )

        assert 'cpu_index' in result.columns
        assert 'normalized_index' not in result.columns

    def test_create_analysis_dataset_with_datetime_index(
        self, sample_monthly_aggregates, sample_cpu_index_with_datetime_index
    ):
        """create_analysis_dataset should handle CPU data with datetime index."""
        result = create_analysis_dataset(
            sample_monthly_aggregates,
            sample_cpu_index_with_datetime_index,
            cpu_column='normalized_index'
        )

        assert 'cpu_index' in result.columns
        assert len(result) == 6

    def test_create_analysis_dataset_alternative_cpu_column(
        self, sample_monthly_aggregates, sample_cpu_index_with_column
    ):
        """create_analysis_dataset should use specified CPU column."""
        result = create_analysis_dataset(
            sample_monthly_aggregates,
            sample_cpu_index_with_column,
            cpu_column='raw_ratio'
        )

        assert 'cpu_index' in result.columns
        # Should have raw_ratio values, not normalized_index
        assert result['cpu_index'].iloc[0] == pytest.approx(0.20)

    def test_create_analysis_dataset_missing_column_raises(
        self, sample_monthly_aggregates, sample_cpu_index_with_column
    ):
        """create_analysis_dataset should raise ValueError for missing column."""
        with pytest.raises(ValueError, match="not found in CPU index data"):
            create_analysis_dataset(
                sample_monthly_aggregates,
                sample_cpu_index_with_column,
                cpu_column='nonexistent_column'
            )

    def test_create_analysis_dataset_partial_overlap(self, sample_monthly_aggregates):
        """create_analysis_dataset should handle partial date overlap."""
        # CPU data only covers 3 months
        cpu_partial = pd.DataFrame({
            'month': pd.to_datetime(['2024-02-01', '2024-03-01', '2024-04-01']),
            'normalized_index': [105.0, 98.0, 110.0],
        })

        result = create_analysis_dataset(
            sample_monthly_aggregates,
            cpu_partial,
            cpu_column='normalized_index'
        )

        # Should only have 3 months (inner join)
        assert len(result) == 3


# =============================================================================
# Tests for compute_rolling_stats()
# =============================================================================


class TestComputeRollingStats:
    """Tests for rolling statistics computation."""

    def test_compute_rolling_stats_basic(self, sample_monthly_aggregates):
        """compute_rolling_stats should add rolling mean and std columns."""
        result = compute_rolling_stats(sample_monthly_aggregates, window=3)

        assert 'deal_count_rolling_mean' in result.columns
        assert 'deal_count_rolling_std' in result.columns
        assert 'total_amount_rolling_mean' in result.columns
        assert 'total_amount_rolling_std' in result.columns

    def test_compute_rolling_stats_window_calculation(self, sample_monthly_aggregates):
        """compute_rolling_stats should compute correct rolling values."""
        result = compute_rolling_stats(sample_monthly_aggregates, window=3)

        # First 2 rows should be NaN (window=3 needs 3 values)
        assert pd.isna(result['deal_count_rolling_mean'].iloc[0])
        assert pd.isna(result['deal_count_rolling_mean'].iloc[1])

        # Third row: mean of [5, 8, 6] = 6.33...
        assert result['deal_count_rolling_mean'].iloc[2] == pytest.approx(6.333, rel=0.01)

    def test_compute_rolling_stats_default_columns(self, sample_monthly_aggregates):
        """compute_rolling_stats should use default columns if not specified."""
        result = compute_rolling_stats(sample_monthly_aggregates, window=3)

        # Default columns are deal_count and total_amount
        assert 'deal_count_rolling_mean' in result.columns
        assert 'total_amount_rolling_mean' in result.columns

    def test_compute_rolling_stats_custom_columns(self, sample_monthly_aggregates):
        """compute_rolling_stats should use specified columns."""
        result = compute_rolling_stats(
            sample_monthly_aggregates,
            window=3,
            columns=['median_amount', 'seed_count']
        )

        assert 'median_amount_rolling_mean' in result.columns
        assert 'median_amount_rolling_std' in result.columns
        assert 'seed_count_rolling_mean' in result.columns
        assert 'seed_count_rolling_std' in result.columns

        # Should NOT have default columns
        assert 'deal_count_rolling_mean' not in result.columns

    def test_compute_rolling_stats_preserves_original(self, sample_monthly_aggregates):
        """compute_rolling_stats should preserve original columns."""
        result = compute_rolling_stats(sample_monthly_aggregates, window=3)

        # Original columns should be preserved
        assert 'deal_count' in result.columns
        assert 'total_amount' in result.columns
        assert 'median_amount' in result.columns

    def test_compute_rolling_stats_missing_column_ignored(self, sample_monthly_aggregates):
        """compute_rolling_stats should ignore columns not in DataFrame."""
        result = compute_rolling_stats(
            sample_monthly_aggregates,
            window=3,
            columns=['deal_count', 'nonexistent_column']
        )

        # Should still work with valid column
        assert 'deal_count_rolling_mean' in result.columns
        # Should not have columns for nonexistent
        assert 'nonexistent_column_rolling_mean' not in result.columns

    def test_compute_rolling_stats_window_12(self, sample_monthly_aggregates):
        """compute_rolling_stats with window=12 should produce NaN for small data."""
        result = compute_rolling_stats(sample_monthly_aggregates, window=12)

        # Only 6 months of data, window=12 means all should be NaN
        assert result['deal_count_rolling_mean'].isna().all()

    def test_compute_rolling_stats_does_not_modify_input(self, sample_monthly_aggregates):
        """compute_rolling_stats should not modify the input DataFrame."""
        original_cols = set(sample_monthly_aggregates.columns)

        compute_rolling_stats(sample_monthly_aggregates, window=3)

        # Original should be unchanged
        assert set(sample_monthly_aggregates.columns) == original_cols


# =============================================================================
# Tests for get_monthly_metrics() with mocked loader
# =============================================================================


class TestGetMonthlyMetrics:
    """Tests for get_monthly_metrics() integration function."""

    @patch('cpu_index.analysis.vc_aggregator.load_vc_deals')
    def test_get_monthly_metrics_calls_loader(self, mock_load, sample_vc_deals):
        """get_monthly_metrics should call load_vc_deals with correct args."""
        mock_load.return_value = sample_vc_deals

        get_monthly_metrics('/path/to/deals.csv', min_date='2020-01-01')

        mock_load.assert_called_once_with('/path/to/deals.csv', min_date='2020-01-01')

    @patch('cpu_index.analysis.vc_aggregator.load_vc_deals')
    def test_get_monthly_metrics_returns_aggregated(self, mock_load, sample_vc_deals):
        """get_monthly_metrics should return aggregated monthly data."""
        mock_load.return_value = sample_vc_deals

        result = get_monthly_metrics('/path/to/deals.csv')

        # Should return aggregated data
        assert 'deal_count' in result.columns
        assert 'total_amount' in result.columns
        assert isinstance(result.index, pd.DatetimeIndex)

    @patch('cpu_index.analysis.vc_aggregator.load_vc_deals')
    def test_get_monthly_metrics_default_min_date(self, mock_load, sample_vc_deals):
        """get_monthly_metrics should use default min_date."""
        mock_load.return_value = sample_vc_deals

        get_monthly_metrics('/path/to/deals.csv')

        # Default min_date is '2008-01-01'
        mock_load.assert_called_once_with('/path/to/deals.csv', min_date='2008-01-01')


# =============================================================================
# Integration-style tests (with mocked dependencies)
# =============================================================================


class TestVCAggregatorIntegration:
    """Integration tests for VC aggregator workflow."""

    def test_full_workflow(self, sample_vc_deals, sample_cpu_index_with_column):
        """Test full workflow: aggregate -> merge -> rolling stats."""
        # Step 1: Aggregate
        monthly = aggregate_monthly(sample_vc_deals)
        assert len(monthly) == 3

        # Step 2: Merge with CPU (need matching dates)
        cpu_data = pd.DataFrame({
            'month': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
            'normalized_index': [100.0, 105.0, 98.0],
        })
        merged = create_analysis_dataset(monthly, cpu_data)
        assert 'cpu_index' in merged.columns

        # Step 3: Rolling stats
        with_rolling = compute_rolling_stats(merged, window=2)
        assert 'deal_count_rolling_mean' in with_rolling.columns

    def test_empty_gap_months_not_affecting_analysis(self, sample_vc_deals_with_gap):
        """Test that zero-filled months work correctly in analysis."""
        monthly = aggregate_monthly(sample_vc_deals_with_gap)

        cpu_data = pd.DataFrame({
            'month': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
            'normalized_index': [100.0, 105.0, 98.0],
        })

        merged = create_analysis_dataset(monthly, cpu_data)

        # February should have 0 deals but still be in the merged data
        assert len(merged) == 3
        assert merged.loc['2024-02-01', 'deal_count'] == 0
