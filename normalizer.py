"""
BBD-style normalization for CPU indices

Implements Baker, Bloom & Davis (2016) methodology:
1. Scale each outlet's count by that outlet's total volume
2. Standardize to unit standard deviation per outlet
3. Average across outlets
4. Normalize to mean=100 over base period

This produces indices comparable across time and outlets.
"""

import numpy as np
from typing import Optional

import config


def scale_by_outlet_volume(outlet_data: dict) -> dict:
    """
    Step 1: Scale numerator counts by denominator (total volume).

    This converts raw counts to ratios, accounting for differences
    in total article volume across months.

    Args:
        outlet_data: Dict mapping months to {numerator, denominator}

    Returns:
        Dict mapping months to scaled ratios
    """
    result = {}
    for month, data in outlet_data.items():
        numerator = data.get("numerator", 0)
        denominator = data.get("denominator", 0)

        if denominator > 0:
            result[month] = numerator / denominator
        else:
            result[month] = 0.0

    return result


def standardize_to_unit_std_dev(series: dict) -> dict:
    """
    Step 2: Standardize series to have unit standard deviation.

    This ensures each outlet contributes equally to the aggregate,
    regardless of their baseline uncertainty levels.

    Args:
        series: Dict mapping months to values

    Returns:
        Dict mapping months to standardized values (mean=0, std=1)
    """
    if not series:
        return {}

    values = np.array(list(series.values()))
    mean = np.mean(values)
    std = np.std(values, ddof=0)  # Population std dev

    result = {}
    for month, value in series.items():
        # Use tolerance for std to handle floating-point precision issues
        if std > 1e-10:
            result[month] = (value - mean) / std
        else:
            # Constant series: all values centered at 0
            result[month] = 0.0

    return result


def average_across_outlets(outlet_series: dict) -> dict:
    """
    Step 3: Average standardized series across outlets.

    Args:
        outlet_series: Dict mapping outlet names to their standardized series
                      {outlet: {month: value}}

    Returns:
        Dict mapping months to averaged values
    """
    if not outlet_series:
        return {}

    # Collect all months
    all_months = set()
    for series in outlet_series.values():
        all_months.update(series.keys())

    # Average values for each month
    result = {}
    for month in sorted(all_months):
        values = [
            series[month]
            for series in outlet_series.values()
            if month in series
        ]
        if values:
            result[month] = np.mean(values)

    return result


def normalize_to_base_period(
    series: dict,
    base_start: str,
    base_end: str,
) -> dict:
    """
    Step 4: Normalize series to have mean=100 over base period.

    Args:
        series: Dict mapping months to values
        base_start: Start of base period (YYYY-MM format)
        base_end: End of base period (YYYY-MM format)

    Returns:
        Dict mapping months to normalized values (base period mean = 100)
    """
    if not series:
        return {}

    # Get values in base period
    base_values = [
        v for m, v in series.items()
        if base_start <= m <= base_end
    ]

    if not base_values:
        # No data in base period, return original
        return dict(series)

    base_mean = np.mean(base_values)

    result = {}
    for month, value in series.items():
        if base_mean != 0:
            result[month] = (value / base_mean) * 100
        else:
            result[month] = 0.0

    return result


def normalize_bbd_style(
    raw_counts_by_outlet: dict,
    base_start: str,
    base_end: str,
    exclude_outlets: Optional[list[str]] = None,
    skip_volume_scaling: bool = False,
    skip_standardization: bool = False,
) -> dict:
    """
    Full BBD-style normalization pipeline.

    This is the main entry point for computing a properly normalized CPU index.

    Args:
        raw_counts_by_outlet: Dict mapping outlet names to monthly data
                             {outlet: {month: {numerator, denominator}}}
        base_start: Start of base period for normalization
        base_end: End of base period for normalization
        exclude_outlets: List of outlet names to exclude (for ablation studies)
        skip_volume_scaling: Skip Step 1 (for ablation testing)
        skip_standardization: Skip Step 2 (for ablation testing)

    Returns:
        Dict mapping months to normalized index values (base mean = 100)
    """
    if not raw_counts_by_outlet:
        return {}

    # Apply outlet exclusions for ablation studies
    working_data = raw_counts_by_outlet
    if exclude_outlets:
        excluded_lower = {o.lower() for o in exclude_outlets}
        working_data = {
            outlet: data for outlet, data in raw_counts_by_outlet.items()
            if outlet.lower() not in excluded_lower
        }

    if not working_data:
        return {}

    # Step 1 & 2: Scale and standardize each outlet
    standardized_by_outlet = {}
    for outlet, monthly_data in working_data.items():
        if skip_volume_scaling:
            # Ablation: use raw numerator counts directly
            scaled = {m: d.get("numerator", 0) for m, d in monthly_data.items()}
        else:
            # Step 1: Scale by volume
            scaled = scale_by_outlet_volume(monthly_data)

        if skip_standardization:
            # Ablation: skip standardization
            standardized_by_outlet[outlet] = scaled
        else:
            # Step 2: Standardize to unit std dev
            standardized = standardize_to_unit_std_dev(scaled)
            standardized_by_outlet[outlet] = standardized

    # Step 3: Average across outlets
    averaged = average_across_outlets(standardized_by_outlet)

    # Step 4: Normalize to base period
    normalized = normalize_to_base_period(averaged, base_start, base_end)

    return normalized


def compute_outlet_level_cpu(
    raw_counts_by_outlet: dict,
    base_start: str,
    base_end: str,
) -> dict:
    """
    Compute outlet-level CPU indices (each outlet normalized independently).

    This is useful for robustness analysis - checking if the index
    is driven by one outlet or consistent across all.

    Args:
        raw_counts_by_outlet: Dict mapping outlet names to monthly data
        base_start: Start of base period
        base_end: End of base period

    Returns:
        Dict mapping outlet names to their individual index series
        {outlet: {month: normalized_value}}
    """
    result = {}

    for outlet, monthly_data in raw_counts_by_outlet.items():
        # Step 1: Scale by volume
        scaled = scale_by_outlet_volume(monthly_data)
        # Skip Step 2 (standardization) for individual outlets
        # Step 4: Normalize to base period (mean=100)
        normalized = normalize_to_base_period(scaled, base_start, base_end)
        result[outlet] = normalized

    return result


def filter_to_bbd_outlets(all_outlet_data: dict) -> dict:
    """
    Filter outlet data to include only BBD-approved newspapers.

    BBD (Baker, Bloom & Davis) uses 8 major newspapers for their
    Economic Policy Uncertainty index.

    Args:
        all_outlet_data: Dict mapping outlet names to their data

    Returns:
        Dict with only BBD outlets
    """
    bbd_outlets_lower = {o.lower() for o in config.BBD_OUTLETS}

    return {
        outlet: data
        for outlet, data in all_outlet_data.items()
        if outlet.lower() in bbd_outlets_lower
    }


def compute_correlation_matrix(outlet_indices: dict) -> dict:
    """
    Compute correlation matrix between outlet-level indices.

    Useful for checking if outlets move together or have divergent patterns.

    Args:
        outlet_indices: Dict mapping outlet names to their series
                       {outlet: {month: value}}

    Returns:
        Dict with correlation coefficients between each outlet pair
        {(outlet1, outlet2): correlation}
    """
    if len(outlet_indices) < 2:
        return {}

    # Align series to common months
    outlets = list(outlet_indices.keys())
    all_months = set()
    for series in outlet_indices.values():
        all_months.update(series.keys())
    all_months = sorted(all_months)

    # Build aligned arrays
    aligned = {}
    for outlet, series in outlet_indices.items():
        aligned[outlet] = [series.get(m, np.nan) for m in all_months]

    # Compute correlations
    result = {}
    for i, outlet1 in enumerate(outlets):
        for outlet2 in outlets[i + 1:]:
            arr1 = np.array(aligned[outlet1])
            arr2 = np.array(aligned[outlet2])

            # Only use months where both have data
            mask = ~np.isnan(arr1) & ~np.isnan(arr2)
            if mask.sum() >= 2:
                corr = np.corrcoef(arr1[mask], arr2[mask])[0, 1]
                result[(outlet1, outlet2)] = corr

    return result


def get_aggregate_statistics(normalized_index: dict) -> dict:
    """
    Compute summary statistics for a normalized index.

    Args:
        normalized_index: Dict mapping months to index values

    Returns:
        Dict with statistics: mean, std, min, max, etc.
    """
    if not normalized_index:
        return {"mean": None, "std": None, "min": None, "max": None}

    values = list(normalized_index.values())
    months = list(normalized_index.keys())

    min_idx = np.argmin(values)
    max_idx = np.argmax(values)

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "min_month": months[min_idx],
        "max_month": months[max_idx],
        "n_months": len(values),
    }
